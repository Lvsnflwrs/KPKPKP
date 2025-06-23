package com.example.realtimefacerecognition.Face_Recognition;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Pair;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class TFLiteFaceRecognition implements FaceClassifier {

    private static final int OUTPUT_SIZE = 192;
    private static final float IMAGE_MEAN = 128.0f;
    private static final float IMAGE_STD = 128.0f;

    private boolean isModelQuantized;
    private int inputSize;

    private int[] intValues;
    private float[] embedding;

    private ByteBuffer imgData;
    private Interpreter tfLite;

    public HashMap<String, Recognition> registered = new HashMap<>();

    public void register(String name, Recognition rec) {
        registered.put(name, rec);
    }

    private TFLiteFaceRecognition() {}

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename) throws IOException {
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public static FaceClassifier create(
            final AssetManager assetManager,
            final String modelFilename,
            final int inputSize,
            final boolean isQuantized) throws IOException {

        final TFLiteFaceRecognition d = new TFLiteFaceRecognition();
        d.inputSize = inputSize;

        try {
            d.tfLite = new Interpreter(loadModelFile(assetManager, modelFilename));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }

        d.isModelQuantized = isQuantized;
        int numBytesPerChannel = isQuantized ? 1 : 4;
        d.imgData = ByteBuffer.allocateDirect(1 * d.inputSize * d.inputSize * 3 * numBytesPerChannel);
        d.imgData.order(ByteOrder.nativeOrder());
        d.intValues = new int[d.inputSize * d.inputSize];
        return d;
    }

    private Pair<String, Float> findNearest(float[] emb) {
        Pair<String, Float> ret = null;
        for (Map.Entry<String, Recognition> entry : registered.entrySet()) {
            final String name = entry.getKey();
            final float[] knownEmb = (float[]) entry.getValue().getEmbeeding();

            float distance = 0;
            for (int i = 0; i < emb.length; i++) {
                float diff = emb[i] - knownEmb[i];
                distance += diff * diff;
            }
            distance = (float) Math.sqrt(distance);
            if (ret == null || distance < ret.second) {
                ret = new Pair<>(name, distance);
            }
        }
        return ret;
    }

    @Override
    public Recognition recognizeImage(final Bitmap bitmap, boolean storeExtra) {
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                if (isModelQuantized) {
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else {
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        Object[] inputArray = {imgData};
        Map<Integer, Object> outputMap = new HashMap<>();

        embedding = new float[OUTPUT_SIZE];
        outputMap.put(0, new float[][] { embedding });

        tfLite.runForMultipleInputsOutputs(inputArray, outputMap);

        float distance = Float.MAX_VALUE;
        String id = "0";
        String label = "?";

        if (registered.size() > 0) {
            final Pair<String, Float> nearest = findNearest(embedding);
            if (nearest != null) {
                final String name = nearest.first;
                label = name;
                distance = nearest.second;
            }
        }

        Recognition rec = new Recognition(id, label, distance, new RectF());

        if (storeExtra) {
            rec.setEmbeeding(embedding);
        }

        return rec;
    }
}
