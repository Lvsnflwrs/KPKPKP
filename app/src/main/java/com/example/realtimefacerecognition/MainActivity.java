package com.example.realtimefacerecognition;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.Dialog;
import android.app.Fragment;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Surface;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import com.example.realtimefacerecognition.Drawing.BorderedText;
import com.example.realtimefacerecognition.Drawing.MultiBoxTracker;
import com.example.realtimefacerecognition.Drawing.OverlayView;
import com.example.realtimefacerecognition.Face_Recognition.FaceClassifier;
import com.example.realtimefacerecognition.Face_Recognition.TFLiteFaceRecognition;
import com.example.realtimefacerecognition.LiveFeed.CameraConnectionFragment;
import com.example.realtimefacerecognition.LiveFeed.ImageUtils;
import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import java.util.concurrent.TimeUnit; // Import for OkHttpClient timeout

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;


public class MainActivity extends AppCompatActivity implements ImageReader.OnImageAvailableListener{
    Handler handler;
    private Matrix frameToCropTransform;
    private int sensorOrientation;
    private Matrix cropToFrameTransform;

    private static final boolean MAINTAIN_ASPECT = false;
    private static final float TEXT_SIZE_DIP = 10;
    OverlayView trackingOverlay;
    private BorderedText borderedText;
    private MultiBoxTracker tracker;
    private Integer useFacing = null;
    private static final String KEY_USE_FACING = "use_facing";
    private static final int CROP_SIZE = 1000;
    private static final int TF_OD_API_INPUT_SIZE2 = 112;
    private String recognizedLabel = null;
    private double recognizedDistance = 0.0;
    private long lastRecognitionTime = 0;


    //    //TODO declare face detector
    FaceDetector detector;

//    //TODO declare face recognizer
    private FaceClassifier faceClassifier;

    boolean registerFace = false;

    private OkHttpClient client;

    private WebSocket webSocket;

    private static final String WEBSOCKET_URL = "ws://10.60.230.171:3000";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        handler = new Handler();

        //TODO handling permissions
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED || checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_DENIED){
                String[] permission = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE};
                requestPermissions(permission, 121);
            }
        }

        Intent intent = getIntent();
        useFacing = intent.getIntExtra(KEY_USE_FACING, CameraCharacteristics.LENS_FACING_BACK);

        //TODO show live camera footage
        handler.postDelayed(() -> setFragment(), 300);
//        setFragment();

        //TODO initialize the tracker to draw rectangles
        tracker = new MultiBoxTracker(this);

        //TODO initalize face detector
        // Multiple object detection in static images
        FaceDetectorOptions highAccuracyOpts =
                new FaceDetectorOptions.Builder()
                        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                        .build();
        detector = FaceDetection.getClient(highAccuracyOpts);

        //TODO initialize FACE Recognition
        try {
            faceClassifier =
                    TFLiteFaceRecognition.create(
                            getAssets(),
                            "mobile_face_net.tflite",
                            TF_OD_API_INPUT_SIZE2,
                            false);

        } catch (final IOException e) {
            e.printStackTrace();
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

        findViewById(R.id.imageView4).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                registerFace = true;
            }
        });



        findViewById(R.id.imageView3).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                switchCamera();
            }
        });
        initWebSocket(); // Initialize WebSocket connection
    }



    private void initWebSocket() {
        client = new OkHttpClient.Builder()
                .readTimeout(0, TimeUnit.MILLISECONDS) // Disable read timeout for WebSockets
                .build();

        Request request = new Request.Builder().url(WEBSOCKET_URL).build();
        webSocket = client.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(@NonNull WebSocket webSocket, @NonNull Response response) {
                super.onOpen(webSocket, response);
                Log.d("WebSocket", "Connected to WebSocket server!");
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Connected to server!", Toast.LENGTH_SHORT).show());
            }

            @Override
            public void onMessage(@NonNull WebSocket webSocket, @NonNull String text) {
                super.onMessage(webSocket, text);
                Log.d("WebSocket", "Receiving: " + text);
                try {
                    JSONObject jsonResponse = new JSONObject(text);
                    String status = jsonResponse.optString("status");
                    String type = jsonResponse.optString("type");

                    if ("recognize_face".equals(type)) {
                        boolean match = jsonResponse.getBoolean("match");
                        String name = jsonResponse.getString("name");
                        double distance = jsonResponse.getDouble("distance");

                        if (match) {
                            recognizedLabel = name;
                            recognizedDistance = distance;
                        } else {
                            recognizedLabel = "Unknown";
                            recognizedDistance = 0.0;
                        }
                        lastRecognitionTime = System.currentTimeMillis();
//                        runOnUiThread(() -> {
//                            String message = match
//                                    ? "Server recognized: " + recognizedLabel + " (Distance: " + String.format("%.2f", recognizedDistance) + ")"
//                                    : "Face not recognized.";
//                            Toast.makeText(MainActivity.this, message, Toast.LENGTH_SHORT).show();
//                        });
                    }

                    else if ("success".equals(status) && "recognition_result".equals(type)) {
                        String user = jsonResponse.optString("user");
                        double confidence = jsonResponse.optDouble("confidence");
                        Log.d("FaceRecognition", "Server Recognition: " + user + " (Confidence: " + String.format("%.2f", confidence) + ")");

                        runOnUiThread(() ->
                                Toast.makeText(MainActivity.this, "Server recognized: " + user, Toast.LENGTH_SHORT).show()
                        );
                    }

                    else if ("success".equals(status) && jsonResponse.has("message")) {
                        String message = jsonResponse.optString("message");
                        runOnUiThread(() ->
                                Toast.makeText(MainActivity.this, message, Toast.LENGTH_SHORT).show()
                        );
                    }

                } catch (JSONException e) {
                    Log.e("WebSocket", "Error parsing JSON message: " + e.getMessage());
                }
            }

            @Override
            public void onMessage(@NonNull WebSocket webSocket, @NonNull ByteString bytes) {
                super.onMessage(webSocket, bytes);
                Log.d("WebSocket", "Receiving bytes: " + bytes.hex());
            }

            @Override
            public void onClosing(@NonNull WebSocket webSocket, int code, @NonNull String reason) {
                super.onClosing(webSocket, code, reason);
                Log.d("WebSocket", "Closing: " + code + " / " + reason);
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "Disconnected from server.", Toast.LENGTH_SHORT).show());
            }

            @Override
            public void onFailure(@NonNull WebSocket webSocket, @NonNull Throwable t, Response response) {
                super.onFailure(webSocket, t, response);
                Log.e("WebSocket", "Error: " + t.getMessage(), t);
                runOnUiThread(() -> Toast.makeText(MainActivity.this, "WebSocket Error: " + t.getMessage(), Toast.LENGTH_LONG).show());
            }
        });
    }

    private void sendWebSocketMessage(String message) {
        if (webSocket != null) {
            webSocket.send(message);
            Log.d("WebSocket", "Sending: " + message);
        } else {
            Log.e("WebSocket", "WebSocket not initialized.");
            Toast.makeText(this, "WebSocket not connected. Please restart app.", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if(requestCode == 121 && grantResults[0] == PackageManager.PERMISSION_GRANTED){
            setFragment();
        }
    }

    //TODO fragment which show live footage from camera
    int previewHeight = 0,previewWidth = 0;
    protected void setFragment() {
        final CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        String cameraId = null;
        try {
            cameraId = manager.getCameraIdList()[useFacing];
        } catch (CameraAccessException | ArrayIndexOutOfBoundsException e) {
            e.printStackTrace();
            return;
        }

        Fragment fragment;
        CameraConnectionFragment camera2Fragment =
                CameraConnectionFragment.newInstance(
                        new CameraConnectionFragment.ConnectionCallback() {
                            @Override
                            public void onPreviewSizeChosen(final Size size, final int rotation) {
                                previewHeight = size.getHeight();
                                previewWidth = size.getWidth();

                                final float textSizePx =
                                        TypedValue.applyDimension(
                                                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
                                borderedText = new BorderedText(textSizePx);
                                borderedText.setTypeface(Typeface.MONOSPACE);


                                int cropSize = CROP_SIZE;

                                previewWidth = size.getWidth();
                                previewHeight = size.getHeight();

                                sensorOrientation = rotation - getScreenOrientation();

                                rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888);
                                croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Bitmap.Config.ARGB_8888);

                                frameToCropTransform =
                                        ImageUtils.getTransformationMatrix(
                                                previewWidth, previewHeight,
                                                cropSize, cropSize,
                                                sensorOrientation, MAINTAIN_ASPECT);

                                cropToFrameTransform = new Matrix();
                                frameToCropTransform.invert(cropToFrameTransform);

                                trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
                                trackingOverlay.addCallback(
                                        new OverlayView.DrawCallback() {
                                            @Override
                                            public void drawCallback(final Canvas canvas) {
                                                tracker.draw(canvas);
                                                Log.d("tryDrawRect","inside draw");
                                            }
                                        });
                                tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
                            }
                        },
                        this,
                        R.layout.camera_fragment,
//                        new Size(480, 480));
                        new Size(640, 480));

        camera2Fragment.setCamera(cameraId);
        fragment = camera2Fragment;
        getFragmentManager().beginTransaction().replace(R.id.container, fragment).commit();
    }


    //TODO getting frames of live camera footage and passing them to model
    private boolean isProcessingFrame = false;
    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private Bitmap rgbFrameBitmap;
    Bitmap croppedBitmap;
    @Override
    public void onImageAvailable(ImageReader reader) {
        if (previewWidth == 0 || previewHeight == 0) {
            return;
        }
        if (rgbBytes == null) {
            rgbBytes = new int[previewWidth * previewHeight];
        }
        try {
            final Image image = reader.acquireLatestImage();

            if (image == null) {
                return;
            }

            if (isProcessingFrame) {
                image.close();
                return;
            }
            isProcessingFrame = true;
            final Image.Plane[] planes = image.getPlanes();
            fillBytes(planes, yuvBytes);
            yRowStride = planes[0].getRowStride();
            final int uvRowStride = planes[1].getRowStride();
            final int uvPixelStride = planes[1].getPixelStride();

            imageConverter =
                    new Runnable() {
                        @Override
                        public void run() {
                            ImageUtils.convertYUV420ToARGB8888(
                                    yuvBytes[0],
                                    yuvBytes[1],
                                    yuvBytes[2],
                                    previewWidth,
                                    previewHeight,
                                    yRowStride,
                                    uvRowStride,
                                    uvPixelStride,
                                    rgbBytes);
                        }
                    };

            postInferenceCallback =
                    new Runnable() {
                        @Override
                        public void run() {
                            image.close();
                            isProcessingFrame = false;
                        }
                    };

            performFaceDetection();

        } catch (final Exception e) {
            Log.d("tryError",e.getMessage()+"abc ");
            return;
        }

    }

    protected void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        // Because of the variable row stride it's not possible to know in
        // advance the actual necessary dimensions of the yuv planes.
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }
    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    List<FaceClassifier.Recognition> mappedRecognitions;

    //TODO Perform face detection
    public void performFaceDetection(){
        imageConverter.run();
        rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

        new Handler().post(new Runnable() {
            @Override
            public void run() {
                mappedRecognitions = new ArrayList<>();
                InputImage image = InputImage.fromBitmap(croppedBitmap,0);
                detector.process(image)
                        .addOnSuccessListener(
                                        new OnSuccessListener<List<Face>>() {
                                            @Override
                                            public void onSuccess(List<Face> faces) {

                                                for(Face face:faces) {
                                                    final Rect bounds = face.getBoundingBox();
                                                    performFaceRecognition(face,croppedBitmap);
                                                }
                                                registerFace = false;
                                                tracker.trackResults(mappedRecognitions, 10);
                                                trackingOverlay.postInvalidate();
                                                postInferenceCallback.run();

                                            }
                                        })
                        .addOnFailureListener(
                                        new OnFailureListener() {
                                            @Override
                                            public void onFailure(@NonNull Exception e) {
                                                // Task failed with an exception
                                                Log.e("FaceDetection", "Face detection failed: " + e.getMessage());
                                                postInferenceCallback.run(); // Ensure frame is closed even on failure
                                            }
                                        });



            }
        });
    }

    //TODO perform face recognition
    public void performFaceRecognition(Face face,Bitmap input){
        //TODO crop the face
        Rect bounds = face.getBoundingBox();
        if(bounds.top<0){
            bounds.top = 0;
        }
        if(bounds.left<0){
            bounds.left = 0;
        }
        if(bounds.left+bounds.width()>input.getWidth()){
            bounds.right = input.getWidth()-1;
        }
        if(bounds.top+bounds.height()>input.getHeight()){
            bounds.bottom = input.getHeight()-1;
        }

        Bitmap crop = Bitmap.createBitmap(input,
                bounds.left,
                bounds.top,
                bounds.width(),
                bounds.height());
        crop = Bitmap.createScaledBitmap(crop,TF_OD_API_INPUT_SIZE2,TF_OD_API_INPUT_SIZE2,false);
        final FaceClassifier.Recognition result = faceClassifier.recognizeImage(crop, registerFace);
        String title = "Unknown";
        float confidence = 0;
        if (result != null) {
            if (registerFace) {
                registerFaceDialogue(crop, result);
            } else {
                float[] embeddingArray = ((TFLiteFaceRecognition) faceClassifier).getLastEmbedding();

                if (embeddingArray != null && webSocket != null) {
                    try {
                        JSONArray jsonEmbedding = new JSONArray();
                        for (float val : embeddingArray) {
                            jsonEmbedding.put(val);
                        }

                        JSONObject jsonRequest = new JSONObject();
                        jsonRequest.put("type", "recognize_face");
                        jsonRequest.put("embedding", jsonEmbedding);
                        sendWebSocketMessage(jsonRequest.toString());
                    } catch (JSONException e) {
                        Log.e("WebSocket", "Error creating recognition request JSON: " + e.getMessage());
                    }
                } else {
                    Log.w("WebSocket", "Embedding is null or WebSocket is null");
                }

                if (result.getDistance() < 0.75f) {
                    confidence = result.getDistance();
                    title = result.getTitle();
                }

                if (System.currentTimeMillis() - lastRecognitionTime < 1000) {
                    title = recognizedLabel;
                    confidence = (float) recognizedDistance;
                }
            }
        }

        RectF location = new RectF(bounds);
        if (bounds != null) {
            if(useFacing == CameraCharacteristics.LENS_FACING_BACK) {
                location.right = input.getWidth() - location.right;
                location.left = input.getWidth() - location.left;
            }
            cropToFrameTransform.mapRect(location);
            FaceClassifier.Recognition recognition = new FaceClassifier.Recognition(face.getTrackingId()+"",title,confidence,location);
            mappedRecognitions.add(recognition);
        }

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        //detector.close();
    }

    //TODO register face dialogue
    private void registerFaceDialogue(Bitmap croppedFace, FaceClassifier.Recognition rec) {
        Dialog dialog = new Dialog(this);
        dialog.setContentView(R.layout.register_face_dialogue);
        ImageView ivFace = dialog.findViewById(R.id.dlg_image);
        EditText nameEd = dialog.findViewById(R.id.dlg_input);
        Button register = dialog.findViewById(R.id.button2);
        ivFace.setImageBitmap(croppedFace);
        register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String name = nameEd.getText().toString();
                if (name.isEmpty()) {
                    nameEd.setError("Enter Name");
                    return;
                }
                // Send registration data to WebSocket server
                if (webSocket != null) {
                    try {
                        JSONObject jsonRequest = new JSONObject();
                        jsonRequest.put("type", "insert_face");
                        jsonRequest.put("name", name);

                        // Konversi embedding float[] ke JSONArray
                        float[] emb = (float[]) rec.getEmbedding();
                        JSONArray embeddingJson = new JSONArray();
                        for (float val : emb) {
                            embeddingJson.put(val);
                        }

                        jsonRequest.put("embedding", embeddingJson);

                        sendWebSocketMessage(jsonRequest.toString());
                    } catch (JSONException e) {
                        Log.e("WebSocket", "Error creating registration request JSON: " + e.getMessage());
                    }
                }

                faceClassifier.register(name, rec);
                Toast.makeText(MainActivity.this, "Face Registered Successfully", Toast.LENGTH_SHORT).show();
                dialog.dismiss();
            }
        });
        dialog.show();
    }

    //TODO switch camera
    public void switchCamera() {

        Intent intent = getIntent();

        if (useFacing == CameraCharacteristics.LENS_FACING_FRONT) {
            useFacing = CameraCharacteristics.LENS_FACING_BACK;
        } else {
            useFacing = CameraCharacteristics.LENS_FACING_FRONT;
        }

        intent.putExtra(KEY_USE_FACING, useFacing);
        intent.addFlags(Intent.FLAG_ACTIVITY_NO_ANIMATION);

        restartWith(intent);

    }

    private void restartWith(Intent intent) {
        finish();
        overridePendingTransition(0, 0);
        startActivity(intent);
        overridePendingTransition(0, 0);
    }
}