package com.example.test1;

import static org.opencv.core.CvType.CV_32SC1;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_UNCHANGED;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.resize;

import android.content.Context;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.face.BasicFaceRecognizer;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.FisherFaceRecognizer;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.face.EigenFaceRecognizer;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getSimpleName();

    JavaCameraView javaCameraView;
    private CascadeClassifier faceDetector;


    // I choose to use EigenFaceRecognizer
    private EigenFaceRecognizer eigenFaceRecognizer;
    private ArrayList<Mat> images;
    private Mat labels;

    private LoaderCallbackInterface initCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    InputStream inputStream = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                    File cascadeClassifier = getDir("cascade", Context.MODE_PRIVATE);
                    File lbpClassifier = new File(cascadeClassifier, "lbpcascade_frontalface.xml");
                    FileOutputStream fos = null;

                    try {
                        fos = new FileOutputStream(lbpClassifier);
                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = inputStream.read(buffer)) != -1) {
                            fos.write(buffer, 0, bytesRead);
                        }
                        inputStream.close();
                        fos.close();

                        faceDetector = new CascadeClassifier(lbpClassifier.getAbsolutePath());

                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    javaCameraView.enableView();
                    break;
                }
                default:
                    super.onManagerConnected(status);
            }
        }
    };

    private Mat matRGB;
    private Mat matGrey;
    private int cameraId;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        javaCameraView = findViewById(R.id.javaCameraView);

        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, initCallback);
        } else
            initCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        javaCameraView.setCvCameraViewListener((CameraBridgeViewBase.CvCameraViewListener2) this);


        // Train the face with my images from raw folder
        try {
            trainFaceRecognizer();
            Log.i(TAG, "Fata antrenata");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    // Adds a face for training
    private void addFaceForTraining(int resourceID, int label) throws IOException {

        Mat image = Utils.loadResource(this, resourceID, Imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
        MatOfRect faceRects = new MatOfRect();
        faceDetector.detectMultiScale(image, faceRects);
        Rect[] faceArray = faceRects.toArray();

        if (faceArray.length > 0) {
            Mat face = new Mat(image, faceArray[0]);
            Size size = new Size(150, 150);
            Imgproc.resize(face, face, size);
            images.add(face);
            labels.push_back(new Mat(1, 1, CvType.CV_32SC1, new Scalar(label)));
        }

    }


    // Function for taining the face recognizer
    public void trainFaceRecognizer() throws IOException {

        eigenFaceRecognizer = EigenFaceRecognizer.create();
        images = new ArrayList<Mat>();
        labels = new Mat();


        // Add the images and labels from raw folder
        addFaceForTraining(R.raw.eu1, 1);
        addFaceForTraining(R.raw.eu2, 1);
        addFaceForTraining(R.raw.eu3, 1);
        addFaceForTraining(R.raw.eu4, 1);
        addFaceForTraining(R.raw.eu5, 1);
        addFaceForTraining(R.raw.eu6, 1);
        addFaceForTraining(R.raw.nueu1, 2);


        // Train the model
        eigenFaceRecognizer.train(images, labels);
    }


    @Override
    public void onCameraViewStarted(int width, int height) {
        matRGB = new Mat();
        matGrey = new Mat();
    }


    @Override
    public void onCameraViewStopped() {
        matRGB.release();
        matGrey.release();
    }



    boolean detectedAtLEastOneFace = false;
    boolean canDisplayMessage = true;

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        matRGB = inputFrame.rgba();
        matGrey = inputFrame.gray();

        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(matRGB, faces);

        for (Rect rect : faces.toArray()) {
            Imgproc.rectangle(matRGB, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(255, 0, 0));

            detectedAtLEastOneFace = true;
        }


        // Display identification messages 3 seconds apart
        if(detectedAtLEastOneFace && canDisplayMessage) {

            canDisplayMessage = false;

            new java.util.Timer().schedule(
                    new java.util.TimerTask() {
                        @Override
                        public void run() {

                            if (eigenFaceRecognizer != null) {

                                int[] label = new int[1];
                                double[] confidence = new double[1];
                                Imgproc.resize(matGrey, matGrey, new Size(150, 150));
                                eigenFaceRecognizer.predict(matGrey, label, confidence);
                                Log.i(TAG, "Found face with id " + label[0] + " with confidance of " + (Math.round((10000f - confidence[0]) / 100)) + "%");
                                canDisplayMessage = true;

                            }

                        }
                    }, 3000
            );


        }



        return matRGB;
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu, menu);
        return true;
    }


    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if (item.getItemId() == R.id.swap)
            swapCamera();
        return super.onOptionsItemSelected(item);
    }

    private void swapCamera() {
        cameraId = cameraId ^ 1;
        javaCameraView.disableView();
        javaCameraView.setCameraIndex(cameraId);
        javaCameraView.enableView();
    }

}