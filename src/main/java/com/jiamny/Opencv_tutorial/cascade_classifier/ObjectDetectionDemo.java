package com.jiamny.Opencv_tutorial.cascade_classifier;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

import javax.swing.*;
import java.awt.image.BufferedImage;
import java.util.List;

class ObjectDetection {
    public void detectAndDisplay(Mat frame, CascadeClassifier faceCascade, CascadeClassifier eyesCascade) {
        Mat frameGray = new Mat();
        Imgproc.cvtColor(frame, frameGray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(frameGray, frameGray);

        // -- Detect faces
        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(frameGray, faces);

        List<Rect> listOfFaces = faces.toList();
        for (Rect face : listOfFaces) {
            Point center = new Point(face.x + face.width / 2, face.y + face.height / 2);
            Imgproc.ellipse(frame, center, new Size(face.width / 2, face.height / 2), 0, 0, 360,
                    new Scalar(255, 0, 255));

            Mat faceROI = frameGray.submat(face);

            // -- In each face, detect eyes
            MatOfRect eyes = new MatOfRect();
            eyesCascade.detectMultiScale(faceROI, eyes);

            List<Rect> listOfEyes = eyes.toList();
            for (Rect eye : listOfEyes) {
                Point eyeCenter = new Point(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);
                int radius = (int) Math.round((eye.width + eye.height) * 0.25);
                Imgproc.circle(frame, eyeCenter, radius, new Scalar(255, 0, 0), 4);
            }
        }

        //-- Show what you got
        HighGui.imshow("Capture - Face detection", frame);
    }

    public void run(String[] args) {
        String filenameFaceCascade = args.length > 2 ? args[0] : "models/Opencv_tutorial/haarcascade_frontalface_alt.xml";
        String filenameEyesCascade = args.length > 2 ? args[1] : "models/Opencv_tutorial/haarcascade_eye_tree_eyeglasses.xml";
        int cameraDevice = args.length > 2 ? Integer.parseInt(args[2]) : 0;

        CascadeClassifier faceCascade = new CascadeClassifier();
        CascadeClassifier eyesCascade = new CascadeClassifier();

        if (!faceCascade.load(filenameFaceCascade)) {
            System.err.println("--(!)Error loading face cascade: " + filenameFaceCascade);
            System.exit(0);
        }
        if (!eyesCascade.load(filenameEyesCascade)) {
            System.err.println("--(!)Error loading eyes cascade: " + filenameEyesCascade);
            System.exit(0);
        }

        boolean useCamera = false;

        if (useCamera) {
            VideoCapture capture = new VideoCapture(cameraDevice);
            if (!capture.isOpened()) {
                System.err.println("--(!)Error opening video capture");
                System.exit(0);
            }

            Mat frame = new Mat();
            while (capture.read(frame)) {
                if (frame.empty()) {
                    System.err.println("--(!) No captured frame -- Break!");
                    break;
                }
                //-- 3. Apply the classifier to the frame
                detectAndDisplay(frame, faceCascade, eyesCascade);

                if (HighGui.waitKey(10) == 27) {
                    break;// escape
                }
            }
        } else {
            String f = "data/videos/Megamind.avi";
            Mat frame = new Mat();
            VideoCapture capture = new VideoCapture(f);
            if (capture.isOpened()) {
                while (true) {
                    capture.read(frame);
                    if (!frame.empty()) {
                        detectAndDisplay(frame, faceCascade, eyesCascade);

                        if (HighGui.waitKey(200) == 27) {
                            break;// escape
                        }
                    } else {
                        System.out.println(" Frame not captured or video has finished");
                        System.exit(0);
                    }
                }
            } else {
                System.out.println("Couldn't open video file.");
            }
        }

        System.exit(0);
    }
}

public class ObjectDetectionDemo {
    static {
        // Load the native OpenCV library
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");
    }

    public static void main(String[] args) {
        new ObjectDetection().run(args);
    }
}
