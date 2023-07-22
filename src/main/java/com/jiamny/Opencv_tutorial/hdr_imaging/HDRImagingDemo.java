package com.jiamny.Opencv_tutorial.hdr_imaging;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.photo.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static com.jiamny.Utils.ImageViewer.show;

class HDRImaging {
    public void loadExposureSeq(String path, List<Mat> images, List<Float> times) {
        path += "/";

        List<String> lines;
        try {
            lines = Files.readAllLines(Paths.get(path + "list.txt"));

            for (String line : lines) {
                String[] splitStr = line.split("\\s+");
                if (splitStr.length == 2) {
                    String name = splitStr[0];
                    Mat img = Imgcodecs.imread(path + name);
                    images.add(img);
                    float val = Float.parseFloat(splitStr[1]);
                    times.add(1 / val);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void run(String[] args) {
        String path = args.length > 0 ? args[0] : "data/opencv_extra-4.x/testdata/cv/hdr/exposures";
        if (path.isEmpty()) {
            System.out.println("Path is empty. Use the directory that contains images and exposure times.");
            System.exit(0);
        }

        //! [Load images and exposure times]
        List<Mat> images = new ArrayList<>();
        List<Float> times = new ArrayList<>();
        loadExposureSeq(path, images, times);
        //! [Load images and exposure times]

        //! [Estimate camera response]
        Mat response = new Mat();
        CalibrateDebevec calibrate = Photo.createCalibrateDebevec();
        Mat matTimes = new Mat(times.size(), 1, CvType.CV_32F);
        float[] arrayTimes = new float[(int) (matTimes.total() * matTimes.channels())];
        for (int i = 0; i < times.size(); i++) {
            arrayTimes[i] = times.get(i);
        }
        matTimes.put(0, 0, arrayTimes);
        calibrate.process(images, response, matTimes);
        //! [Estimate camera response]

        //! [Make HDR image]
        Mat hdr = new Mat();
        MergeDebevec mergeDebevec = Photo.createMergeDebevec();
        mergeDebevec.process(images, hdr, matTimes);
        //! [Make HDR image]

        //! [Tonemap HDR image]
        Mat ldr = new Mat();
        Tonemap tonemap = Photo.createTonemap(2.2f);
        tonemap.process(hdr, ldr);
        //! [Tonemap HDR image]

        //! [Perform exposure fusion]
        Mat fusion = new Mat();
        MergeMertens mergeMertens = Photo.createMergeMertens();
        mergeMertens.process(images, fusion);
        //! [Perform exposure fusion]

        //! [Write results]
        Core.multiply(fusion, new Scalar(255, 255, 255), fusion);
        Core.multiply(ldr, new Scalar(255, 255, 255), ldr);

        Imgcodecs.imwrite("fusion.png", fusion);
        Mat img = Imgcodecs.imread("fusion.png", Imgcodecs.IMREAD_COLOR);
        HighGui.imshow( "fusion", img);
        HighGui.waitKey(0);

        Imgcodecs.imwrite("ldr.png", ldr);
        img = Imgcodecs.imread("ldr.png", Imgcodecs.IMREAD_COLOR);
        HighGui.imshow("ldr", img);
        HighGui.waitKey(0);

        Imgcodecs.imwrite("hdr.png", hdr);
        img = Imgcodecs.imread("hdr.png", Imgcodecs.IMREAD_COLOR);
        HighGui.imshow("hdr", img);
        HighGui.waitKey(0);
        //! [Write results]

        System.exit(0);
    }
}

public class HDRImagingDemo {
    static {
        // Load the native OpenCV library
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");
    }

    public static void main(String[] args) {
        new HDRImaging().run(args);
    }
}
