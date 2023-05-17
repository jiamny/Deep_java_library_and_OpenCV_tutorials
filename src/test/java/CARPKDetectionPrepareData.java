import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.util.JsonUtils;
import com.google.gson.reflect.TypeToken;
import com.jiamny.Object_detection.FeatureExtraction.FeatureComparisonExample;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;
import ai.djl.basicdataset.cv.PikachuDetection;

import ai.djl.training.dataset.Dataset;

import java.io.IOException;
import java.io.File;
import java.io.Reader;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static com.jiamny.Utils.ImageHelper.ndarrayToMat;
import static com.jiamny.Utils.ImageViewer.show;
import static com.jiamny.Utils.HelperFunctions.printListObjects;

public class CARPKDetectionPrepareData {
    private static final Logger logger = LoggerFactory.getLogger(CARPKDetectionPrepareData.class);
    private static Random rng = new Random(12345);

    private static void resizeImages() {
        String filename = "";

        final File folder = new File("data/carpk");

        for (final File fileEntry : folder.listFiles()) {

            if (fileEntry.isDirectory()) {
            } else {
                if( fileEntry.getName().contains(".jpg")) {
                    logger.info(String.valueOf(fileEntry.getName()));
                    logger.info(String.valueOf(fileEntry));
                    Mat oimg = Imgcodecs.imread(String.valueOf(fileEntry), Imgcodecs.IMREAD_COLOR);
                    Imgproc.resize(oimg, oimg, new Size(256, 256));
                    Imgcodecs.imwrite("data/carpk_256/" + String.valueOf(fileEntry.getName()), oimg);
                } else {
                    logger.info("TXT: " + String.valueOf(fileEntry.getName()));
                }
            }
        }

        filename = "data/images/carpk_test.jpg";
        Mat img = Imgcodecs.imread(filename, Imgcodecs.IMREAD_COLOR);
        Imgproc.resize(img, img, new Size(256, 256));
        Imgcodecs.imwrite("data/images/carpk_256_test.jpg", img);
    }

    private static void checkImages() {
        NDManager manager = NDManager.newBaseManager();
        String filename = "/home/stree/.djl.ai/cache/repo/dataset/cv/ai/djl/basicdataset/pikachu/1.0/train/img_0.jpg";
        filename = "data/carpk/20161225_TPZ_00095.jpg";
        Mat img = Imgcodecs.imread(filename, Imgcodecs.IMREAD_COLOR);
        HighGui.imshow( "image", img);
        HighGui.waitKey(0);

        logger.info("H: " + Float.toString(img.height()));
        logger.info("W: " + Float.toString(img.width()));

        Imgproc.resize(img, img, new Size(256, 256));
        HighGui.imshow( "image_resize", img);
        HighGui.waitKey(0);

        logger.info("H: " + Float.toString(img.height()));
        logger.info("W: " + Float.toString(img.width()));

        double[] v = {0.49296875, 0.7930555555555555, 0.55078125, 0.8361111111111111};
        Point tl = new Point(v[0]*256, v[1]*256);
        Point br = new Point(v[2]*256, v[3]*256);

        Scalar color = new Scalar(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256));
        Imgproc.rectangle(img, tl, br, color, 2);
        HighGui.imshow( "bound_resized", img);
        HighGui.waitKey(0);

        double[] v2 = {0.71171875, 0.9625, 0.76953125, 0.9986111111111111};
        Point tl2 = new Point(v2[0]*256, v2[1]*256);
        Point br2 = new Point(v2[2]*256, v2[3]*256);

        Scalar color2 = new Scalar(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256));
        Imgproc.rectangle(img, tl2, br2, color2, 2);
        HighGui.imshow( "bound_resized2", img);
        HighGui.waitKey(0);
    }

    public static void main(String[] args)  throws IOException {
        System.load("/usr/local/share/java/opencv4/libopencv_java470.so");

        // resizeImages();

        NDManager manager = NDManager.newBaseManager();
        /*
        NDArray array = manager.arange(18f).reshape(2, 9);
        array.split(new long[] {2,4,5}, 1).forEach(System.out::println);

        System.out.println("================================");
        array.split(new long[] {3}, 1).forEach(System.out::println);
         */

        //checkImages();
        Path usagePath = Paths.get("data/carpk");
        Path indexFile = usagePath.resolve("train_index.file");

        System.out.println(indexFile);
        try (Reader reader = Files.newBufferedReader(indexFile)) {
            Type mapType = new TypeToken<Map<String, List<List<Float>>>>() {}.getType();
            Map<String, List<List<Float>>> metadata = JsonUtils.GSON.fromJson(reader, mapType);
            for (Map.Entry<String, List<List<Float>>> entry : metadata.entrySet()) {
                String imgName = entry.getKey();
                //imagePaths.add(usagePath.resolve(imgName));
                System.out.println(usagePath.resolve(imgName));
                File dest = new File("data/CARPK/train/" + imgName);
                System.out.println("dest: " + dest.getAbsolutePath());
                //Files.copy(usagePath.resolve(imgName), dest.toPath());

                Mat oimg = Imgcodecs.imread(String.valueOf(usagePath.resolve(imgName)), Imgcodecs.IMREAD_COLOR);
                Imgproc.resize(oimg, oimg, new Size(256, 256));
                Imgcodecs.imwrite("data/CARPK/train/" + String.valueOf(imgName), oimg);

                List<List<Float>> singleLabels = entry.getValue();
                System.out.println("singleLabels: " + singleLabels.size());
                for (List<Float> label : singleLabels){

                    //printListObjects(Arrays.asList(label.toArray()));

                    long objectClass = label.get(4).longValue();
                    Rectangle objectLocation =
                            new Rectangle(
                                    new ai.djl.modality.cv.output.Point(label.get(5), label.get(6)), label.get(7), label.get(8));
                    //labels.add(objectClass, objectLocation);
                    //break;
                }
            }
        }

        System.exit(0);
    }
}
