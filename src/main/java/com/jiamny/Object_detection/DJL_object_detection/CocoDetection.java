package com.jiamny.Object_detection.DJL_object_detection;

import ai.djl.Application;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.jiamny.Utils.ImageHelper.mat2DjlImage;
import static com.jiamny.Utils.ImageHelper.ndarrayToMat;
import static com.jiamny.Utils.ImageViewer.show;
import static com.jiamny.Utils.HelperFunctions.printListObjects;
import static org.opencv.imgcodecs.Imgcodecs.imread;

// Download model at: https://pan.baidu.com/s/1HYTfWazlk8W9pQDGVWAYbA?pwd=8mwt

public final class CocoDetection {

  private static final Logger logger = LoggerFactory.getLogger(CocoDetection.class);

  static {
    //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    //System.load("C:\\Program Files\\Opencv4\\java\\x64\\opencv_java454.dll");
    System.load("/usr/local/share/java/opencv4/libopencv_java470.so");
  }

  private CocoDetection() {}

  public static void main(String[] args) throws IOException, ModelException, TranslateException {
    String imagePath = "./data/images/ped_vec.jpeg";
    BufferedImage img = ImageIO.read(new File(imagePath));
    Image image = ImageFactory.getInstance().fromImage(img);

    /*
    Path imageFile = Paths.get(imagePath);
    Mat img = imread(imageFile.toString());
    Image image = mat2DjlImage(img);
     */

    DetectedObjects detections = CocoDetection.predict(image);

    int width = image.getWidth();
    int height = image.getHeight();
    System.out.println("Image width: " + width + " height: " + height);

    for (DetectedObjects.DetectedObject obj : detections.<DetectedObjects.DetectedObject>items()) {
      BoundingBox bbox = obj.getBoundingBox();

      Rectangle rectangle = bbox.getBounds();
      String showText = String.format("%s: %.2f", obj.getClassName(), obj.getProbability());
    }

    saveBoundingBoxImage(image, detections, "ped_vec_result.png", "output");

    logger.info("{}", detections);
  }

  public static DetectedObjects predict(Image img)
      throws IOException, ModelException, TranslateException {
    img.getWrappedImage();

    Criteria<Image, DetectedObjects> criteria =
            Criteria.builder()
                    .optEngine("PaddlePaddle")
                    .setTypes(Image.class, DetectedObjects.class)
                    .optModelPath(Paths.get("models/object_detection_coco/traffic.zip"))
                    .optModelName("inference")
                    .optTranslator(new TrafficTranslator())
                    .optProgress(new ProgressBar())
                    .build();

    try (ZooModel model = ModelZoo.loadModel(criteria)) {
      try (Predictor<Image, DetectedObjects> predictor = model.newPredictor()) {
        DetectedObjects objects = predictor.predict(img);
        return objects;
      }
    }
  }

  private static final class TrafficTranslator implements Translator<Image, DetectedObjects> {

    private List<String> className;

    TrafficTranslator() {}

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
      Model model = ctx.getModel();
      try (InputStream is = model.getArtifact("label_file.txt").openStream()) {
        className = Utils.readLines(is, true);
        //            classes.add(0, "blank");
        //            classes.add("");
      }

      printListObjects(Arrays.asList(className.toArray()));
    }

    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
      return processImageOutput(list);
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
      NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
      array = NDImageUtils.resize(array, 512, 512);
      if (!array.getDataType().equals(DataType.FLOAT32)) {
        array = array.toType(DataType.FLOAT32, false);
      }
      //      array = array.div(255f);
      NDArray mean = ctx.getNDManager().create(new float[] {104f, 117f, 123f}, new Shape(1, 1, 3));
      NDArray std = ctx.getNDManager().create(new float[] {1f, 1f, 1f}, new Shape(1, 1, 3));
      array = array.sub(mean);
      array = array.div(std);

      array = array.transpose(2, 0, 1); // HWC -> CHW RGB
      array = array.expandDims(0);

      return new NDList(array);
    }

    @Override
    public Batchifier getBatchifier() {
      return null;
    }

    DetectedObjects processImageOutput(NDList list) {
      NDArray result = list.singletonOrThrow();
      float[] probabilities = result.get(":,1").toFloatArray();
      List<String> names = new ArrayList<>();
      List<Double> prob = new ArrayList<>();
      List<BoundingBox> boxes = new ArrayList<>();
      for (int i = 0; i < probabilities.length; i++) {
        if (probabilities[i] < 0.55) continue;

        float[] array = result.get(i).toFloatArray();
        //        [  0.          0.9627503 172.78745    22.62915   420.2703    919.949    ]
        //        [  0.          0.8364255 497.77234   161.08307   594.4088    480.63745  ]
        //        [  0.          0.7247823  94.354065  177.53668   169.24417   429.2456   ]
        //        [  0.          0.5549363  18.81821   209.29712   116.40645   471.8595   ]
        // 1-person 行人 2-bicycle 自行车 3-car 小汽车 4-motorcycle 摩托车 6-bus 公共汽车 8-truck 货车

        int index = (int) array[0];
        names.add(className.get(index));
        // array[0] category_id
        // array[1] confidence
        // bbox
        // array[2]
        // array[3]
        // array[4]
        // array[5]
        prob.add((double) probabilities[i]);
        // x, y , w , h
        // dt['left'], dt['top'], dt['right'], dt['bottom'] = clip_bbox(bbox, org_img_width,
        // org_img_height)
        boxes.add(new Rectangle(array[2], array[3], array[4] - array[2], array[5] - array[3]));
      }
      return new DetectedObjects(names, prob, boxes);
    }
  }
  private static void saveBoundingBoxImage(
      Image img, DetectedObjects detection, String name, String path) throws IOException {
    // Make image copy with alpha channel because original image was jpg
    img.drawBoundingBoxes(detection);

    NDManager manager = NDManager.newBaseManager();
    Mat im = ndarrayToMat(img.toNDArray(manager));
    show(im, "Detected_image");

    Path outputDir = Paths.get(path);
    Files.createDirectories(outputDir);
    Path imagePath = outputDir.resolve(name);
    // OpenJDK can't save jpg with alpha channel
    img.save(Files.newOutputStream(imagePath), "png");
  }
}
