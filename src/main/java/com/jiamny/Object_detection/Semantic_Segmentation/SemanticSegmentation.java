package com.jiamny.Object_detection.Semantic_Segmentation;
import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.CategoryMask;
import ai.djl.modality.cv.translator.SemanticSegmentationTranslatorFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.Color;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * An example of inference using a semantic segmentation model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/semantic_segmentation.md">doc</a>
 * for information about this example.
 */
public final class SemanticSegmentation {

  private static final Logger logger = LoggerFactory.getLogger(SemanticSegmentation.class);

  private SemanticSegmentation() {}

  public static void main(String[] args) throws IOException, ModelException, TranslateException {
    java.util.Set<java.lang.String> egs = Engine.getAllEngines();
    for( String s : egs )
      System.out.println(s);

    Engine eg = Engine.getEngine("PyTorch");
    System.out.println("Engine: " + eg.getEngineName());

    ai.djl.Device [] ds = eg.getDevices();
    for(int i = 0; i < ds.length; i++)
      System.out.println(ds[i].toString());

    System.out.println("Default device: " + eg.defaultDevice().toString());

    SemanticSegmentation.predict();
  }

  public static void predict() throws IOException, ModelException, TranslateException {
    Path imageFile = Paths.get("data/images/person.jpg");
    ImageFactory factory = ImageFactory.getInstance();
    Image img = factory.fromFile(imageFile);

    String url =
            "https://mlrepo.djl.ai/model/cv/semantic_segmentation/ai/djl/pytorch/deeplabv3/0.0.1/deeplabv3.zip";

    Criteria<Image, CategoryMask> criteria =
            Criteria.builder()
                    .setTypes(Image.class, CategoryMask.class)
                    .optModelUrls(url)
                    .optTranslatorFactory(new SemanticSegmentationTranslatorFactory())
                    .optEngine("PyTorch")
                    .optDevice(Device.cpu()) //use CPU !!! 'prepacked::conv2d_clamp_run' only available for CPU
                    .optProgress(new ProgressBar())
                    .build();
    Image bg = factory.fromFile(Paths.get("data/images/stars-in-the-night-sky.jpg"));
    try (ZooModel<Image, CategoryMask> model = criteria.loadModel();
         Predictor<Image, CategoryMask> predictor = model.newPredictor()) {
      CategoryMask mask = predictor.predict(img);

      // Highlights the detected object on the image with random opaque colors.
      Image img1 = img.duplicate();
      mask.drawMask(img1, 255);
      saveSemanticImage(img1, "semantic_instances1.png");

      // Highlights the detected object on the image with random colors.
      Image img2 = img.duplicate();
      mask.drawMask(img2, 180, 0);
      saveSemanticImage(img2, "semantic_instances2.png");

      // Highlights only the dog with blue color.
      Image img3 = img.duplicate();
      mask.drawMask(img3, 12, Color.BLUE.brighter().getRGB(), 180);
      saveSemanticImage(img3, "semantic_instances3.png");

      // Extract dog from the image
      Image dog = mask.getMaskImage(img, 12);
      dog = dog.resize(img.getWidth(), img.getHeight(), true);
      saveSemanticImage(dog, "semantic_instances4.png");

      // Replace background with an image
      bg = bg.resize(img.getWidth(), img.getHeight(), true);
      bg.drawImage(dog, true);
      saveSemanticImage(bg, "semantic_instances5.png");
    }
  }

  private static void saveSemanticImage(Image img, String fileName) throws IOException {
    Path outputDir = Paths.get("output");
    Files.createDirectories(outputDir);

    Path imagePath = outputDir.resolve(fileName);
    img.save(Files.newOutputStream(imagePath), "png");
    logger.info("Segmentation result image has been saved in: {}", imagePath.toAbsolutePath());
  }
}
