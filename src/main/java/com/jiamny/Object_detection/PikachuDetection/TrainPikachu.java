/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.jiamny.Object_detection.PikachuDetection;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.cv.object_detection.ssd.SingleShotDetection;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import com.jiamny.Utils.Arguments;
import com.jiamny.Utils.ImageViewer;
import com.jiamny.Utils.ImageUtils;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.MultiBoxDetection;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.SingleShotDetectionTranslator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.BoundingBoxError;
import ai.djl.training.evaluator.SingleShotDetectionAccuracy;
import ai.djl.training.listener.SaveModelTrainingListener;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.SingleShotDetectionLoss;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.TranslateException;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static com.jiamny.Utils.ImageHelper.*;
import static com.jiamny.Utils.ImageViewer.show;

/**
 * An example of training a simple Single Shot Detection (SSD) model.
 *
 * <p>See this <a
 * href="https://github.com/deepjavalibrary/djl/blob/master/examples/docs/train_pikachu_ssd.md">doc</a>
 * for information about this example.
 */
public final class TrainPikachu {
    static {
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");
    }
    private TrainPikachu() {}

    public static void main(String[] args) throws IOException, TranslateException {
        System.out.println(System.getProperty("user.dir"));

        boolean train_model = false;

        if( train_model ) {
            // training the model
            TrainPikachu.runExample(args);
        } else {
            /*
            Arguments arguments = new Arguments().parseArgs(args);
            NDManager manager = NDManager.newBaseManager();
            RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);

            BufferedImage[] aimgs = new BufferedImage[5];
            for( long i = 0; i < 5; i++ ) {
                ai.djl.training.dataset.Record rd = trainingSet.get(manager, i);
                NDArray im = rd.getData().get(0); //.transpose(2, 0, 1); // HWC -> CHW RGB
                NDArray lb = rd.getLabels().get(0).squeeze();
                System.out.println(im);
                System.out.println(lb);
                Mat img = ndarrayToMat(im);

                //Mat bgr = img.clone();
                //Imgproc.cvtColor(bgr, bgr, Imgproc.COLOR_RGB2BGR);
                //show(bgr, "pikachu_image");

                float img_width = img.width();
                float img_height = img.height();
                System.out.println("width: " + img_width + " height: " + img_height);

                Image imm = mat2DjlImage(img);

                List<String> classNames = new ArrayList<>();
                List<Double> prob = new ArrayList<>();
                List<BoundingBox> boundBoxes = new ArrayList<>();

                float width = (lb.getFloat(3) - lb.getFloat(1));
                float height = (lb.getFloat(4) - lb.getFloat(2));
                Rectangle rect = new Rectangle(lb.getFloat(1), lb.getFloat(2), width, height);

                classNames.add(String.valueOf(lb.getFloat(0)));
                prob.add(1.0);
                boundBoxes.add(rect);

                DetectedObjects detectedObjects = new DetectedObjects(classNames, prob, boundBoxes);
                imm.drawBoundingBoxes(detectedObjects);

                img = ndarrayToMat(imm.toNDArray(manager));
                //show(img, "BoundBoxed_image");
                aimgs[(int)i] = matToBufferedImage(img);
            }
            BufferedImage image = ImageUtils.showImages(aimgs, 256, 256);
            ImageViewer.displayImage(image);
             */

            // prediction
            try {
                TrainPikachu.predict("output", "data/images/pikachu.jpg");
            } catch(Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static TrainingResult runExample(String[] args) throws IOException, TranslateException {
        Arguments arguments = new Arguments().parseArgs(args);
        if (arguments == null) {
            return null;
        }
        System.out.println("out.dir: " + arguments.getOutputDir());

        try (Model model = Model.newInstance("pikachu-ssd")) {
            model.setBlock(getSsdTrainBlock());
            RandomAccessDataset trainingSet = getDataset(Dataset.Usage.TRAIN, arguments);
            RandomAccessDataset validateSet = getDataset(Dataset.Usage.TEST, arguments);

            DefaultTrainingConfig config = setupTrainingConfig(arguments);

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.setMetrics(new Metrics());

                Shape inputShape = new Shape(arguments.getBatchSize(), 3, 256, 256);
                trainer.initialize(inputShape);

                EasyTrain.fit(trainer, arguments.getEpoch(), trainingSet, validateSet);

                return trainer.getTrainingResult();
            }
        }
    }

    public static int predict(String outputDir, String imageFile)
            throws IOException, MalformedModelException, TranslateException {
        try (Model model = Model.newInstance("pikachu-ssd")) {
            float detectionThreshold = 0.6f;
            // load parameters back to original training block
            model.setBlock(getSsdTrainBlock());
            model.load(Paths.get(outputDir));
            // append prediction logic at end of training block with parameter loaded
            Block ssdTrain = model.getBlock();
            model.setBlock(getSsdPredictBlock(ssdTrain));

            Path imagePath = Paths.get(imageFile);
            SingleShotDetectionTranslator translator =
                    SingleShotDetectionTranslator.builder()
                            .addTransform(new ToTensor())
                            .optSynset(Collections.singletonList("pikachu"))
                            .optThreshold(detectionThreshold)
                            .build();

            try (Predictor<Image, DetectedObjects> predictor = model.newPredictor(translator)) {
                Image image = ImageFactory.getInstance().fromFile(imagePath);
                DetectedObjects detectedObjects = predictor.predict(image);
                image.drawBoundingBoxes(detectedObjects);
                Path out = Paths.get(outputDir).resolve("pikachu_output.png");
                image.save(Files.newOutputStream(out), "png");
                System.out.println("detectedObjects.getNumberOfObjects(): " + detectedObjects.getNumberOfObjects());
                System.out.println("detectedObjects: " + detectedObjects);
                // return number of pikachu detected
                return detectedObjects.getNumberOfObjects();
            }
        }
    }

    private static RandomAccessDataset getDataset(Dataset.Usage usage, Arguments arguments)
            throws IOException {
        Pipeline pipeline = new Pipeline(new ToTensor());
        PikachuDetection pikachuDetection =
                PikachuDetection.builder()
                        .optUsage(usage)
                        .optLimit(arguments.getLimit())
                        .optPipeline(pipeline)
                        .setSampling(1, true)
                        .build();
        pikachuDetection.prepare(new ProgressBar());

        return pikachuDetection;
    }

    private static DefaultTrainingConfig setupTrainingConfig(Arguments arguments) {
        String outputDir = arguments.getOutputDir();
        SaveModelTrainingListener listener = new SaveModelTrainingListener(outputDir);
        listener.setSaveModelCallback(
                trainer -> {
                    TrainingResult result = trainer.getTrainingResult();
                    Model model = trainer.getModel();
                    float accuracy = result.getValidateEvaluation("classAccuracy");
                    model.setProperty("ClassAccuracy", String.format("%.5f", accuracy));
                    model.setProperty("Loss", String.format("%.5f", result.getValidateLoss()));
                });

        return new DefaultTrainingConfig(new SingleShotDetectionLoss())
                .addEvaluator(new SingleShotDetectionAccuracy("classAccuracy"))
                .addEvaluator(new BoundingBoxError("boundingBoxError"))
                .optDevices(Engine.getInstance().getDevices(arguments.getMaxGpus()))
                .addTrainingListeners(TrainingListener.Defaults.logging(outputDir))
                .addTrainingListeners(listener);
    }

    public static Block getSsdTrainBlock() {
        int[] numFilters = {16, 32, 64};
        SequentialBlock baseBlock = new SequentialBlock();
        for (int numFilter : numFilters) {
            baseBlock.add(SingleShotDetection.getDownSamplingBlock(numFilter));
        }

        List<List<Float>> sizes = new ArrayList<>();
        List<List<Float>> ratios = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            ratios.add(Arrays.asList(1f, 2f, 0.5f));
        }
        sizes.add(Arrays.asList(0.2f, 0.272f));
        sizes.add(Arrays.asList(0.37f, 0.447f));
        sizes.add(Arrays.asList(0.54f, 0.619f));
        sizes.add(Arrays.asList(0.71f, 0.79f));
        sizes.add(Arrays.asList(0.88f, 0.961f));

        return SingleShotDetection.builder()
                .setNumClasses(1)
                .setNumFeatures(3)
                .optGlobalPool(true)
                .setRatios(ratios)
                .setSizes(sizes)
                .setBaseNetwork(baseBlock)
                .build();
    }

    public static Block getSsdPredictBlock(Block ssdTrain) {

        // add prediction process
        SequentialBlock ssdPredict = new SequentialBlock();
        ssdPredict.add(ssdTrain);
        ssdPredict.add(
                new LambdaBlock(
                        output -> {
                            NDArray anchors = output.get(0);
                            NDArray classPredictions = output.get(1).softmax(-1).transpose(0, 2, 1);
                            NDArray boundingBoxPredictions = output.get(2);
                            MultiBoxDetection multiBoxDetection =
                                    MultiBoxDetection.builder().build();
                            NDList detections =
                                    multiBoxDetection.detection(
                                            new NDList(
                                                    classPredictions,
                                                    boundingBoxPredictions,
                                                    anchors));

                            System.out.println("detections.size: " + detections.size());
                            System.out.println("detections.singletonOrThrow(): " +
                                    detections.singletonOrThrow().getShape());
                            System.out.println("detections.singletonOrThrow().split(new long[] {1, 2}, 2): " +
                                    detections.singletonOrThrow().split(new long[] {1, 2}, 2));

                            NDList rlt =detections.singletonOrThrow().split(new long[] {1, 2}, 2);
                            System.out.println( "rlt[0]: " + rlt.get(0).get(":,0:10"));
                            System.out.println( "rlt[1]: " + rlt.get(1).get(":,0:10"));
                            System.out.println( "rlt[2]: " + rlt.get(2).get(":,0:10"));
                            //NDIndex idx = rlt.get(0).get(1) >= 0;
                            return detections.singletonOrThrow().split(new long[] {1, 2}, 2);
                        }));
        //System.out.println("ssdPredict: " + ssdPredict);
        return ssdPredict;
        /*
        // add prediction process
        SequentialBlock ssdPredict = new SequentialBlock();
        ssdPredict.add(ssdTrain);
        ssdPredict.add(
                new LambdaBlock(
                        output -> {
                            NDArray anchors = output.get(0);
                            NDArray classPredictions = output.get(1).softmax(-1).transpose(0, 2, 1);
                            NDArray boundingBoxPredictions = output.get(2);
                            MultiBoxDetection multiBoxDetection =
                                    MultiBoxDetection.builder().build();
                            NDList detections =
                                    multiBoxDetection.detection(
                                            new NDList(
                                                    classPredictions,
                                                    boundingBoxPredictions,
                                                    anchors));
                            return detections.singletonOrThrow().split(new long[] {1, 2}, 2);
                        }));
        return ssdPredict;

         */
    }
}
