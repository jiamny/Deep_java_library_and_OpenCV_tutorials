package com.jiamny.DJL_modern_cnn;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.basicdataset.cv.classification.*;
import ai.djl.engine.Engine;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.convolutional.Conv2d;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.pooling.Pool;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import com.jiamny.Utils.Training;
import org.apache.commons.lang3.ArrayUtils;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.LinePlot;

import java.io.IOException;
import java.util.*;

public class VGG {

    public SequentialBlock vggBlock(int numConvs, int numChannels) {

        SequentialBlock tempBlock = new SequentialBlock();
        for (int i = 0; i < numConvs; i++) {
            // DJL has default stride of 1x1, so don't need to set it explicitly.
            tempBlock
                    .add(Conv2d.builder()
                            .setFilters(numChannels)
                            .setKernelShape(new Shape(3, 3))
                            .optPadding(new Shape(1, 1))
                            .build()
                    )
                    .add(Activation::relu);
        }
        tempBlock.add(Pool.maxPool2dBlock(new Shape(2, 2), new Shape(2, 2)));
        return tempBlock;
    }

    public SequentialBlock createVggModel(int[][] convArch) {

        SequentialBlock block = new SequentialBlock();
        // The convolutional layer part
        for (int i = 0; i < convArch.length; i++) {
            block.add(vggBlock(convArch[i][0], convArch[i][1]));
        }

        // The fully connected layer part
        block
                .add(Blocks.batchFlattenBlock())
                .add(Linear
                        .builder()
                        .setUnits(4096)
                        .build())
                .add(Activation::relu)
                .add(Dropout
                        .builder()
                        .optRate(0.5f)
                        .build())
                .add(Linear
                        .builder()
                        .setUnits(4096)
                        .build())
                .add(Activation::relu)
                .add(Dropout
                        .builder()
                        .optRate(0.5f)
                        .build())
                .add(Linear.builder().setUnits(10).build());

        return block;
    }

    public static void main(String[] args) throws IOException, TranslateException, ModelException {

        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        VGG vgg = new VGG();

        // Since this network uses 8 convolutional layers and 3 fully-connected layers,
        // it is often called VGG-11.
        int[][] convArch = {{1, 64}, {1, 128}, {2, 256}, {2, 512}, {2, 512}};

        SequentialBlock vgg11 = vgg.createVggModel(convArch);

        // --------------------------------------------------------
        // construct a single-channel data example with a height and
        // width of 224 to observe the output shape of each layer.
        // --------------------------------------------------------

        float lr = 0.05f;
        Model model = Model.newInstance("vgg-display");
        model.setBlock(vgg11);

        Loss loss = Loss.softmaxCrossEntropyLoss();

        Tracker lrt = Tracker.fixed(lr);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        DefaultTrainingConfig config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
                .optDevices(Engine.getInstance().getDevices(1)) // single GPU
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Trainer trainer = model.newTrainer(config);

        Shape inputShape = new Shape(1, 1, 224, 224);

        try(NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.randomUniform(0f, 1.0f, inputShape);
            trainer.initialize(inputShape);

            Shape currentShape = X.getShape();

            for (int i = 0; i < vgg11.getChildren().size(); i++) {
                Shape[] newShape = vgg11.getChildren().get(i).getValue().getOutputShapes(new Shape[]{currentShape});
                currentShape = newShape[0];
                System.out.println(vgg11.getChildren().get(i).getKey() + " layer output : " + currentShape);
            }
        }
        // save memory on VGG params
        model.close();

        // ------------------------------------------------------------
        // Model Training
        // ------------------------------------------------------------

        int ratio = 4;

        for(int i=0; i < convArch.length; i++){
            convArch[i][1] = convArch[i][1] / ratio;
        }

        inputShape = new Shape(1, 1, 96, 96); // resize the input shape to save memory

        model = Model.newInstance("vgg-tiny");
        SequentialBlock newBlock = vgg.createVggModel(convArch);
        model.setBlock(newBlock);
        loss = Loss.softmaxCrossEntropyLoss();

        lrt = Tracker.fixed(lr);
        sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        config = new DefaultTrainingConfig(loss).optOptimizer(sgd) // Optimizer (loss function)
                .optDevices(Engine.getInstance().getDevices(1)) // single GPU
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        trainer = model.newTrainer(config);
        trainer.initialize(inputShape);

        int batchSize = 128;
        int numEpochs = Integer.getInteger("MAX_EPOCH", 10);

        double[] trainLoss;
        double[] testAccuracy;
        double[] epochCount;
        double[] trainAccuracy;

        epochCount = new double[numEpochs];

        for (int i = 0; i < epochCount.length; i++) {
            epochCount[i] = i+1;
        }

        FashionMnist trainIter = FashionMnist.builder()
                .addTransform(new Resize(96))
                .addTransform(new ToTensor())
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, true)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        FashionMnist testIter = FashionMnist.builder()
                .addTransform(new Resize(96))
                .addTransform(new ToTensor())
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, true)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        trainIter.prepare();
        testIter.prepare();

        // Apart from using a slightly larger learning rate
        Map<String, double[]> evaluatorMetrics = new HashMap<>();
        double avgTrainTimePerEpoch = Training.trainingChapter6(trainIter,
                                        testIter, numEpochs, trainer, evaluatorMetrics);

        trainLoss = evaluatorMetrics.get("train_epoch_SoftmaxCrossEntropyLoss");
        trainAccuracy = evaluatorMetrics.get("train_epoch_Accuracy");
        testAccuracy = evaluatorMetrics.get("validate_epoch_Accuracy");

        System.out.printf("loss %.3f,", trainLoss[numEpochs - 1]);
        System.out.printf(" train acc %.3f,", trainAccuracy[numEpochs - 1]);
        System.out.printf(" test acc %.3f\n", testAccuracy[numEpochs - 1]);
        System.out.printf("%.1f examples/sec", trainIter.size() / (avgTrainTimePerEpoch / Math.pow(10, 9)));
        System.out.println();

        String[] lossLabel = new String[trainLoss.length + testAccuracy.length + trainAccuracy.length];

        Arrays.fill(lossLabel, 0, trainLoss.length, "train loss");
        Arrays.fill(lossLabel, trainAccuracy.length, trainLoss.length + trainAccuracy.length, "train acc");
        Arrays.fill(lossLabel, trainLoss.length + trainAccuracy.length,
                trainLoss.length + testAccuracy.length + trainAccuracy.length, "test acc");

        Table data = Table.create("Data").addColumns(
                DoubleColumn.create("epoch", ArrayUtils.addAll(epochCount, ArrayUtils.addAll(epochCount, epochCount))),
                DoubleColumn.create("metrics", ArrayUtils.addAll(trainLoss, ArrayUtils.addAll(trainAccuracy, testAccuracy))),
                StringColumn.create("lossLabel", lossLabel)
        );

        Plot.show(LinePlot.create("", data, "epoch", "metrics", "lossLabel"));

        System.exit(0);
    }
}
