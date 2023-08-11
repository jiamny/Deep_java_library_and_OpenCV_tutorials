package com.jiamny.DJL_NeuralNetworks_Basics;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.basicdataset.cv.classification.FashionMnist;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.jiamny.Utils.Training;

import java.util.Arrays;
import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;


public class SoftmaxRegression {

    public static void main(String[] args) {
        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        // Initializing Model Parameters
        int numInputs = 784;
        int numOutputs = 10;
        int batchSize = 256;
        boolean randomShuffle = true;
        NDManager manager = NDManager.newBaseManager();

        // ---------------------------------------------------------
        // Concise Implementation of Softmax Regression
        // ---------------------------------------------------------

        FashionMnist trainingSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TRAIN)
                .setSampling(batchSize, randomShuffle)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        FashionMnist validationSet = FashionMnist.builder()
                .optUsage(Dataset.Usage.TEST)
                .setSampling(batchSize, false)
                .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                .build();

        // Initializing Model Parameters
        Model model = Model.newInstance("softmax-regression");

        SequentialBlock net = new SequentialBlock();
        net.add(Blocks.batchFlattenBlock(28 * 28)); // flatten input
        net.add(Linear.builder().setUnits(10).build()); // set 10 output channels

        model.setBlock(net);

        // The Softmax
        Loss loss = Loss.softmaxCrossEntropyLoss();

        // Optimization Algorithm
        Tracker lrt = Tracker.fixed(0.1f);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        // Instantiate Configuration
        DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
                .optOptimizer(sgd) // Optimizer
                .optDevices(manager.getEngine().getDevices(1)) // single GPU
                .addEvaluator(new Accuracy()) // Model Accuracy
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

        Trainer trainer = model.newTrainer(config);

        // Initializing Trainer
        trainer.initialize(new Shape(1, 28 * 28)); // Input Images are 28 x 28

        // Metrics
        Metrics metrics = new Metrics();
        trainer.setMetrics(metrics);

        // Training
        int numEpochs = 3;
        try {
            EasyTrain.fit(trainer, numEpochs, trainingSet, validationSet);
            var result = trainer.getTrainingResult();
            System.out.println(result);
        } catch(Exception e) {
            e.printStackTrace();
        }

        System.exit(0);
    }
}
