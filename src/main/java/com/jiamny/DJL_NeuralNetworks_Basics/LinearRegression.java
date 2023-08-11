package com.jiamny.DJL_NeuralNetworks_Basics;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.internal.NDArrayEx;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.jiamny.Utils.DataPoints;
import com.jiamny.Utils.UtilFunctions;
import org.apache.commons.lang3.time.StopWatch;
import tech.tablesaw.api. *;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api. *;
import tech.tablesaw.plotly.components. *;
import tech.tablesaw.api.FloatColumn;

import org.apache.commons.lang3.ArrayUtils;

//# Linear Regression
public class LinearRegression {
    //## The Normal Distribution and Squared Loss
    public static float[] normal(float[] z, float mu, float sigma) {
        float[] dist = new float[z.length];
        for (int i = 0; i < z.length; i++) {
            float p = 1.0f / (float) Math.sqrt(2 * Math.PI * sigma * sigma);
            dist[i] = p * (float) Math.pow(Math.E, -0.5 / (sigma * sigma) * (z[i] - mu) * (z[i] - mu));
        }
        return dist;
    }

    public static float[] combine3(float[] x, float[] y, float[] z) {
        return ArrayUtils.addAll(ArrayUtils.addAll(x, y), z);
    }

    public static void normalDist( boolean showFigure ) {
        int n = 10000;
        NDManager manager = NDManager.newBaseManager();
        NDArray a = manager.ones(new Shape(n));
        NDArray b = manager.ones(new Shape(n));
        NDArray c = manager.zeros(new Shape(n));

        StopWatch stopWatch = new StopWatch();
        stopWatch.start();
        for (int i = 0; i < n; i++) {
            c.set(new NDIndex(i), a.getFloat(i) + b.getFloat(i));
        }
        stopWatch.stop();
        System.out.println(String.format("%.5f millisec", stopWatch.getTime(TimeUnit.MILLISECONDS) * 1.0f));

        stopWatch.reset();
        stopWatch.start();
        NDArray d = a.add(b);
        stopWatch.stop();
        System.out.println(String.format("%.5f milsec", stopWatch.getTime(TimeUnit.MILLISECONDS) * 1.0f));

        int start = -7;
        int end = 14;
        float step = 0.01f;
        int count = (int) (end / step);

        float[] x = new float[count];

        for (int i = 0; i < count; i++) {
            x[i] = start + i * step;
        }

        float[] y1 = normal(x, 0, 1);
        float[] y2 = normal(x, 0, 2);
        float[] y3 = normal(x, 3, 1);

        String[] params = new String[x.length * 3];

        Arrays.fill(params, 0, x.length, "mean 0, var 1");
        Arrays.fill(params, x.length, x.length * 2, "mean 0, var 2");
        Arrays.fill(params, x.length * 2, x.length * 3, "mean 3, var 1");

        Table normalDistributions = Table.create("normal")
                .addColumns(
                        FloatColumn.create("z", combine3(x, x, x)),
                        FloatColumn.create("p(z)", combine3(y1, y2, y3)),
                        StringColumn.create("params", params)
                );

        Figure fig = LinePlot.create("Normal Distributions", normalDistributions, "z", "p(z)", "params");
        if( showFigure ) Plot.show(fig);
    }

    // ----------------------------------------------------------------------
    // Linear Regression Implementation from Scratch
    // ----------------------------------------------------------------------
    // Generate y = X w + b + noise
    public static DataPoints syntheticData(NDManager manager, NDArray w, float b, int numExamples) {
        NDArray X = manager.randomNormal(new Shape(numExamples, w.size()));
        // Dimension mismatch or high dimensional dot operation is not supported. Please use .matMul instead.
        // NDArray y = X.dot(w).add(b);
        NDArray y = X.matMul(w).add(b);
        // Add noise
        y = y.add(manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32));
        return new DataPoints(X, y);
    }

    //## Defining the Model
    // Saved in Training.java for later use
    public static NDArray linreg(NDArray X,NDArray w,NDArray b){
        return X.dot(w).add(b);
    }

    //## Defining the Loss Function
    // Saved in Training.java for later use
    public static NDArray squaredLoss(NDArray yHat,NDArray y){
        return(yHat.sub(y.reshape(yHat.getShape()))).mul
                ((yHat.sub(y.reshape(yHat.getShape())))).div(2);
    }

    // Defining the Optimization Algorithm
    // Saved in Training.java for later use
    public static void sgd(NDList params, float lr, int batchSize){

        try {
            for (int i = 0; i < params.size(); i++) {
                NDArray param = params.get(i);
                param.subi(param.getGradient().mul(lr).div(batchSize));
            }
        }catch(Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        boolean showFigure = false;

        // ----------------------------------------------------------------------
        // Normal Distributions
        // ----------------------------------------------------------------------
        normalDist( showFigure );

        // ----------------------------------------------------------------------
        // Linear Regression Implementation from Scratch
        // ----------------------------------------------------------------------
        System.out.println("Linear Regression Implementation from Scratch");
        try {
            //## Generating the Dataset
            NDManager manager = NDManager.newBaseManager();
            NDArray trueW = manager.create(new float[]{2, -3.4f});
            float trueB = 4.2f;

            DataPoints dp = syntheticData(manager, trueW, trueB, 1000);
            NDArray features = dp.getX();
            NDArray labels = dp.getY();

            System.out.printf("features: [%f, %f]\n", features.get(0).getFloat(0), features.get(0).getFloat(1));
            System.out.println("label: " + labels.getFloat(0));

            float[] X = features.get(new NDIndex(":, 1")).toFloatArray();
            float[] y = labels.toFloatArray();

            Table data = Table.create("Data")
                    .addColumns(
                            FloatColumn.create("X", X),
                            FloatColumn.create("y", y)
                    );

            Figure fig = ScatterPlot.create("Synthetic Data", data, "X", "y");
            if( showFigure  )
                Plot.show(fig);

            //## Reading the Dataset
            int batchSize=10;
            ArrayDataset dataset=new ArrayDataset.Builder()
                    .setData(features) // Set the Features
                    .optLabels(labels) // Set the Labels
                    .setSampling(batchSize,false) // set the batch size and random sampling to false
                    .build();

            Batch batch = dataset.getData(manager).iterator().next();
            // Call head() to get the first NDArray
            var Xt = batch.getData().head();
            var yt = batch.getLabels().head();
            System.out.println(Xt);
            System.out.println(yt);
            // Don't forget to close the batch!
            batch.close();

            //## Initializing Model Parameters
            NDArray w = manager.randomNormal(0,0.01f,new Shape(2,1),DataType.FLOAT32);
            NDArray b = manager.zeros(new Shape(1));
            NDList params = new NDList(w,b);

            //## Training
            float lr=0.03f;    // Learning Rate
            int numEpochs=20;  // Number of Iterations

            // Attach Gradients
            for(NDArray param:params){
                param.setRequiresGradient(true);
            }

            for(int epoch=0; epoch<numEpochs; epoch++){
                // Assuming the number of examples can be divided by the batch size, all
                // the examples in the training dataset are used once in one epoch
                // iteration. The features and tags of minibatch examples are given by X
                // and y respectively.
                int cnt = 0;
                for(Batch bch : dataset.getData(manager) ){
                    Xt = bch.getData().head();
                    yt = bch.getLabels().head();
                    batchSize = (int)Xt.getShape().getShape()[0];

                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        // Minibatch loss in X and y
                        NDArray l = squaredLoss(linreg(Xt, params.get(0), params.get(1)), yt);
                        gc.backward(l);  // Compute gradient on l with respect to w and b
                    }
                    sgd(params, lr, batchSize);  // Update parameters using their gradient

                    bch.close();
                    cnt++;
                    params.attach(manager);
                    //System.out.printf("epoch %d, batch %d\n",epoch+1, cnt);
                }
                NDArray trainL=squaredLoss(linreg(features,params.get(0),params.get(1)),labels);
                if( (epoch+1) % 10 == 0 )
                    System.out.printf("epoch %d, loss %f\n",epoch+1,trainL.mean().getFloat());
            }

            float[] wt = trueW.sub(params.get(0).reshape(trueW.getShape())).toFloatArray();
            System.out.println(String.format("Error in estimating w: [%f, %f]", wt[0], wt[1]));
            System.out.println(String.format("Error in estimating b: %f",trueB-params.get(1).getFloat()));

            System.out.printf("Estimating w: [%f %f]\n",
                    params.get(0).reshape(trueW.getShape()).toFloatArray()[0],
                    params.get(0).reshape(trueW.getShape()).toFloatArray()[1]);
            System.out.printf("Estimating b: %f\n", params.get(1).getFloat());

            System.out.println(String.format("True w: [%f, %f]", trueW.toFloatArray()[0], trueW.toFloatArray()[1]));
            System.out.println(String.format("True b: %f", trueB));

        }catch(Exception e) {
            e.printStackTrace();
        }

        // ----------------------------------------------------------------------
        // Linear Regression Implementation from djl
        // ----------------------------------------------------------------------
        System.out.println("Linear Regression Implementation from DJL");
        try {
            NDManager manager=NDManager.newBaseManager();
            NDArray trueW=manager.create(new float[]{2,-3.4f});
            float trueB=4.2f;

            DataPoints dp = DataPoints.syntheticData(manager, trueW, trueB,1000);
            NDArray features=dp.getX();
            NDArray labels=dp.getY();

            //## Reading the Dataset
            int batchSize=10;
            ArrayDataset dataset = UtilFunctions.loadArray(features,labels,batchSize,false);

            Batch batch=dataset.getData(manager).iterator().next();
            NDArray X=batch.getData().head();
            NDArray y=batch.getLabels().head();
            System.out.println(X);
            System.out.println(y);
            batch.close();

            //## Defining the Model
            Model model=Model.newInstance("lin-reg");

            SequentialBlock net = new SequentialBlock();
            Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build();
            net.add(linearBlock);

            model.setBlock(net);

            //## Defining the Loss Function
            Loss l2loss = Loss.l2Loss();

            //## Defining the Optimization Algorithm
            Tracker lrt = Tracker.fixed(0.03f);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            //## Instantiate Configuration and Trainer

            DefaultTrainingConfig config=new DefaultTrainingConfig(l2loss)
                    .optOptimizer(sgd) // Optimizer (loss function)
                    .optDevices(manager.getEngine().getDevices(1))      // single GPU
                    .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

            Trainer trainer = model.newTrainer(config);

            //## Initializing Model Parameters
            // First axis is batch size - won't impact parameter initialization
            // Second axis is the input size
            trainer.initialize(new Shape(batchSize,2));

            //## Metrics
            Metrics metrics = new Metrics();
            trainer.setMetrics(metrics);

            //## Training
            int numEpochs = 20;

            for(int epoch=1; epoch<=numEpochs; epoch++){
                System.out.printf("Epoch %d\n", epoch);
                // Iterate over dataset
                for(Batch bch : trainer.iterateDataset(dataset)){
                    // Update loss and evaulator
                    EasyTrain.trainBatch(trainer, bch);

                    // Update parameters
                    trainer.step();

                    bch.close();
                }
                // reset training and validation evaluators at end of epoch
                trainer.notifyListeners(listener->listener.onEpoch(trainer));
            }

            Block layer=model.getBlock();
            ParameterList params = layer.getParameters();
            NDArray wParam = params.valueAt(0).getArray();
            NDArray bParam = params.valueAt(1).getArray();

            float[]w=trueW.sub(wParam.reshape(trueW.getShape())).toFloatArray();
            System.out.printf("Error in estimating w: [%f %f]\n", w[0], w[1]);
            System.out.printf("Error in estimating b: %f\n", trueB-bParam.getFloat());

            System.out.printf("Estimating w: [%f %f]\n",
                    wParam.get(0).reshape(trueW.getShape()).toFloatArray()[0],
                    wParam.get(0).reshape(trueW.getShape()).toFloatArray()[1]);
            System.out.printf("Estimating b: %f\n", bParam.getFloat());

            System.out.println(String.format("True w: [%f, %f]", trueW.toFloatArray()[0], trueW.toFloatArray()[1]));
            System.out.println(String.format("True b: %f", trueB));

            //## Saving Your Model
            Path modelDir= Paths.get("./models/lin-reg");
            Files.createDirectories(modelDir);

            model.setProperty("Epoch",Integer.toString(numEpochs)); // save epochs trained as metadata
            model.save(modelDir,"lin-reg");
            System.out.println(model);

        } catch(Exception e) {
            e.printStackTrace();
        }

        System.exit(0);
    }
}


