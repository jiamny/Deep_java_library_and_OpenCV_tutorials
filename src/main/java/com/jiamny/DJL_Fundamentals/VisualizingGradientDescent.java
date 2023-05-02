package com.jiamny.DJL_Fundamentals;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import com.jiamny.Utils.DataPoints;
import com.jiamny.Utils.HelperFunctions;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.FloatColumn;
import tech.tablesaw.api.NumericColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.ScatterPlot;
import tech.tablesaw.plotly.components.*;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.util.*;
import java.util.stream.Collectors;

public class VisualizingGradientDescent {

    // Generate y = X w + b + noise
    public static DataPoints syntheticData(NDManager manager, NDArray w, float b, int numExamples) {
        NDArray X = manager.randomNormal(new Shape(numExamples, w.size()));
        NDArray y = X.matMul(w).add(b);
        // Add noise
        y = y.add(manager.randomNormal(0, 0.2f, y.getShape(), DataType.FLOAT32));
        return new DataPoints(X, y);
    }

    public static void main(String[] args) {
        Engine.getInstance().setRandomSeed(123);

        Device device = Engine.getInstance().defaultDevice();
        if (Engine.getInstance().getGpuCount() > 0)
            device = Engine.getInstance().getDevices(1)[0];

        NDManager manager = NDManager.newBaseManager();

        try {

            // Synthetic Data Generation
            int N = 100;
            NDArray trueW = manager.create(new float[]{2.0f});
            float trueB = 1.0f;

            DataPoints dp = syntheticData(manager, trueW, trueB, N);
            NDArray features = dp.getX();
            NDArray labels = dp.getY();

            System.out.printf("features: [%f]\n", features.getFloat(0)); //, features.get(0).getFloat(1));
            System.out.println("label: " + labels.getFloat(0));

            //float[] X = features.get(new NDIndex(":, 1")).toFloatArray();
            float[] X = features.toFloatArray();
            float[] y = labels.toFloatArray();

            Table data = Table.create("Data")
                    .addColumns(
                            FloatColumn.create("X", X),
                            FloatColumn.create("y", y)
                    );

            //Plot.show(ScatterPlot.create("Synthetic Data", data, "X", "y"));

            // load dataset
            int batchSize = 80;

            ArrayDataset dataset = new ArrayDataset.Builder()
                    .setData(features) // Set the Features
                    .optLabels(labels) // Set the Labels
                    .setSampling(batchSize, false) // set the batch size and random sampling to false
                    .build();

            // read and print the first small batch of data examples
            Batch batch = dataset.getData(manager).iterator().next();

            // Call head() to get the first NDArray
            NDArray X1 = batch.getData().head();
            NDArray y1 = batch.getLabels().head();
            System.out.println(X1);
            System.out.println(y1);

            // Random Initialization
            NDArray w = manager.randomNormal(0, 0.01f, new Shape(1), DataType.FLOAT32);
            float[] inw = {-0.16f};
            w.set(inw);
            NDArray b = manager.zeros(new Shape(1));
            float[] inb = {0.52f};
            b.set(inb);

            NDList params = new NDList(w.duplicate(), b.duplicate());

            System.out.printf("w: %.4f\tb: %.4f", w.duplicate().toFloatArray()[0], b.duplicate().toFloatArray()[0]);

            // Step 1 - Computes our model's predicted output - forward pass
            NDArray yhat = X1.matMul(w).add(b);

            // Step 2 - Computing the loss
            NDArray error = yhat.sub(y1);

            // It is a regression, so it computes mean squared error (MSE)
            NDArray loss = (error.pow(2)).mean();
            System.out.println(loss);

            // Reminder:
            // true_b = 1
            // true_w = 2

            // we have to split the ranges in 100 evenly spaced intervals each
            NDArray b_range = HelperFunctions.linspace(trueB - 3.0f, trueB + 3.0f, 100);
            NDArray w_range = HelperFunctions.linspace(trueW.toFloatArray()[0] - 3.0f, trueW.toFloatArray()[0] + 3.0f, 100);
            // meshgrid is a handy function that generates a grid of b and w values for all combinations
            ArrayList<NDArray> dt = HelperFunctions.meshgrid(b_range, w_range);
            NDArray bs = dt.get(0), ws = dt.get(1);
            System.out.println("bs.shape: " + bs.getShape().toString() + " ws.shape: " + ws.getShape().toString());

            NDArray sample_x = X1.get(0);
            NDArray sample_yhat = (ws.mul(sample_x)).add(bs);
            System.out.println("sample_yhat.shape: " + sample_yhat.getShape().toString());
            System.out.println("X1.shape: " + X1.getShape().toString());

            List<NDArray> Adata = new ArrayList<>();
            Adata.add(sample_yhat.duplicate());
            for (long i = 1; i < X1.getShape().getShape()[0]; i++)
                Adata.add((ws.mul(X1.get(i))).add(bs));
            //all_predictions = all_predictions.concat((ws.mul(X1.get(i))).add(bs), 0);

            // -------------------------------------------------------------------
            // stack NDArray
            // -------------------------------------------------------------------
            NDArray all_predictions = NDArrays.stack(Adata.stream()
                    .map(list -> list)
                    .collect(Collectors.toCollection(NDList::new)));

            System.out.println("all_predictions.shape: " + all_predictions.getShape().toString());

            NDArray all_labels = y1.reshape(-1, 1, 1);
            System.out.println("all_labels.shape: " + all_labels.getShape().toString());

            NDArray all_errors = all_predictions.sub(all_labels);
            System.out.println("all_errors.shape: " + all_errors.getShape().toString());

            NDArray all_losses = all_errors.pow(2).mean(new int[]{0});
            System.out.println("all_losses.shape: " + all_losses.getShape().toString());

            // Step 3 - Computes gradients for both "b" and "w" parameters
            var b_grad = error.mean().mul(2);
            var w_grad = ((X1.mul(error)).mean()).mul(2);
            System.out.println(b_grad);
            System.out.println(w_grad);

            var b_rang = bs.get("0, :");
            var w_rang = ws.get(":, 0");

            //b_idx, w_idx, fixedb, fixedw = find_index(b, w, bs, ws)

            //def find_index(b, w, bs, ws):
            // Looks for the closer indexes for the updated b and w inside their respective ranges
            var b_idx = Arrays.toString((b_rang.sub(b).abs()).argMin().toLongArray());
            var w_idx = Arrays.toString((w_rang.sub(w).abs()).argMin().toLongArray());
            b_idx = b_idx.replace("[", "");
            b_idx = b_idx.replace("]", "");
            w_idx = w_idx.replace("[", "");
            w_idx = w_idx.replace("]", "");
            System.out.println("b_idx: " + b_idx);
            System.out.println("w_idx: " + w_idx);

            // Closest values for b and w
            var fixedb = bs.get("0, " + b_idx);
            var fixedw = ws.get(w_idx + ", 0");
            System.out.println("fixedb: " + fixedb);
            System.out.println("fixedw: " + fixedw);
            //return b_idx, w_idx, fixedb, fixedw

            var ls = all_losses.get(":, " + b_idx);

            System.out.println(ls.size());
            System.out.println(w_range.size());

            NumericColumn<?> xw = FloatColumn.create("x", w_range.toFloatArray());
            NumericColumn<?> yl = FloatColumn.create("y", ls.toFloatArray());

            Layout layout = Layout.builder()
                    .title("Fixed: b = " + b.toFloatArray()[0])
                    .yAxis(Axis.builder().title("MSE (loss)").build())
                    .xAxis(Axis.builder().title("w").showGrid(false).build())
                    .showLegend(false)
                    .build();
            ScatterTrace trace = ScatterTrace.builder(xw, yl)
                    .mode(ScatterTrace.Mode.LINE)
                    .line(Line.builder().dash(Line.Dash.DASH).color("red").build())
                    .build();

            ScatterTrace pt = ScatterTrace.builder(fixedw.mul(1.0).toDoubleArray(),
                            all_losses.get(w_idx + "," + b_idx).mul(1.0).toDoubleArray())
                    .marker(Marker.builder().symbol(Symbol.DIAMOND).size(10.0).color("blue").build())
                    .build();
            Plot.show(new Figure(layout, trace, pt));
            // Don't forget to close the batch!
            batch.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
        System.exit(0);
    }
}
