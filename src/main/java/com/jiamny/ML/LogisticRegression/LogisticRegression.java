package com.jiamny.ML.LogisticRegression;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class LogisticRegression {
    private double lr;
    private NDArray weights, bias;
    private long epochs, m, n;
    /**
     * @param X: Input tensor
     * keyword lr: learning rate
     * keyword epochs: number of times the model iterates over complete dataset
     * keyword weights: parameters learned during training
     * keyword bias: parameter learned during training
     */
    public LogisticRegression(NDArray X) {
        NDManager manager = NDManager.newBaseManager();
        lr = 0.1;
        epochs = 1000;
        m = X.getShape().getShape()[0];
        n = X.getShape().getShape()[1];
        weights = manager.zeros(new Shape(n, 1), DataType.FLOAT64);
        bias = manager.zeros(new Shape(1, 1), DataType.FLOAT64);
    }

    public LogisticRegression() {}

    /**
     * @param z: latent variable represents (wx + b)
     * @return: squashes the real value between 0 and 1 representing probability score.
     */
    public NDArray sigmoid(NDArray z) {
        NDManager manager = NDManager.newBaseManager();
        z = z.mul(-1);
        z = z.exp().add(1);
        NDArray one = manager.ones(z.getShape(), DataType.FLOAT64);
        return one.div(z);
    }

    /**
     * @param yhat: Estimated y
     * @return: Log loss - When y=1, it cancels out half function, remaining half is considered
     * for loss calculation and vice-versa
     */
    public NDArray loss(NDArray yhat, NDArray y) {
        // -(1 / self.m) * torch.sum(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))
        NDArray y1 = y.mul(-1).add(1);
        NDArray yhl1 = (yhat.mul(-1).add(1)).log();
        NDArray ylyh = y.mul(yhat.log());
        NDArray rlt = ylyh.add(y1.mul(yhl1)).sum();
        return rlt.mul(-1.0/m);
    }

    /**
     * @param y_predict: Estimated y
     * @return: gradient is calculated to find how much change is required in parameters to reduce the loss.
     */
    public ArrayList<NDArray> gradient(NDArray y_predict, NDArray X, NDArray y) {
        ArrayList<NDArray> rlt = new ArrayList<>();
        NDArray dw = (X.transpose().matMul(y_predict.sub(y))).mul(1.0/m);  // 1 / m * torch.mm(X.T, (y_predict - y))
        NDArray db = (y_predict.sub(y).sum()).mul(1.0/m);   // 1 / m * torch.sum(y_predict - y)
        rlt.add(dw);
        rlt.add(db);
        return rlt;
    }

    /**
     * @param X: Input tensor
     * @param y: Output tensor
     * var y_predict: Predicted tensor
     * var cost: Difference between ground truth and predicted
     * var dw, db: Weight and bias update for weight tensor and bias scalar
     * @return: updated weights and bias
     */
    public ArrayList<NDArray> run(NDArray X, NDArray y) {

        ArrayList<NDArray> r = new ArrayList<>();
        for( int epoch = 1; epoch < (epochs + 1); epoch++ ) {
            NDArray y_predict = sigmoid(X.matMul(weights).add(bias));
            NDArray cost = loss(y_predict, y);
            ArrayList<NDArray> dwb = gradient(y_predict, X, y);
            NDArray dw = dwb.get(0);
            NDArray db = dwb.get(1);

            weights = weights.sub(dw.mul(lr));
            bias = bias.sub( db.mul(lr) );

            if( epoch % 100 == 0 )
                System.out.printf("Cost after iteration %d: %.4f\n", epoch, cost.toDoubleArray()[0]);
        }
        r.add(weights);
        r.add(bias);
        return r;
    }

    /**
     * @param X: Input tensor
     * @var y_predict_labels: Converts float value to int/bool true(1) or false(0)
     * @return: outputs labels as 0 and 1
     */
    public NDArray predict(NDArray X) {
        NDArray y_predict = sigmoid(X.matMul(weights).add(bias) );
        //NDArray y_predict_labels = y_predict.gt(0.5); // > 0.5;
        return y_predict.gt(0.5);
    }

    public ArrayList<NDArray> loadData(String fname) {
        ArrayList<String> contents = new ArrayList<>();
        int ncol = 0;
        try {
            File fr = new File(fname);
            BufferedReader in = null;

            in = new BufferedReader( new InputStreamReader(new FileInputStream(fr)));
            String line = in.readLine();  // skip column name line
            String[] curLine = line.strip().split(",");
            ncol = curLine.length;
            while( (line = in.readLine()) != null) {
                //System.out.println(line);
                contents.add(line.strip());
            }
            in.close();
        } catch(Exception e) {
            e.printStackTrace();
        }

        double [][] Xd = new double[contents.size()][ncol - 1];
        int [][]  Yd  = new int[contents.size()][1];

        for( int j = 0; j < contents.size(); j++ ) {
            if (contents.get(j).length() < 1)
                continue;
            String[] curLine = contents.get(j).strip().split(",");
            for(int i = 0; i < (curLine.length - 1); i++) {
                Xd[j][i] = Double.parseDouble(curLine[i]);
            }
            if( curLine[ncol - 1].equalsIgnoreCase("benign") )
                Yd[j][0] = 0;
            else
                Yd[j][0] = 1;
        }
        NDManager manager = NDManager.newBaseManager();
        ArrayList<NDArray> xy = new ArrayList<>();
        xy.add(manager.create(Xd));
        xy.add(manager.create(Yd));
        return xy;
    }

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        NDArray nd = manager.create(new float[][]{{-5.0f, -4.5f}, {-4.0f, -3.5f}});
        LogisticRegression LR = new LogisticRegression();
        System.out.println(LR.sigmoid(nd));


        String fName = "./data/ML/tumors.csv";
        ArrayList<NDArray> xy  = LR.loadData(fName);
        NDArray X = xy.get(0), y = xy.get(1);
        System.out.println(X.getShape().toString());
        System.out.println(y.getShape().toString());

        LR = new LogisticRegression(X);
        ArrayList<NDArray> wb = LR.run(X, y);
        NDArray w = wb.get(0), b = wb.get(1);
        NDArray y_predict = LR.predict(X);
        double ac = ((y.eq(y_predict)).sum().toLongArray()[0]*1.0 / X.getShape().getShape()[0])*100;
        System.out.println("\nAccuracy: " + ac);
        System.exit(0);
    }
}
