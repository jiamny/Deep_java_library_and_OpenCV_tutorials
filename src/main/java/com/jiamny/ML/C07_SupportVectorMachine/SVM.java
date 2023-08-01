package com.jiamny.ML.C07_SupportVectorMachine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;

import static com.jiamny.Utils.UtilFunctions.shuffle;
import static com.jiamny.Utils.UtilFunctions.train_test_split;

class SupportVectorMachine {
    int total_samples, features_count, n_classes;
    double learning_rate, C;

    public SupportVectorMachine(NDArray X, NDArray y, double C) {
        total_samples = (int) X.getShape().get(0);
        features_count = (int) X.getShape().get(1);
        n_classes = y.unique().size();
        learning_rate = 0.001;
        this.C = C;
    }

    /*
        C parameter tells the SVM optimization how much you want to avoid misclassifying each training
        example. For large values of C, the optimization will choose a smaller-margin hyperplane if that
        hyperplane does a better job of getting all the training points classified correctly. Conversely,
        a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane,
        even if that hyperplane misclassifies more points. For very tiny values of C, you should get
        misclassified examples, often even if your training data is linearly separable.

        :param X:
        :param W:
        :param y:
        :return:
     */
    public NDArray loss(NDArray X, NDArray W, NDArray y) {
        int num_samples = (int) X.getShape().get(0);
        //distances = 1 - y * (torch.mm(X, W.T))
        /*
        System.out.println("y.shape " + y.getShape().toString() +
                " X.shape " + X.getShape().toString() +
                " W.shape " + W.getShape().toString());
         */
        NDArray distances = (y.toType(DataType.FLOAT64, false).mul(X.matMul(W.transpose()))).mul(-1.0);
        distances = distances.add(1.0);
        NDArray lsz = distances.lt(0.0);
        //distances[distances < 0] = 0
        distances.set(lsz, 0.0);

        NDArray hinge_loss = (distances.sum().div(num_samples).floor()).mul(C); //self.C * (torch.sum(distances) // num_samples)
        // 1 / 2 * torch.mm(W, W.T) + hinge_loss
        NDArray cost = (W.matMul(W.transpose()).mul(1.0 / 2)).add(hinge_loss);
        return cost;
    }

    /*
    :param W: Weight Matrix
    :param X: Input Tensor
    :param y: Ground truth tensor
    :return: change in weight
     */
    public NDArray gradient_update(NDArray W, NDArray X, NDArray y) {
        NDManager manager = NDManager.newBaseManager();
        /*
        System.out.println("y.shape " + y.getShape().toString() +
                " X.shape " + X.getShape().toString() +
                " W.shape " + W.getShape().toString());
         */
        //distance = 1 - (y * torch.mm(X, W.T))
        NDArray distances = X.matMul(W.transpose()).mul(-1.0 * y.toIntArray()[0]);
        distances = distances.add(1.0);

        NDArray dw = manager.zeros(new Shape(1, X.getShape().get(1)), DataType.FLOAT64);
        //dw = torch.zeros((1, X.shape[1]),dtype=torch.double)
        for (long idx = 0; idx < distances.getShape().get(0); idx++) {
            double dist = distances.get(idx).toDoubleArray()[0];
            NDArray di = null;
            if (Math.max(0., dist) == 0.) {
                di = W;
            } else {
                // di = W - (self.C * y[idx] * X[idx])
                di = W.sub(X.get(idx)).mul(C*y.toIntArray()[0]);
            }

            dw.addi(di);
        }

        dw = dw.div(y.getShape().get(0));
        return dw;
    }

    /*
    :param X: Input Tensor
    :param y: Output tensor
    :param max_epochs: Number of epochs the complete dataset is passed through the model
    :return: learned weight of the svm model
     */
    public NDArray fit(NDArray X, NDArray y, int max_epochs) {
        NDManager manager = NDManager.newBaseManager();
        NDArray t = manager.randomUniform(0.0f, 1.0f, new Shape(1, X.getShape().get(1)), DataType.FLOAT64);
        //weight = torch.randn((1, X.shape[1]), dtype=torch.double) * torch.sqrt(torch.scalar_tensor(1./X.shape[1]))
        NDArray weight = t.mul((manager.create(1. / X.getShape().get(1))).sqrt());
        double cost_threshold = 0.0001;
        double previous_cost = Double.POSITIVE_INFINITY; //float('inf')
        int nth = 0;

        for (int epoch = 1; epoch < (max_epochs + 1); epoch++) {
            ArrayList<NDArray> res = shuffle(X, y);
            X = res.get(0);
            y = res.get(1);

            for (int idx = 0; idx < (int) X.getShape().get(0); idx++) {
                //weight_update = self.gradient_update(weight, torch.tensor(x).unsqueeze(0), y[idx])
                NDArray x = X.get(idx);
                // x.unsqueeze(0)
                x = x.reshape( new Shape(1, x.getShape().get(0)) );
                NDArray weight_update = gradient_update(weight, x, y.get(idx));
                weight = weight.subi(weight_update.mul(learning_rate));
            }

            if (epoch % 100 == 0) {
                double cost = loss(X, weight, y).toDoubleArray()[0];
                System.out.printf("Loss at epoch %d : %.4f\n", epoch, cost);
                if (Math.abs(previous_cost - cost) < cost_threshold * previous_cost)
                    return weight;
                previous_cost = cost;
                nth += 1;
            }
        }
        return weight;
    }
}
public class SVM {

    public ArrayList<NDArray> load_breast_cancer(String fName) {
        ArrayList<String> contents = new ArrayList<>();
        int ncol = 0;
        try {
            File fr = new File(fName);
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

        double [][] Xd = new double[contents.size()][ncol - 2];
        int [][]  Yd  = new int[contents.size()][1];

        for( int j = 0; j < contents.size(); j++ ) {
            if (contents.get(j).length() < 1)
                continue;
            String[] curLine = contents.get(j).strip().split(",");
            // skip
            for(int i = 2; i < curLine.length; i++) {
                Xd[j][i-2] = Double.parseDouble(curLine[i]);
            }
            if( curLine[1].equalsIgnoreCase("M") )
                Yd[j][0] = 0;
            else
                Yd[j][0] = 1;
        }
        NDManager manager = NDManager.newBaseManager();
        ArrayList<NDArray> xy = new ArrayList<>();
        xy.add(manager.create(Xd).toType(DataType.FLOAT64, false));
        xy.add(manager.create(Yd).toType(DataType.INT32, false));
        return xy;
    }

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        SVM svm = new SVM();
        String fName = "./data/ML/breast_cancer.csv";
        ArrayList<NDArray> xy  = svm.load_breast_cancer(fName);
        NDArray X = xy.get(0), y = xy.get(1);
        System.out.println(X.getShape().toString());
        System.out.println(y.getShape().toString());
        System.out.println("X " + X.get("0:3,..."));

        /******************************************************
         * MinMaxScaler normalize data
         ******************************************************/
        int [] idx = new int[]{0};
        NDArray min = X.min(idx);
        NDArray max = X.max(idx);
        NDArray XX = (X.sub(min)).div(max.sub(min));

        System.out.println("XX " + XX.get("0:3,..."));
        System.out.println("XX.sum " + XX.sum(idx));

        NDArray bias = manager.ones(new Shape(XX.getShape().get(0), 1), DataType.FLOAT64);
        XX = bias.concat(XX, 1);  //torch.cat((bias, X), dim=1)
        System.out.println("XX " + XX.get("0:3,..."));

        ArrayList<NDArray> res = train_test_split(XX, y, 0.2);
        NDArray Xtrain = res.get(0), Ytrain=res.get(1), Xtest = res.get(2), Ytest=res.get(3);
        System.out.println("Xtrain: " + Xtrain);
        System.out.println("Ytrain: " + Ytrain.getShape().toString());
        System.out.println("Xtest: " + Xtest);
        System.out.println("Ytest: " + Ytest);

        SupportVectorMachine spt = new SupportVectorMachine(Xtrain, Ytrain, 1.0);
        int num_epochs = 1000;
        NDArray model_weights = spt.fit(Xtrain, Ytrain, num_epochs);
        NDArray y_pred = (Xtest.matMul(model_weights.transpose())).sign(); //torch.sign(torch.mm(x_test, model_weights.T))
        y_pred = y_pred.toType(DataType.INT32, false);
        /*
        System.out.print("y_pred: " );
        System.out.print("[ ");
        Arrays.stream(y_pred.toIntArray()).forEach(num -> System.out.print(num + " "));
        System.out.println("]");
        System.out.print("Ytest: " );
        System.out.print("[ ");
        Arrays.stream(Ytest.toIntArray()).forEach(num -> System.out.print(num + " "));
        System.out.println("]");
         */

        int correct = (int)y_pred.eq(Ytest).sum().toLongArray()[0];
        System.out.println("correct: " + correct + " num: " + Ytest.getShape().getShape()[0]);
        System.out.println("Accuracy Score: " + (1.0*correct/Ytest.getShape().getShape()[0]));

        System.exit(0);
    }
}