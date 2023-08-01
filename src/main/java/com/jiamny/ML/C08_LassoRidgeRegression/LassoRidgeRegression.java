package com.jiamny.ML.C08_LassoRidgeRegression;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.jiamny.Utils.CombineAndArrangement;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.NumericColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.ScatterPlot;

import java.awt.geom.QuadCurve2D;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import static com.jiamny.Utils.HelperFunctions.range;
import static com.jiamny.Utils.UtilFunctions.train_test_split;

interface Regularization {
    public NDArray call(NDArray w);
    public NDArray grad(NDArray w);
}

class LassoRegularization implements Regularization {
    private double alpha;

    /*
        :param alpha:
        * When 0, the lasso regression turns into Linear Regression
        * When increases towards infinity, it turns features coefficients into zero.
        * Try out different value to find out optimized values.
     */
    public LassoRegularization(double alpha) {
        this.alpha = alpha;
    }

    /*
    :param w: Weight vector
    :return: Penalization value for MSE
     */
    public NDArray call(NDArray w) {
        // self.alpha * torch.norm(w, p=1)
        return ((w.abs()).sum()).mul(alpha);
    }

    /*
    :param w: weight vector
        :return: weight update based on sign value, it helps in removing coefficients from W vector
        torch.sign:
        a
        tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
        torch.sign(a)
        tensor([ 1., -1.,  0.,  1.])
     */
    public NDArray grad(NDArray w) {
        return w.sign().mul(alpha);
    }
}

class RidgeRegularization implements Regularization {
    private double alpha;

    /*
        :param learning_rate: constant step while updating weight
        :param epochs: Number of epochs the data is passed through the model
        Initalizing regularizer for Lasso Regression.
     */
    public RidgeRegularization(double alpha) {
        this.alpha = alpha;
    }

    /*
    :param w: Weight vector
    :return: Penalization value for MSE
     */
    public NDArray call(NDArray w) {
        // self.alpha * 0.5 * torch.mm(w.T, w)
        return (w.transpose().matMul(w)).mul(alpha * 0.5);
    }

    /*
    :param w: weight vector
    :return: weight update based on sign value, it helps in reducing the coefficient values from W vector
     */
    public NDArray grad(NDArray w) {
        return w.mul(alpha);
    }
}

public class LassoRidgeRegression {
    private double lr;
    private int epochs;
    private Regularization regularization;
    private NDArray w;
    private ArrayList<Double> [] training_error = new ArrayList[] {
            new ArrayList<>(), new ArrayList<>()};

    /*
    :param learning_rate: constant step while updating weight
    :param epochs: Number of epochs the data is passed through the model
    Initalizing regularizer for Lasso Regression.
     */
    public LassoRidgeRegression(double learning_rate, int epochs, String regression_type) {
        lr = learning_rate;
        this.epochs = epochs;
        if (regression_type.equalsIgnoreCase("lasso"))
            regularization = new LassoRegularization(1.0);
        else
            regularization = new RidgeRegularization(2.0);
    }

    public LassoRidgeRegression(){}

    public NDArray normalization(NDArray X) {
        /*
        :param X: Input tensor
        :return: Normalized input using l2 norm.
         */
        // torch.norm(X,p=2,dim=-1)
        NDArray l2 = X.norm(2, new int[] {1}, false);
        //l2[l2 ==0]=1
        l2.set(l2.eq(0), 1);
        // l2.unsqueeze(1)
        NDArray l2s = l2.reshape(new Shape(l2.getShape().getShape()[0], 1));
        return X.div(l2s);
    }

    /*
    It creates polynomial features from existing set of features. For instance,
    X_1, X_2, X_3 are available features, then polynomial features takes combinations of
    these features to create new feature by doing X_1*X_2, X_1*X_3, X_2*X3.

    combinations output: [(), (0,), (1,), (2,), (3,), (0, 0), (0, 1), (0, 2), (0, 3),
    (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    :param X: Input tensor (For Iris Dataset, (150, 4))
    :param degree: Polynomial degree of 2, i.e we'll have product of two feature vector at max.
    :return: Output tensor (After adding polynomial features, the number of features increases to 15)
    */

    public NDArray polynomial_features(NDArray X, int degree) {
        int n_samples = (int)X.getShape().getShape()[0], n_features = (int)X.getShape().getShape()[1];

        //def index_combination():
        //combinations = [combinations_with_replacement(range(n_features), i) for i in range(0, degree+1)]
        //flat_combinations = [item for sublists in combinations for item in sublists]
        //return flat_combinations
        int [] com = range(0, n_features);
        int [] degrees = range(1, degree+1);

        ArrayList<int []> combinations = new ArrayList<>();
        for( int i : degrees) {
            CombineAndArrangement.reset();
            CombineAndArrangement.repeatableArrangement(i, com);

            for( var t : CombineAndArrangement.results) {
                combinations.add(t);
            }
        }

        NDManager manager = NDManager.newBaseManager();
        //combinations = index_combination()
        int n_output_features = combinations.size() + 1;
        NDArray X_new = manager.zeros(new Shape(n_samples, n_output_features), DataType.FLOAT64);

        //for i, index_combs in enumerate(combinations):
        //X_new[:, i] = torch.prod(X[:, index_combs], dim=1)

        // empty combination
        NDArray eptC = manager.ones(new Shape(n_samples), DataType.FLOAT64);
        X_new.set(new NDIndex(":,0"), eptC);
        for(int i = 0; i < combinations.size(); i++) {
            NDList t = new NDList();
            int [] cmb = combinations.get(i);
            for( int j = 0; j < cmb.length; j++){
                t.add(X.get(":," + cmb[j]));
            }
            //System.out.println(NDArrays.stack(t, 1).prod(new int [] {1}));
            X_new.set(new NDIndex(":," + (i + 1)),
                    NDArrays.stack(t, 1).prod(new int [] {1}));
        }
        //X_new = X_new.type(torch.DoubleTensor)
        return X_new;
    }

    /*
    :param n_features: Number of features in the data
    :return: creating weight vector using uniform distribution.
     */
    public void weight_initialization(int n_features) {
        NDManager manager = NDManager.newBaseManager();
        float limit = (float)(1.0 / Math.sqrt(n_features * 1.0));
        //self.w = torch.FloatTensor((n_features,)).uniform(-limit, limit)
        //this.w = torch.distributions.uniform.Uniform(-limit, limit).sample((n_features, 1))
        this.w = manager.randomUniform(-limit, limit, new Shape(n_features, 1), DataType.FLOAT64);
        //this.w = this.w.type(torch.DoubleTensor)
    }

    /*
    :param X: Input tensor
    :param y: ground truth tensor
    :return: updated weight vector for prediction
     */
    public void fit(NDArray X, NDArray y) {
        this.training_error[0].clear();
        this.training_error[1].clear();
        weight_initialization((int)X.getShape().getShape()[1]);
        for( var epoch : range(1, this.epochs+1) ){
            NDArray y_pred = NDArrays.matMul(X, this.w);
            //NDArray mse = torch.mean(0.5 * (y - y_pred) * * 2 + self.regularization(self.w))
            NDArray mse = (((y.sub(y_pred)).pow(2)).mul(0.5)).add(
                    this.regularization.call(this.w));
            //self.training_error[epoch] = mse.item()
            this.training_error[0].add(epoch*1.0);
            this.training_error[1].add(mse.toDoubleArray()[0]);
            //grad_w = torch.mm(-(y - y_pred).T, X).T + self.regularization.grad(self.w)
            NDArray t = (y.sub(y_pred).transpose()).mul(-1.0);
            NDArray grad_w = (t.matMul(X).transpose()).add(this.regularization.grad(this.w));
            this.w = this.w.sub(grad_w.mul(this.lr));
        }
    }

    /*
     :param X: input tensor
     :return: predicted output using learned weight vector
     */
    public NDArray predict(NDArray X) {
        NDArray y_pred = NDArrays.matMul(X, this.w);
        return y_pred;
    }

    public ArrayList<NDArray> diabetes_data(String fdir) {
        ArrayList<String> contents = new ArrayList<>();
        double [][]  Yd = null;
        int ncol = 0;
        try {
            File fr = new File(fdir + "/" + "diabetes_data.csv");
            BufferedReader in = null;

            in = new BufferedReader( new InputStreamReader(new FileInputStream(fr)));
            String line = in.readLine();
            String[] curLine = line.strip().split(" ");
            ncol = curLine.length;
            contents.add(line.strip());
            while( (line = in.readLine()) != null) {
                contents.add(line.strip());
            }
            in.close();

            Yd  = new double[contents.size()][1];
            int r = 0;
            fr = new File(fdir + "/" + "y.csv");
            in = null;
            in = new BufferedReader( new InputStreamReader(new FileInputStream(fr)));
            contents.add(line.strip());
            while( (line = in.readLine()) != null) {
                Yd[r][0] = Double.parseDouble(line.strip());
            }
            in.close();
        } catch(Exception e) {
            e.printStackTrace();
        }

        double [][] Xd = new double[contents.size()][ncol];

        for( int j = 0; j < contents.size(); j++ ) {
            if (contents.get(j).length() < 1)
                continue;
            String[] curLine = contents.get(j).strip().split(" ");
            for(int i = 0; i < curLine.length; i++) {
                Xd[j][i] = Double.parseDouble(curLine[i]);
            }
        }

        NDManager manager = NDManager.newBaseManager();
        ArrayList<NDArray> xy = new ArrayList<>();
        xy.add(manager.create(Xd).toType(DataType.FLOAT64, false));
        xy.add(manager.create(Yd).toType(DataType.FLOAT64, false));
        return xy;
    }

    public static void main(String[] args) {
/*
        NDManager manager = NDManager.newBaseManager();
        NDArray array = manager.create(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new Shape(2, 3));
        System.out.println(array);
        NDList t = new NDList();
        t.add(array.get(":,0"));
        t.add(array.get(":,0"));
        System.out.println(NDArrays.stack(t, 1).prod(new int [] {1}));

        NDArray X_new = manager.zeros(new Shape(2, 3), DataType.FLOAT64);
        System.out.println("X_new\n" + X_new);

        // empty combination
        NDArray eptC = manager.ones(new Shape(2), DataType.FLOAT64);
        System.out.println("eptC\n" + eptC);
        X_new.set(new NDIndex(":,0"), eptC);
        System.out.println(X_new);
        X_new.set(new NDIndex(":,1"),
                NDArrays.stack(t, 1).prod(new int [] {1}));
        System.out.println(X_new);
/*
        int [] com = range(0, 10);
        CombineAndArrangement.reset();
        CombineAndArrangement.repeatableArrangement(1, com);
        for( var i : CombineAndArrangement.results )
            System.out.print(Arrays.toString(i) + ",");
*/

        NDManager manager = NDManager.newBaseManager();
        LassoRidgeRegression lrr = new LassoRidgeRegression();
        String fdir = "./data/ML/diabetes_data";
        ArrayList<NDArray> xy  = lrr.diabetes_data(fdir);
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
        float limit = 0.3f;
        System.out.println("-limit " + -limit);

        LassoRidgeRegression regression = new LassoRidgeRegression(
                0.0001, 3000, "lasso");
        regression.fit(regression.normalization(regression.polynomial_features(Xtrain, 1)), Ytrain);
        NDArray y_pred = regression.predict(
                regression.normalization(regression.polynomial_features(Xtest, 1)));

        System.out.println("y_pred: " + y_pred.get("0:10,:"));

        NumericColumn<?> xw = DoubleColumn.create("epoch",
                regression.training_error[0].stream().mapToDouble(x->x).toArray());
        NumericColumn<?> yl = DoubleColumn.create("training_error",
                regression.training_error[1].stream().mapToDouble(x->x).toArray());

        Table data = Table.create(xw, yl);

        Plot.show(ScatterPlot.create("LassoRidgeRegression",
                data, "epoch", "training_error"));

        System.exit(0);
    }
}
