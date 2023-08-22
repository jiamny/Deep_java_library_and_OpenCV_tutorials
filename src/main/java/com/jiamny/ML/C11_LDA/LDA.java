package com.jiamny.ML.C11_LDA;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.jiamny.ML.C05_NaiveBayes.NaiveBayes;

import java.util.ArrayList;
import static com.jiamny.Utils.UtilFunctions.*;
import static com.jiamny.Utils.UtilFunctions.train_test_split;

public class LDA {
    private NDArray w;

    public LDA() {
        this.w = null;
    }

    /*
     :param X: Input tensor
     :return: cavariance of input tensor
     */
    public NDArray covariance_matrix(NDArray X) {
        NDArray centering_X = NDArrays.sub(X, X.mean(new int[]{0}));  //.mean(X, dim=0)
        //cov = torch.mm(centering_X.T, centering_X) / (centering_X.shape[0] - 1)
        int cls = (int) (centering_X.getShape().getShape()[0]);
        NDArray c = NDArrays.matMul(centering_X.transpose(), centering_X);
        //System.out.println("c:\n" + c);
        NDArray cov = NDArrays.div(c, cls - 1);
        //System.out.println("cov-1:\n" + cov);
        return cov;
    }

    /*
        :param X: Input tensor
        :param y: output tensor
        :return: transformation vector - to convert high dimensional input space into lower dimensional
        subspace.
        X1, X2 are samples based on class. cov_1 and cov_2 measures how features of samples of each class are related.
     */
    public void fit(NDArray X, NDArray y) {
        NDArray X1 = X.get(y.eq(0).squeeze());
        NDArray X2 = X.get(y.eq(1).squeeze());
        NDArray cov_1 = covariance_matrix(X1);
        NDArray cov_2 = covariance_matrix(X2);
        NDArray cov_total = NDArrays.add(cov_1, cov_2);
        NDArray mean1 = X1.mean(new int[]{0}); // dim=0)
        NDArray mean2 = X2.mean(new int[]{0});
        NDArray mean_diff = NDArrays.sub(mean1,  mean2);

        System.out.println("cov_total: " + cov_total.get("0:3,:"));
        // Determine the vector which when X is projected onto it best separates the
        // data by class. w = (mean1 - mean2) / (cov1 + cov2)
        // torch.mm(torch.pinverse(cov_total), mean_diff.unsqueeze(1))
        this.w = NDArrays.matMul(cov_total.inverse(),  mean_diff.expandDims(1));
    }

    public NDArray transform(NDArray X, NDArray y) {
        fit(X, y);
        NDArray X_transformed = NDArrays.matMul(X, this.w);
        return X_transformed;
    }

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        String fName = "./data/ML/breast_cancer.csv";
        ArrayList<NDArray> xy  = load_breast_cancer(fName);
        NDArray X = xy.get(0), y = xy.get(1);
        System.out.println(X.getShape().toString());
        System.out.println(y.getShape().toString());
        System.out.println("X " + X.get("0:3,..."));

        ArrayList<NDArray> res = train_test_split(X, y, 0.2);
        NDArray Xtrain = res.get(0), Ytrain=res.get(1), Xtest = res.get(2), Ytest=res.get(3);
        System.out.println("Xtrain: " + Xtrain);
        System.out.println("Ytrain: " + Ytrain.unique().getShapes().toString());
        System.out.println("Xtest: " + Xtest);
        System.out.println("Ytest: " + Ytest);

        LDA lda = new LDA();
        NDArray X_transformed = lda.transform(Xtrain, Ytrain);
        System.out.println("X_transformed: " + X_transformed.getShape().toString());

        NaiveBayes GNB = new NaiveBayes(X_transformed, Ytrain);
        GNB.find_mu_and_sigma(X_transformed, Ytrain);

        NDArray X_test_transformed = lda.transform(Xtest, Ytest);
        NDArray y_pred = GNB.predict_probability(X_test_transformed).toType(DataType.INT32, false);

        int correct = (int)y_pred.eq(Ytest.squeeze()).sum().toLongArray()[0];
        System.out.println("correct: " + correct + " num: " + Ytest.getShape().getShape()[0]);
        System.out.println("Accuracy Score: " + (1.0*correct/Ytest.getShape().getShape()[0]));

        System.exit(0);
    }
}
