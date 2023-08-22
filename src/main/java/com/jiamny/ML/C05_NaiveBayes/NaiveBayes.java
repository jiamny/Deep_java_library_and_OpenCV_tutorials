package com.jiamny.ML.C05_NaiveBayes;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import static com.jiamny.Utils.UtilFunctions.loadIrisData;
import static com.jiamny.Utils.UtilFunctions.train_test_split;

public class NaiveBayes {

    /**
     * why e - epsilon ?
     * If the ratio of data variance between dimensions is too small, it
     * will cause numerical errors. To address this, we artificially
     * boost the variance by epsilon, a small fraction of the standard
     * deviation of the largest dimension.
     *
     *         @param X: input tensor
     *         @param y: target tensor
     */
    public int total_samples;      // Number of Samples
    public int feature_count;      // Number of Features
    public HashMap<Integer, NDArray> mu;    // mean
    public HashMap<Integer, NDArray> sigma; // variance
    public double  e;               //epsilon
    public int n_classes;           // number of classes
    public HashMap<Integer, Double> prior_probability_X;

    public NaiveBayes(NDArray X, NDArray y) {
        total_samples = (int)X.getShape().getShape()[0];
        feature_count = (int)X.getShape().getShape()[1];
        mu = new HashMap<>();
        sigma = new HashMap<>();
        prior_probability_X = new HashMap<>();
        e = 1e-4;
        n_classes = y.unique().size();
    }

    /*
    Bayes Theorem:
        P(Y|X) = P(X|Y) * P(Y) / P(X)

        :type mu: dict
        :type sigma: dict
        :type prior_probability: dict
        :describe mu: keys are class label and values are feature's mean values.
        :describe sigma: keys are class label and values are feature's variance values.
        :describe prior probability of x: It calculates the prior prabability of X for each class. P(X).
        :return:
     */
    public void find_mu_and_sigma(NDArray X, NDArray y) {
        NDManager manager = NDManager.newBaseManager();

        for(int cls = 0; cls < n_classes; cls++) {
            //System.out.println("y.eq(cls): " + y.eq(cls).toIntArray());
            ArrayList<Integer> cidx = new ArrayList<>();
            for(int i = 0; i < total_samples; i++)
                if( y.get(i).toIntArray()[0] == cls)
                    cidx.add(i);

            NDArray cIdx = manager.create(cidx.stream().mapToInt(m -> m).toArray()).toType(DataType.INT32, false);
            NDArray X_class = X.get(cIdx);

            int [] idx = new int[]{0};
            NDArray mn = X_class.mean(idx);
            NDArray var = ((X_class.sub(mn).square()).sum(idx)).div(X_class.getShape().getShape()[0] - 1);

            mu.put(cls, mn);
            sigma.put(cls, var);
            prior_probability_X.put(cls,
                    (1.0*X_class.getShape().getShape()[0] / X.getShape().getShape()[0]));
            //mu[cls] = torch.mean(X_class, dim = 0)
            //sigma[cls] = torch.var(X_class, dim = 0)
            //self.prior_probability_X[cls] = X_class.shape[0] / X.shape[0]
        }
        //Arrays.stream(prior_probability_X).forEach(System.out::println);
    }

    /*
    :return: Multivariate normal(gaussian) distribution - Maximum Likelihood Estimation
        https://www.statlect.com/fundamentals-of-statistics/multivariate-normal-distribution-maximum-likelihood

        Log Likelihood Function = Constant - probability
     */
    public NDArray gaussian_naive_bayes(NDArray X, NDArray mu, NDArray sigma) {
        NDManager manager = NDManager.newBaseManager();
        // - feature_count / 2 * manager.log(2 * torch.tensor(np.pi)) - 0.5 * torch.sum(torch.log(sigma+self.e))
        NDArray m = ((manager.create(Math.PI).mul(2)).log()).mul(-1.0*feature_count/2);
        NDArray t = ((((sigma).add(e)).log()).sum()).mul(0.5);
        NDArray constant = m.sub(t);
        // 0.5 * torch.sum(torch.pow(X-mu, 2) / (sigma + self.e), dim=1)
        int [] idx = new int[]{1};
        //NDArray probability = ((X.sub(mu).pow(2)).div(sigma.add(e))).sum(idx).mul(0.5);
        NDArray Z = (X.sub(mu)).pow(2);
        NDArray probability = (Z.div(sigma.add(e))).sum(idx).mul(0.5);
        return constant.sub(probability);
    }

    /*
        Calculating probabilities for each sample input in X using prior probability
        and gaussian density function.

        torch.argmax: To find the class with max-probability.

        Note: We are calculate log probabilities as in Sklearn's predict_log_proba, that why we have + sign between
        prior probabilites and likelihood (class probability).

        :return:
    */
    public NDArray predict_probability(NDArray X) {
        NDManager manager = NDManager.newBaseManager();

        NDArray probabilities = manager.zeros(new Shape(X.getShape().getShape()[0], n_classes));
        for(int cls = 0; cls < n_classes; cls++) {
            NDArray class_probability = gaussian_naive_bayes(X, mu.get(cls), sigma.get(cls));
            //System.out.println("class_probability: \n" + class_probability);
            //probabilities[:,cls] = class_probability + torch.log(torch.scalar_tensor(self.prior_probability_X[cls]))
            NDArray V = class_probability.add(manager.create(prior_probability_X.get(cls)).log());
            probabilities.set(new NDIndex("...,"+cls), V);
        }

        //System.out.println("probabilities: " + probabilities.get("0:5,:"));
        return probabilities.argMax(1); // torch.argmax(probabilities, dim=1)
    }

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();

        String f = "data/ML/iris.data";
        ArrayList<NDArray> res = loadIrisData(f);
        res = train_test_split(res.get(0), res.get(1), 0.2);

        NDArray X_train = res.get(0), y_train = res.get(1);
        NDArray X_test = res.get(2), y_test = res.get(3);

        NaiveBayes nb = new NaiveBayes(X_train, y_train.flatten());
        nb.find_mu_and_sigma(X_train, y_train.flatten());

        NDArray y_pred = nb.predict_probability(X_test);
        int correct = (int)y_pred.eq(y_test.flatten()).sum().toLongArray()[0];
        System.out.println("correct: " + correct + " num: " + y_test.getShape().getShape()[0]);
        System.out.println("Accuracy Score: " + (1.0*correct/y_test.getShape().getShape()[0]));

        System.exit(0);
    }
}
