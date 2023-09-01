package com.jiamny.ML.C10_GaussianMixtureModel;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.jiamny.Utils.DeterminantCalc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import static com.jiamny.Utils.DeterminantCalc.Determinant;
import static com.jiamny.Utils.UtilFunctions.*;

public class GaussianMixtureModel {

    /*
    :param k: the number of clusters the algorithm will form.
    :param max_epochs: The number of iterations the algorithm will run for if it does
    not converge before that.
    :param tolerance: float
    If the difference of the results from one iteration to the next is
    smaller than this value we will say that the algorithm has converged.
     */
    private int k, max_epochs;
    private double tolerance;
    private ArrayList<HashMap<String, NDArray>> parameters;
    private NDArray prior, responsibility, sample_assignments;
    private NDList responsibilities;

    public GaussianMixtureModel(int k, int max_epochs, double tolerance) {
        this.k = k;
        this.parameters = new ArrayList<>();
        this.max_epochs = max_epochs;
        this.tolerance = tolerance;
        responsibilities = new NDList();
    }

    /*
     :param X: Input tensor
     :return: Normalized input using l2 norm.
     */
    public NDArray normalization(NDArray X) {

        int dimidx = X.getShape().getShape().length - 1;
        int dim = (int) (X.getShape().getShape()[dimidx]);
        NDArray l2 = X.norm(2, new int[]{dim}, true);    //torch.norm(X, p=2, dim=-1)
        l2.set(l2.eq(0), 1);
        return X.div(l2.expandDims(1));                     //unsqueeze(1));
    }

    /*
     :param X: Input tensor
     :return: cavariance of input tensor
     */
    public NDArray covariance_matrix(NDArray X) {
        NDArray centering_X = NDArrays.sub(X, X.mean(new int[]{0}));  //.mean(X, dim=0)
        //cov = torch.mm(centering_X.T, centering_X) / (centering_X.shape[0] - 1)
        int cls = (int)(centering_X.getShape().getShape()[0]);
        NDArray c = NDArrays.matMul(centering_X.transpose(), centering_X);
        //System.out.println("c:\n" + c);
        NDArray cov = NDArrays.div(c, cls - 1);
        //System.out.println("cov-1:\n" + cov);
        return cov;
    }

    /*
        Since we are using iris dataset, we know the no. of class is 3.
        We create three gaussian distribution representing each class with
        random sampling of data to find parameters like μ and 𝚺/N (covariance matrix)
        for each class
        :param X: input tensor
        :return: 3 randomly selected mean and covariance of X, each act as a separate cluster
    */
    public void random_gaussian_initialization(NDArray X) {
        NDManager manager = NDManager.newBaseManager();

        int n_samples = (int) (X.getShape().getShape()[0]);
        int[] ridx = new int[n_samples];
        for (int i = 0; i < n_samples; i++)
            ridx[i] = i;

        prior = NDArrays.mul((1 / k), manager.ones(new Shape(k), DataType.FLOAT64));
        for (int cls = 0; cls < k; cls++) {
            HashMap<String, NDArray> parameter = new HashMap<>();
            ridx = ShuffleArray(ridx);
            parameter.put("mean", X.get(ridx[0]));
            //parameter['mean'] = X[torch.randperm(n_samples)[:1]]
            parameter.put("cov", covariance_matrix(X));
            parameters.add(parameter);
        }
    }

    /*
        Checkout the equation from Multi-Dimensional Model from blog link posted above.
        We find the likelihood of each sample w.r.t to the parameters initialized above for each separate cluster.
        :param X: Input tensor
        :param parameters: mean, cov of the randomly initialized gaussian
        :return: Likelihood of each sample belonging to a cluster with random initialization of mean and cov.
        Since it is a multivariate problem we have covariance and not variance.
     */
    public NDArray multivariate_gaussian_distribution(NDArray X, HashMap<String, NDArray> parameters) {
        NDManager manager = NDManager.newBaseManager();

        int n_features = (int) (X.getShape().getShape()[1]);
        NDArray mean = parameters.get("mean");
        NDArray cov = parameters.get("cov");
        //System.out.println(cov);

        int rows = (int)(cov.getShape().getShape()[0]);
        int cols = (int)(cov.getShape().getShape()[1]);
        double [][] t = new double[rows][cols];
        for( int r = 0; r < rows; r++)
            t[r] = (double[])Arrays.copyOf(cov.get(r).toDoubleArray(), cols);

        // determinant = torch.det(cov)
        NDArray determinant = manager.create(DeterminantCalc.Determinant(t));
        NDArray likelihoods = manager.zeros(new Shape(X.getShape().getShape()[0]));

        for (int i = 0; i < (int) (X.getShape().getShape()[0]); i++) {
            NDArray sample = X.get(i);
            //dim = torch.scalar_tensor(n_features, dtype = torch. float)
            //NDArray dim = manager.create(n_features).toType(DataType.FLOAT64, false);
            //coefficients = 1.0 / torch.sqrt(torch.pow((2.0 * math.pi), dim) * determinant)
            double p = Math.pow(2.0 * Math.PI, n_features);
            NDArray a =  manager.create(p).mul(determinant).sqrt();
            NDArray coefficients = NDArrays.div(1.0, a);
            //exponent = torch.exp(-0.5 * torch.mm(torch.mm((sample - mean), torch.pinverse(cov)), (sample - mean).T))
            //torch.mm((sample - mean), torch.pinverse(cov))
            NDArray p1 = NDArrays.matMul(sample.sub(mean), cov.inverse());
            NDArray p2 = NDArrays.matMul(p1, sample.sub(mean).transpose());
            NDArray exponent = (NDArrays.mul(-0.5, p2)).exp();
            likelihoods.set(new NDIndex(i), coefficients.mul(exponent)); //coefficients * exponent;
        }
        return likelihoods;
    }

    /*
        Previously, we have initialized 3 different mean and covariance in random_gaussian_initialization(). Now around
        each of these mean and cov, we see likelihood of the each sample using multivariate gaussian distribution.
        :param X:
        :return: Storing the likelihood of each sample belonging to a cluster with random initialization of mean and cov.
        Since it is a multivariate problem we have covariance and not variance.
     */
    public NDArray get_likelihood(NDArray X) {
        NDManager manager = NDManager.newBaseManager();
        int n_samples = (int) (X.getShape().getShape()[0]);
        NDArray likelihoods_cls = manager.zeros(new Shape(n_samples, k)).toType(DataType.FLOAT64, false);

        for (int cls = 0; cls < k; cls++) {
            NDIndex idx = new NDIndex(":," + cls);
            likelihoods_cls.set(idx,
                    multivariate_gaussian_distribution(X, parameters.get(cls)));
        }
        return likelihoods_cls;
    }

    /*
        Expectation Maximization Algorithm is used to find the optimized value of randomly initialized mean and cov.
        Expectation refers to probability. Here, It calculates the probabilities of X belonging to different cluster.
        :param X: input tensor
        :return: Max probability of each sample belonging to a particular class.
     */
    public void expectation(NDArray X) {
        //System.out.println("get_likelihood(X): " + get_likelihood(X).getShape().toString());
        //System.out.println("prior: " + prior.getShape().toString());
        NDArray weighted_likelihood = NDArrays.mul(get_likelihood(X), prior);
        //System.out.println("weighted_likelihood: " + weighted_likelihood.getShape().toString());
        //torch.sum(weighted_likelihood, dim=1).unsqueeze(1)
        NDArray sum_likelihood = weighted_likelihood.sum(new int[]{1}).expandDims(1);

        // Determine responsibility as P(X|y)*P(y)/P(X)
        // responsibility stores each sample's probability score corresponding to each class
        responsibility = NDArrays.div(weighted_likelihood, sum_likelihood);

        // Assign samples to cluster that has largest probability
        sample_assignments = responsibility.argMax(1); // argmax( dim=1)

        // Save value for convergence check
        responsibilities.add(responsibility.max(new int[]{1}));    //append(torch.max(self.responsibility, dim=1))
    }

    /*
        Iterate through clusters and updating mean and covariance.
        Finding updated mean and covariance using probability score of each sample w.r.t each class
        :param X:
        :return: Updated mean, covariance and priors
     */
    public void maximization(NDArray X) {
        for (int i = 0; i < k; i++) {
            NDArray resp = responsibility.get(":," + i).expandDims(1);   //[:,i].unsqueeze(1)
            // torch.sum(resp * X, dim = 0) / torch.sum(resp)
            NDArray mean = NDArrays.div(resp.mul(X).sum(new int[]{0}), resp.sum());
            // covariance = torch.mm((X - mean).T, (X - mean) * resp) / resp.sum()
            NDArray covariance = NDArrays.div(NDArrays.matMul(X.sub(mean).transpose(),
                    NDArrays.mul(X.sub(mean), resp)), resp.sum());
            // parameters[i]['mean'], self.parameters[i]['cov'] = mean.unsqueeze(0), covariance
            parameters.get(i).put("mean", mean.expandDims(0));
            parameters.get(i).put("cov", covariance);
        }
        int n_samples = (int) (X.getShape().getShape()[0]);
        prior = NDArrays.div(responsibility.sum(new int[]{0}), n_samples);
    }

    // Convergence if || likehood - last_likelihood || < tolerance
    public boolean convergence(NDArray X) {
        if (responsibilities.size() < 2)
            return false;
        int sz = responsibilities.size();
        NDArray dif = NDArrays.sub(responsibilities.get(sz - 1), responsibilities.get(sz - 2));
        // torch.norm(self.responsibilities[-1].values - self.responsibilities[-2].values)
        double difference = dif.norm().toDoubleArray()[0];
        return difference <= tolerance;
    }

    public NDArray predict(NDArray X) {
        random_gaussian_initialization(X);

        for (int i = 0; i < max_epochs; i++) {
            System.out.println("---> epoch: " + (i+1));
            expectation(X);
            maximization(X);

            if (convergence(X))
                break;
        }
        expectation(X);
        return sample_assignments;
    }

    public static void main(String[] args) {

        String f = "data/ML/iris.data";
        ArrayList<NDArray> res = loadIrisData(f);
        NDArray X = res.get(0);
        NDArray y = res.get(1);
        int n_classes = y.unique().size();

        ArrayList<NDArray> spt = train_test_split(X, y, 0.2);
        NDArray X_train = spt.get(0), y_train = spt.get(1);
        NDArray X_test = spt.get(2), y_test = spt.get(3);
        //System.out.println("X_train: " + X_train.get("0:10,:"));

        double tolerance = 1e-8;
        GaussianMixtureModel gmm = new GaussianMixtureModel(n_classes, 100, tolerance);
        NDArray y_pred = gmm.predict(X_train);

        int correct = (int) y_pred.eq(y_train.flatten()).sum().toLongArray()[0];
        System.out.println("correct: " + correct + " num: " + y_train.getShape().getShape()[0]);
        System.out.println("Accuracy Score: " + (1.0 * correct / y_train.getShape().getShape()[0]));

        System.exit(0);
    }
}
