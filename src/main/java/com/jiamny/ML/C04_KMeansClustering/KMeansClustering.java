package com.jiamny.ML.C04_KMeansClustering;


import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.Arrays;

import static com.jiamny.Utils.UtilFunctions.loadIrisData;
import static com.jiamny.Utils.UtilFunctions.train_test_split;

class KMeans {
    public int k, samples, features, max_iterations;
    public NDArray KMeans_Centroids;
    public String method = "";
    /**
     @param X: input tensor
     @param k: Number of clusters
     :variable samples: Number of samples
     :variable features: Number of features
     */
    public KMeans(NDArray X, int k, int iterations, String method) {
        this.k = k;
        this.max_iterations = iterations;
        this.samples = (int)X.getShape().getShape()[0];
        this.features = (int)X.getShape().getShape()[1];
        this.KMeans_Centroids = null;
        this.method = method;
    }

    /*
    Initialization Technique is KMeans++. Thanks to stackoverflow.
        https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
        :param X: Input Tensor
        :param K: Number of clusters to build
        :return: Selection of three centroid vector from X
     */
    public NDArray initialize_centroid(NDArray X, int K) {
        NDManager manager = NDManager.newBaseManager();
        ArrayList<Integer> I = new ArrayList<>();
        ArrayList<NDArray> C = new ArrayList<>();

        I.add(0);
        C.add(X.get(0));

        for( int k = 1; k < K; k++ ) {
            NDArray D2 = manager.zeros(new Shape(samples), DataType.FLOAT64);

            for (int i = 0; i < samples; i++) {
                NDArray x = X.get(i);
                NDArray cd = manager.zeros(new Shape(C.size()));
                //D2 = np.array([min([np.inner(c - x, c - x) for c in C]) for x in X])
                for(int j = 0; j < C.size(); j++) {
                    NDArray c = C.get(j);
                    NDArray t = c.sub(x);
                    cd.set(new NDIndex(j), t.dot(t));
                }
                D2.set(new NDIndex(i), cd.min());
            }
            NDArray probs = D2.div(D2.sum());
            NDArray cumprobs = probs.cumSum();
            double r = manager.randomUniform(0, 1,
                    new Shape(1)).toType(DataType.FLOAT64, false).toDoubleArray()[0];
            //r = torch.rand(1).item()
            int l = (int)cumprobs.getShape().getShape()[0];
            int idx = 0;
            for(int n = 0; n < l; n++) {
                if( r < cumprobs.getDouble(n) ) {
                    idx = n;
                    break;
                }
            }
            I.add(idx);
        }
        int [] sidx = I.stream().mapToInt(i -> i).toArray();

        return X.get(manager.create(sidx));
    }

    public NDArray distance(NDArray sample, NDArray centroid, String method ) {
        if( method.equalsIgnoreCase( "euclidean") ){
            return (sample.sub(centroid)).norm(); //torch.norm(sample - centroid, 2, 0)
        } else if(method.equalsIgnoreCase("manhattan" )){
            //return torch.sum(torch.abs(sample - centroid))
            return (sample.sub(centroid).abs()).sum();
        } else if( method.equalsIgnoreCase("cosine") ) {
            //return torch.sum(sample * centroid) / (torch.norm(sample) * torch.norm(centroid))
            return ((sample.mul(centroid)).div(sample.norm().mul(centroid.norm()))).sum();
        } else {
                System.out.println("Unknown similarity distance type");
                return null;
        }
    }

    /*
    :param sample: sample whose distance from centroid is to be measured
    :param centroids: all the centroids of all the clusters
    :return: centroid's index is passed for each sample
     */
    public int closest_centroid(NDArray sample, NDArray centroids) {
        int closest = -1;
        double min_distance = Double.MAX_VALUE;
        for( int idx = 0; idx < (int)centroids.getShape().getShape()[0]; idx++ ) {
            NDArray centroid = centroids.get(idx);
            double distance = distance(sample, centroid, method).toDoubleArray()[0];
            if (distance < min_distance) {
                closest = idx;
                min_distance = distance;
            }
        }
        return closest;
    }

    /*
     :param centroids: Centroids of all clusters
     :param X: Input tensor
     :return: Assigning each sample to a cluster.
     */
    public ArrayList<ArrayList<Integer>> create_clusters(NDArray centroids, NDArray X) {
        int n_samples = (int)X.getShape().getShape()[0];
        ArrayList<ArrayList<Integer>> k_clusters = new ArrayList<>(); //[[] for _ in range(self.k)]
        for(int j = 0; j < k; j++)
            k_clusters.add(new ArrayList<Integer>());

        for(int idx = 0; idx < n_samples; idx++) {
            NDArray sample = X.get(idx);
            int centroid_index = closest_centroid(sample, centroids);
            k_clusters.get(centroid_index).add(idx);
        }

        return k_clusters;
    }

    /*
    :return: Updating centroids after each iteration.
     */
    public NDArray update_centroids(ArrayList<ArrayList<Integer>> clusters, NDArray X) {
        NDManager manager = NDManager.newBaseManager();
        NDArray centroids = manager.zeros( new Shape(k, features), DataType.FLOAT64 );
        for( int idx = 0; idx < k; idx++) {
            NDArray cIdx = manager.create(clusters.get(idx).stream().mapToInt(m -> m).toArray()).toType(DataType.INT32, false);
            NDArray centroid = X.get(cIdx);

            int [] didx = new int[]{0};
            NDArray mn = centroid.mean(didx);
            centroids.set(new NDIndex(idx + ",..."), mn);
        }

        return centroids;
    }

    /*
    Labeling the samples with index of clusters
    :return: labeled samples
     */
    public NDArray label_clusters( ArrayList<ArrayList<Integer>> clusters, NDArray X) {
        NDManager manager = NDManager.newBaseManager();
        NDArray y_pred = manager.zeros(new Shape(X.getShape().getShape()[0]), DataType.INT32);

        for(int idx = 0; idx < k; idx++) {
            int [] cluster = clusters.get(idx).stream().mapToInt(m -> m).toArray();
            for( int sample_idx : cluster )
                y_pred.set(new NDIndex(sample_idx), idx);
        }

        return y_pred;
    }

    /*
        Initializing centroid using Kmeans++, then find distance between each sample and initial centroids, then assign
        cluster label based on min_distance, repeat this process for max_iteration and simultaneously updating
        centroid by calculating distance between sample and updated centroid. Convergence happen when difference between
        previous and updated centroid is None.
     */
    public void fit(NDArray X) {
        NDArray centroids = initialize_centroid(X, k);
        for( int i = 0; i < max_iterations; i++ ) {
            ArrayList<ArrayList<Integer>> clusters = create_clusters(centroids, X);
            NDArray previous_centroids = centroids;
            centroids = update_centroids(clusters, X);
            NDArray difference = centroids.sub(previous_centroids);

            if( difference.sum().toDoubleArray()[0] != 0 )
                continue;
            else
                break;
        }
        KMeans_Centroids = centroids;
    }

    /*
    :return: label/cluster number for each input sample is returned
     */
    public NDArray predict(NDArray X) {
        if(KMeans_Centroids == null) {
            System.out.println("No Centroids Found. Run KMeans fit");
            return null;
        }

        ArrayList<ArrayList<Integer>> clusters = create_clusters(KMeans_Centroids, X);
        NDArray labels = label_clusters(clusters, X);

        return labels;
    }
}
public class KMeansClustering {

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();

        String f = "data/ML/iris.data";
        ArrayList<NDArray> res = loadIrisData(f);

        /******************************************************
         * normalize data
         ******************************************************/
        NDArray X = res.get(0);
        int [] idx = new int[]{0};
        NDArray mn = X.mean(idx);
        NDArray var = ((X.sub(mn).square()).sum(idx)).div(X.getShape().getShape()[0] - 1);

        NDArray XX = X.sub(mn).div(var.sqrt());
        System.out.println("XX " + XX.get("0:10,..."));
        System.out.println("XX.sum " + XX.sum(idx));

        int n_classes = res.get(1).unique().size();
        res = train_test_split(XX, res.get(1), 0.2);

        NDArray X_train = res.get(0), y_train = res.get(1);
        NDArray X_test = res.get(2), y_test = res.get(3);

        int iterations = 300;
        KMeans km = new KMeans(X_train, n_classes, iterations, "cosine");

        km.fit(X_train);
        NDArray ypred = km.predict(X_test);
        int correct = (int)ypred.eq(y_test.flatten()).sum().toLongArray()[0];
        System.out.println("correct: " + correct + " num: " + y_test.getShape().getShape()[0]);
        System.out.println("Accuracy Score: " + (1.0*correct/y_test.getShape().getShape()[0]));

        System.exit(0);
    }
}
