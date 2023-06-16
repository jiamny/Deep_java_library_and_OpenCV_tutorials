package com.jiamny.ML.C06_K_NearestNeighbor;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

import java.util.*;

import static com.jiamny.Utils.UtilFunctions.loadIrisData;
import static com.jiamny.Utils.UtilFunctions.train_test_split;

public class K_NearestNeighbor {
    public int K;

    /*
    :param k: Number of Neighbors
     */
    public K_NearestNeighbor(int K) {
        this.K = K;
    }

    public NDArray distance(NDArray point1, NDArray point2, String method ) {
        if( method.equalsIgnoreCase( "euclidean") ){
            return (point1.sub(point2)).norm();
        } else if(method.equalsIgnoreCase("manhattan" )){
            return (point1.sub(point2).abs()).sum();
        } else if( method.equalsIgnoreCase("cosine") ) {
            return ((point1.mul(point2)).div(point1.norm().mul(point2.norm()))).sum();
        } else {
            System.out.println("Unknown similarity distance type");
            return null;
        }
    }

    /*
        * Iterate through each datapoints (item/y_test) that needs to be classified
        * Find distance between all train data points and each datapoint (item/y_test)
          using euclidean distance
        * Sort the distance using argsort, it gives indices of the y_test
        * Find the majority label whose distance closest to each datapoint of y_test.

        :param X: Input tensor
        :param y: Ground truth label
        :param item: tensors to be classified
        :return: predicted labels
     */
    public NDArray fit_predict(NDArray X, NDArray y, NDArray item) {
        NDManager manager = NDManager.newBaseManager();

        int [] sIdx = new int[K];
        for(int j = 0; j < K; j++)
            sIdx[j] = j;

        int [] y_predict = new int[(int)item.getShape().get(0)];
        for(int i = 0; i < (int)item.getShape().get(0); i++ ) {
            ArrayList<Double> point_dist = new ArrayList<>();
            for(int ipt = 0; ipt < (int)X.getShape().get(0); ipt++ ) {
                NDArray distances = distance(X.get(ipt), item.get(i), "euclidean");
                point_dist.add(distances.toDoubleArray()[0]);
            }

            NDArray point_distances = manager.create(point_dist.stream().mapToDouble(m ->m).toArray());

            NDArray k_neighbors = (point_distances.argSort()).get(manager.create(sIdx));
            NDArray y_label = y.get(k_neighbors);
            int major_class =  findMajorClass(y_label.flatten());
            //y_predict.append(major_class)
            y_predict[i] = major_class;
        }

        return manager.create(y_predict);
    }

    public int findMajorClass(NDArray X) {
        HashMap<Integer, Integer> class_count = new HashMap<>();
        int [] labels = X.toIntArray();

        for (int label : labels) {
            if (class_count.containsKey(label)) {
                int c = class_count.get(label);
                class_count.put(label, (c + 1));
            } else {
                class_count.put(label, 1);
            }
        }

        // Create a list from elements of HashMap
        List<Map.Entry<Integer, Integer> > list =
                new LinkedList<Map.Entry<Integer, Integer> >(class_count.entrySet());

        // Sort the list
        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer> >() {
            public int compare(Map.Entry<Integer, Integer> o1,
                               Map.Entry<Integer, Integer> o2)
            {
                return (o2.getValue()).compareTo(o1.getValue());
            }
        });

        //System.out.println("cls = " + cls);
        return list.get(0).getKey();
    }

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();

        String f = "data/ML/iris.data";
        ArrayList<NDArray> res = loadIrisData(f);

        res = train_test_split(res.get(0), res.get(1), 0.2);
        NDArray X_train = res.get(0), y_train = res.get(1);
        NDArray X_test = res.get(2), y_test = res.get(3);

        K_NearestNeighbor knn = new K_NearestNeighbor(5);
        NDArray y_pred = knn.fit_predict(X_train, y_train, X_test);
        System.out.print("y_pred: " );
        System.out.print("[ ");
        Arrays.stream(y_pred.toIntArray()).forEach(num -> System.out.print(num + " "));
        System.out.println("]");
        System.out.print("y_test: " );
        System.out.print("[ ");
        Arrays.stream(y_test.toIntArray()).forEach(num -> System.out.print(num + " "));
        System.out.println("]");

        int correct = (int)y_pred.eq(y_test.flatten()).sum().toLongArray()[0];
        System.out.println("correct: " + correct + " num: " + y_test.getShape().getShape()[0]);
        System.out.println("Accuracy Score: " + (1.0*correct/y_test.getShape().getShape()[0]));

        System.exit(0);
    }
}