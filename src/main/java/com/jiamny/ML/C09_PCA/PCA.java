package com.jiamny.ML.C09_PCA;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import tech.tablesaw.api.*;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.api.ScatterPlot;
import tech.tablesaw.plotly.components.*;
import tech.tablesaw.plotly.traces.ScatterTrace;

import java.util.ArrayList;
import java.util.Arrays;

import static com.jiamny.Utils.UtilFunctions.loadIrisData;
import static org.opencv.core.Core.eigen;

public class PCA {

    static {
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");
    }
    private int n_components;

    // param n_components: Number of principal components the data should be reduced too.
    public PCA(int n_components) {
        this.n_components = n_components;
    }

    /*
        * Centering our inputs with mean
        * Finding covariance matrix using centered tensor
        * Finding eigen value and eigen vector using torch.eig()
        * Sorting eigen values in descending order and finding index of high eigen values
        * Using sorted index, get the eigen vectors
        * Tranforming the Input vectors with n columns into PCA components with reduced dimension
        :param X: Input tensor with n columns.
        :return: Output tensor with reduced principal components
     */
    public NDArray fit_transform(NDArray X) {
        NDArray centering_X = X.sub(X.mean(new int[]{0})); // - torch.mean(X, dim=0)
        //covariance_matrix = torch.mm(centering_X.T, centering_X)/(centering_X.shape[0] - 1)
        int st = (int)centering_X.getShape().getShape()[0];
        NDArray covariance_matrix = (centering_X.transpose().matMul(centering_X)).div(st - 1);
        //eigen_values, eigen_vectors = torch.linalg.eig(covariance_matrix)

        // convert to opencv Mat
        int h = (int) (covariance_matrix.getShape().getShape()[0]);
        int w = (int) (covariance_matrix.getShape().getShape()[1]);
        Mat mat = new Mat(h, w, CvType.CV_64FC1);

        double[] data = covariance_matrix.toDoubleArray();
        mat.put(0, 0, data);

        Mat eigen_valuesMat = new Mat(), eigen_vectorsMat = new Mat();
        eigen(mat, eigen_valuesMat, eigen_vectorsMat);

        double [] eValues = new double[eigen_valuesMat.rows()];
        for (int r = 0; r < eigen_valuesMat.rows(); r++) {
            eValues[r] = eigen_valuesMat.get(r, 0)[0];
        }

        double [][] eVectors = new double[eigen_vectorsMat.cols()][eigen_vectorsMat.rows()];
        for (int c = 0; c < eigen_vectorsMat.cols(); c++) {
            for(int r = 0; r < eigen_vectorsMat.rows(); r++) {
                eVectors[c][r] = eigen_vectorsMat.get(r, c)[0];
            }
        }

        NDManager manager = NDManager.newBaseManager();

        NDArray eigen_values = manager.create(eValues).toType(DataType.FLOAT64, false);
        NDArray eigen_vectors = manager.create(eVectors).toType(DataType.FLOAT64, false);
        //eigen_vectors = eigen_vectors.transpose();

        //System.out.println(eigen_values);
        //System.out.println(eigen_vectors);
        NDArray eigen_sorted_index = eigen_values.argSort(0, false);
        //System.out.println(eigen_sorted_index);
        NDArray eigen_vectors_sorted = eigen_vectors.get(eigen_sorted_index); //[:,eigen_sorted_index]
        //System.out.println(eigen_vectors_sorted);
        NDArray component_vector = eigen_vectors_sorted.get("...,0:" + n_components); //[:,0:self.components];
        //System.out.println(component_vector);
        // torch.mm(component_vector.T, centering_X.T).T
        NDArray transformed = component_vector.transpose().matMul(centering_X.transpose()).transpose();
        return transformed;
    }


    public static void main(String[] args) {

        String f = "data/ML/iris.data";
        ArrayList<NDArray> res = loadIrisData(f);
        NDArray X = res.get(0);
        NDArray y = res.get(1);

        PCA pca = new PCA(2);
        NDArray pca_vector = pca.fit_transform(X);
        System.out.println(pca_vector.get("0:10,..."));
        System.out.println(y.get("0:10,0"));

        var xx = pca_vector.get(":, 0");
        var yy = pca_vector.get(":, 1");
        System.out.println(xx.get("0:10"));

        NumericColumn<?> xw = DoubleColumn.create("pca_1", xx.toDoubleArray());
        NumericColumn<?> yl = DoubleColumn.create("pca_2", yy.toDoubleArray());
        StringColumn tp = StringColumn.create("tp",
                Arrays.stream(y.get("...,0").toIntArray())
                .mapToObj(String::valueOf)
                .toArray(String[]::new));

        Table data = Table.create(xw, yl, tp);

        Plot.show(ScatterPlot.create("PCA - iris data",
                data, "pca_1", "pca_2", "tp"));

        System.exit(0);
    }
}
