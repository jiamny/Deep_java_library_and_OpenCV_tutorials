package com.jiamny.Utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class DataPoints {

    private NDArray X;
    private NDArray y;

    public DataPoints(NDArray X, NDArray y) {
        this.X = X;
        this.y = y;
    }

    public NDArray getX() {
        return X;
    }

    public NDArray getY() {
        return y;
    }

    // Generate y = X w + b + noise
    public static DataPoints syntheticData(NDManager manager, NDArray w, float b, int numExamples) {
        NDArray X = manager.randomNormal(new Shape(numExamples, w.size()));
        // Dimension mismatch or high dimensional dot operation is not supported. Please use .matMul instead.
        NDArray y;
        System.out.println(w.getShape().getShape()[0]);
        if( w.getShape().getShape().length > 1  || w.getShape().getShape()[0] > 1)
            y = X.matMul(w).add(b);
        else
            y = X.dot(w).add(b);
        // NDArray y = X.dot(w).add(b);
        //NDArray y = X.matMul(w).add(b);
        // Add noise
        y = y.add(manager.randomNormal(0, 0.01f, y.getShape(), DataType.FLOAT32));
        return new DataPoints(X, y);
    }
}
