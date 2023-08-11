package com.jiamny.ML.C01_LinearRegression;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import static com.jiamny.Utils.HelperFunctions.printVectorElements;

public class LinearRegression {
    private double lr;
    private long iterations, m, n;

    public LinearRegression() {
        lr = 0.01;
        iterations = 1000;
    }

    /**
     * @desc w: weight tensor
     * @desc X: input tensor
     */
    public NDArray y_pred(NDArray X, NDArray w) {
        return w.transpose(1, 0).matMul(X); // torch.mm(torch.transpose(w, 0, 1), X)
    }

    /**
     * @desc c: cost function - to measure the loss between estimated vs ground truth
     */
    public double loss(NDArray ypred, NDArray y) {
        try (NDManager manager = NDManager.newBaseManager()) {
            // l = 1 / self.m * torch.sum(torch.pow(ypred - y, 2))
            NDArray l = ((ypred.sub(y)).pow(2)).sum().mul(1.0/m);
            return l.toDoubleArray()[0];
        }
    }

    /**
     *  @desc dCdW: derivative of cost function
     *  @desc w_update: change in weight tensor after each iteration
     */
    public NDArray gradient_descent(NDArray w, NDArray X, NDArray y, NDArray ypred) {
        try (NDManager manager = NDManager.newBaseManager()) {
            //dCdW = 2 / self.m * torch.mm(X, torch.transpose(ypred - y, 0, 1))
            NDArray dCdW = X.matMul((ypred.sub(y)).transpose(1, 0)).mul(2.0/m);
            NDArray w_update = w.sub(dCdW.mul(lr));
            return w_update;
        }
    }

    /**
     * @type y: tensor object
     * @type X: tensor object
     */
    public NDArray run(NDArray X, NDArray y) {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray bias = manager.ones(new Shape(1, X.getShape().get(1)), DataType.FLOAT64);
            X = bias.concat(X, 0);
            m = X.getShape().get(1);
            n = X.getShape().get(0);

            NDArray w = manager.zeros(new Shape(n, 1), DataType.FLOAT64);

            for (long iteration = 1; iteration < (iterations + 1); iteration++) {
                NDArray ypred = y_pred(X, w);

                double cost = loss(ypred, y);

                if( iteration % 100 == 0 ) {
                    System.out.printf("Loss at iteration %d is %.4f\n", iteration, cost);
                }

                w = gradient_descent(w, X, y, ypred);
            }
            return w;
        }
    }

    public static void main(String[] args) {
        // set specific version of torch & CUDA
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");

        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray X = manager.randomUniform(0.0f, 1.0f, new Shape(1, 500), DataType.FLOAT64);
            NDArray y = (X.mul(2).add(3)).add( manager.randomNormal(0.0f, 1.0f, new Shape(1, 500), DataType.FLOAT64).mul(0.1));

            LinearRegression regression = new LinearRegression();
            NDArray w = regression.run(X, y);
        }
        System.exit(0);
    }
}
