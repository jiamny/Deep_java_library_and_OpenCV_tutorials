package com.jiamny.DJL_Fundamentals;

//# Automatic Differentiation

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;

public class Autograd {

    //## Computing the Gradient of Java Control Flow
    public static NDArray f(NDArray a) {
        NDArray b = a.mul(2);
        while (b.norm().getFloat() < 1000) {
            b = b.mul(2);
        }
        NDArray c;
        if (b.sum().getFloat() > 0) {
            c = b;
        } else {
            c = b.mul(100);
        }
        return c;
    }


    public static void main(String[] args) {
        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");

        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());


        NDManager manager = NDManager.newBaseManager();

        //## A Simple Example
        NDArray x = manager.arange(4f);
        System.out.println(x);

        // We allocate memory for a NDArrays's gradient by invoking `setRequiresGradient(true)`
        x.setRequiresGradient(true);

        // After we calculate a gradient taken with respect to `x`, we will be able to
        // access it via the `getGradient` attribute, whose values are initialized with 0s
        System.out.println("x.getGradient(): \n" +
                x.getGradient());

        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            NDArray y = x.dot(x).mul(2);
            System.out.println(y);
            gc.backward(y);
        }
        System.out.println("x.getGradient(): \n" +
                x.getGradient());

        System.out.println("x.getGradient().eq(x.mul(4)): " + x.getGradient().eq(x.mul(4)));

        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            NDArray y = x.sum();
            gc.backward(y);
        }
        // Overwritten by the newly calculated gradient
        System.out.println("x.getGradient(): \n" +
                x.getGradient());

        //## Backward for Non-Scalar Variables
        // When we invoke `backward` on a vector-valued variable `y` (function of `x`),
        // a new scalar variable is created by summing the elements in `y`. Then the
        // gradient of that scalar variable with respect to `x` is computed
        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            NDArray y = x.mul(x); // y is a vector
            gc.backward(y);
        }
        // Overwritten by the newly calculated gradientdient
        System.out.println("x.getGradient(): \n" +
                x.getGradient());

        //## Detaching Computation
        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            NDArray y = x.mul(x);
            NDArray u = y.stopGradient();
            NDArray z = u.mul(x);
            gc.backward(z);
            System.out.println(x.getGradient().eq(u));
        }

        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            NDArray y = x.mul(x);
            y = x.mul(x);
            gc.backward(y);
            System.out.println(x.getGradient().eq(x.mul(2)));
        }

        NDArray a = manager.randomNormal(new Shape(1));
        a.setRequiresGradient(true);
        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            NDArray d = f(a);
            gc.backward(d);

            System.out.println(a.getGradient().eq(d.div(a)));
        }

        System.exit(0);
    }
}


