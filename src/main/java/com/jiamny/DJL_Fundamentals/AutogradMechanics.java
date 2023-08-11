package com.jiamny.DJL_Fundamentals;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;

public class AutogradMechanics {

    public static void main(String[] args) {
        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        NDManager manager = NDManager.newBaseManager();

        /***************************************************************************
         * 一，利用backward方法求导数
         * backward 方法通常在一个标量张量上调用，该方法求得的梯度将存在对应自变量张量的grad属性下。
         * 如果调用的张量非标量，则要传入一个和它同形状 的gradient参数张量。
         * 相当于用该gradient参数张量与调用张量作向量点乘，得到的标量结果再反向传播。
         */
        // 1, 标量的反向传播
        // # f(x) = a*x**2 + b*x + c的导数

        var x = manager.create(0.0);
        x.setRequiresGradient(true);        // x 需要被求导
        var a = manager.create(1.0);
        var b = manager.create(-2.0);
        var c = manager.create(1.0);
        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            var y = a.mul(x.pow(2)).add(b.mul(x)).add(c);
            gc.backward(y);
            var dy_dx = x.getGradient();
            System.out.println("dy_dx: " + dy_dx);
        }


        // 2, 非标量的反向传播
        //# f(x) = a*x**2 + b*x + c
        x = manager.create(new double[][]{{0.0, 0.0}, {1.0, 2.0}});
        x.setRequiresGradient(true);       // x 需要被求导
        a = manager.create(1.0);
        b = manager.create(-2.0);
        c = manager.create(1.0);

        System.out.println("x: \n" + x.getShape());
        System.out.println("a: \n" + a.getShape());
        System.out.println("EngineName: \n" + Engine.getDefaultEngineName());


        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            var y =  NDArrays.mul(x.pow(2), a).add(x.mul(b)).add(c);
            var gd = manager.create(new double[][]{{1.0, 1.0}, {1.0, 1.0}});

            gc.backward(y);
            var x_grad = x.getGradient();
            System.out.println("x_grad: " + x_grad);
        }

        // 3, 非标量的反向传播可以用标量的反向传播实现
        //# f(x) = a*x**2 + b*x + c

        x = manager.create(new double[][]{{0.0, 0.0}, {1.0, 2.0}});
        x.setRequiresGradient(true);       // x 需要被求导
        a = manager.create(1.0).broadcast(x.getShape());
        b = manager.create(-2.0).broadcast(x.getShape());
        c = manager.create(1.0).broadcast(x.getShape());

        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            var y = a.dot(x.pow(2)).add(b.dot(x)).add(c);

            var gd = manager.create(new double[][]{{1.0, 1.0}, {1.0, 1.0}});
            var z = gd.add(y);

            System.out.println("x: \n" + x);
            System.out.println("y: \n" + y);
            gc.backward(z);

            var x_grad = x.getGradient();
            System.out.println("x_grad: " + x_grad);
        }

        /**************************************************
         * 二，利用自动微分和优化器求最小值
         */
        // f(x) = a*x**2 + b*x + c的最小值

        x = manager.create(new double[][]{{0.0, 0.0}, {1.0, 2.0}});
        x.setRequiresGradient(true);       // x 需要被求导
        a = manager.create(1.0);
        b = manager.create(-2.0);

        Tracker lrt = Tracker.fixed(0.01f);
        Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

        var weight = manager.create(new double[][]{{0.0, 0.0}, {0.0, 0.0}});
        var gd = manager.create(new double[][]{{1.0, 1.0}, {1.0, 1.0}});

        try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
            NDArray y = manager.create(0);
            for (int i = 0; i < 500; i++) {
                y = a.mul(x.pow(2)).add(b.mul(x)).add(c);
                gc.backward(y);
                sgd.update("", weight, gd);
            }
            System.out.println("x: \n" + x);
            System.out.println("y: \n" + y);

            var x_grad = x.getGradient();
            System.out.println("x_grad: " + x_grad);
        }

        System.exit(0);
    }
}
