package com.jiamny.DJL_Fundamentals;

//# Linear Algebra
//## Scalars

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.*;

public class LinearAlgebra {

    //## Norms
    public static NDArray l2Norm(NDArray w) {
        return ((w.pow(2)).sum()).sqrt();
    }

    public static void main(String[] args) {

        NDManager manager = NDManager.newBaseManager();
        NDArray x = manager.create(3f);
        NDArray y = manager.create(2f);

        System.out.println("x.add(y): \n" + x.add(y));

        System.out.println("x.mul(y): \n" + x.mul(y));

        System.out.println("x.div(y): \n" + x.div(y));

        System.out.println(" x.pow(y): \n" + x.pow(y));

        //## Vectors
        x = manager.arange(4f);
        System.out.println(x);

        System.out.println("x.get(3): \n" + x.get(3));

        //### Length, Dimensionality, and Shape
        System.out.println("x.size(0): " + x.size(0));

        System.out.println("x.getShape(): " + x.getShape());

        //## Matrices
        NDArray A = manager.arange(20f).reshape(5, 4);
        System.out.println(A);

        System.out.println("A.transpose(): \n" +
                A.transpose());

        NDArray B = manager.create(new float[][]{{1, 2, 3}, {2, 0, 4}, {3, 4, 5}});
        System.out.println(B);

        System.out.println("B.eq(B.transpose()): \n" +
                B.eq(B.transpose()));

        //## NDArrays
        NDArray X = manager.arange(24f).reshape(2, 3, 4);
        System.out.println(X);

        //## Basic Properties of NDArray Arithmetic
        System.out.println("---------------Basic Properties of NDArray Arithmetic \n");
        A = manager.arange(20f).reshape(5, 4);
        B = A.duplicate(); // Assign a copy of `A` to `B` by allocating new memory
        System.out.println(A);

        System.out.println("A.add(B): \n" + A.add(B));

        System.out.println("A.mul(B): \n" + A.mul(B));

        int a = 2;
        X = manager.arange(24f).reshape(2, 3, 4);

        System.out.println("X.add(a): \n" + X.add(a));

        System.out.println("(X.mul(a)).getShape(): " + (X.mul(a)).getShape());

        //## Reduction
        System.out.println("---------------Reduction \n");
        x = manager.arange(4f);
        System.out.println("x: \n" + x);

        System.out.println("x.sum(): \n" + x.sum());

        System.out.println("A.getShape(): " + A.getShape());

        System.out.println("A.sum(): \n" + A.sum());

        NDArray ASumAxis0 = A.sum(new int[]{0});
        System.out.println(ASumAxis0);

        System.out.println("ASumAxis0.getShape(): " + ASumAxis0.getShape());

        NDArray ASumAxis1 = A.sum(new int[]{1});
        System.out.println(ASumAxis1);

        System.out.println("ASumAxis1.getShape(): " + ASumAxis1.getShape());

        // Same as `A.sum()`
        System.out.println("A.sum(new int[] {0,1}): \n" +
                A.sum(new int[]{0, 1}));

        System.out.println("A.mean(): \n" +
                A.mean());

        System.out.println("A.sum().div(A.size()): \n" +
                A.sum().div(A.size()));

        System.out.println("A.mean(new int[] {0}): \n" +
                A.mean(new int[]{0}));

        System.out.println("A.sum(new int[] {0}).div(A.getShape().get(0)): \n" +
                A.sum(new int[]{0}).div(A.getShape().get(0)));

        //### Non-Reduction Sum
        System.out.println("---------------Non-Reduction Sum \n");
        NDArray sumA = A.sum(new int[]{1}, true);
        System.out.println(sumA);

        System.out.println("A.div(sumA): \n" + A.div(sumA));

        System.out.println("A.cumSum(0): \n" + A.cumSum(0));

        //## Dot Products
        System.out.println("---------------Dot Products \n");
        y = manager.ones(new Shape(4));
        System.out.println(x);
        System.out.println(y);

        System.out.println("x.dot(y): \n" + x.dot(y));

        System.out.println("x.mul(y).sum(): \n" + x.mul(y).sum());

        //## Matrix-Vector Products
        System.out.println("---------------Matrix-Vector Products \n");
        System.out.println("x: \n" + x);
        System.out.println("A: \n" + A);

        System.out.println("A.getShape(): " + A.getShape());

        System.out.println("x.getShape(): " + x.getShape());

        System.out.println("A.dot(x): \n" + A.dot(x.reshape(4,1)));

        //## Matrix-Matrix Multiplication
        B = manager.ones(new Shape(4, 3));
        System.out.println("A.dot(B): \n" + A.dot(B));

        NDArray u = manager.create(new float[]{3, -4});
        System.out.println("l2Norm(u): \n" + l2Norm(u));

        System.out.println("u.abs().sum(): \n" + u.abs().sum());

        System.out.println("l2Norm(manager.ones(new Shape(4,9))): \n" +
                l2Norm(manager.ones(new Shape(4, 9))));

        System.exit(0);
    }
}
