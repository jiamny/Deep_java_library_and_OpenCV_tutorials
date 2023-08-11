package com.jiamny.DJL_Fundamentals;

/* -----------------------------------
Data Manipulation
Getting Started
*/

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

public class ndarray {

    public static void main(String[] args) {
        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        NDManager manager = NDManager.newBaseManager();
        var x = manager.arange(12);
        System.out.println(x);

        x = manager.arange(12);
        System.out.println(x.getShape());
        System.out.println(x.size());

        x = x.reshape(3, 4);
        System.out.println(x);

        var a = manager.create(new Shape(3, 4));
        var b = manager.zeros(new Shape(2, 3, 4));
        var c = manager.ones(new Shape(2, 3, 4));

        var d = manager.randomNormal(0f, 1f, new Shape(3, 4), DataType.FLOAT32);
        var e = manager.randomNormal(new Shape(3, 4));
        var f = manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(3, 4));

        // Operations
        x = manager.create(new float[]{1f, 2f, 4f, 8f});
        var y = manager.create(new float[]{2f, 2f, 2f, 2f});

        System.out.println("x.add(y): \n" + x.add(y));
        System.out.println("x.sub(y): \n" + x.sub(y));
        System.out.println("x.mul(y): \n" + x.mul(y));
        System.out.println("x.div(y): \n" + x.div(y));
        System.out.println("x.pow(y): \n" + x.pow(y));
        System.out.println("x.exp(): \n" + x.exp());

        x = manager.arange(12f).reshape(3, 4);
        y = manager.create(new float[]{2, 1, 4, 3, 1, 2, 3, 4, 4, 3, 2, 1}, new Shape(3, 4));

        // default axis = 0
        System.out.println("x.concat(y): \n" + x.concat(y));
        System.out.println("x.concat(y, 1): \n" + x.concat(y, 1));
        System.out.println("x.eq(y): \n" + x.eq(y));
        System.out.println("x.sum(): \n" + x.sum());

        // Broadcasting Mechanism
        a = manager.arange(3f).reshape(3, 1);
        b = manager.arange(2f).reshape(1, 2);

        System.out.println(a);
        System.out.println(b);
        System.out.println(a.add(b).exp());

        // Indexing and Slicing
        System.out.println("x.get(':-1'): \n" + x.get(":-1"));
        System.out.println("x.get('1:3'): \n" + x.get("1:3"));

        x.set(new NDIndex("1, 2"), 9);
        System.out.println(x);

        x.set(new NDIndex("0:2, :"), 12);
        System.out.println(x);

        // Saving Memory
        var original = manager.zeros(y.getShape());
        var actual = original.addi(x);
        System.out.println((original == actual));

        System.exit(0);
    }
}

