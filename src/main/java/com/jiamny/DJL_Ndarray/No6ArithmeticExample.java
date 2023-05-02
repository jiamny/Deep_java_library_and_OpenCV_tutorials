package com.jiamny.DJL_Ndarray;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;

/**
 * Ndarray 算术函数
 * http://aias.top/
 *
 * @author Calvin
 */

public final class No6ArithmeticExample {

    private No6ArithmeticExample() {
    }

    public static void main(String[] args) {
        try (NDManager manager = NDManager.newBaseManager()) {
            // 1. 加减乘除
            NDArray a = manager.arange(0, 9, 1, DataType.FLOAT32).reshape(3, 3);
            System.out.println("第一个数组：");
            System.out.println(a);
            System.out.println("第二个数组：");
            NDArray b = manager.create(new int[]{10, 10, 10});
            System.out.println(b);
            System.out.println("两个数组相加：");
            NDArray c = a.add(b);
            System.out.println(c);
            System.out.println("两个数组相减：");
            c = a.sub(b);
            System.out.println(c);
            System.out.println("两个数组相乘：");
            c = a.mul(b);
            System.out.println(c);
            System.out.println("两个数组相除：");
            c = a.div(b);
            System.out.println(c);

            // 2. 幂函数
            a = manager.create(new int[]{10, 20, 30});
            System.out.println("第一个数组：");
            System.out.println(a);
            System.out.println("第二个数组：");
            b = manager.create(new int[]{3, 5, 7});
            System.out.println(b);
            System.out.println("调用 mod() 函数：");
            c = a.mod(b);
            System.out.println(c);
        }
        System.exit(0);
    }
}
