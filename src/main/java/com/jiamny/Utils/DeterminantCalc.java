package com.jiamny.Utils;

public class DeterminantCalc {
    public static double[][] swap(double[][] arr, int i1, int j1, int i2,
                          int j2) {
        double temp = arr[i1][j1];
        arr[i1][j1] = arr[i2][j2];
        arr[i2][j2] = temp;
        return arr;
    }
    public static double Determinant(double[][] matrix) {
        double det=1;
        double num1, num2, total = 1;
        int index, n = matrix.length;

        double[] temp = new double[n + 1];
        // обхід діагональних елементів
        for (int i = 0; i < n; i++) {
            index = i;
            while (index < n && matrix[index][i] == 0) {
                index++;
            }
            if (index == n)
            {
                // детермінант матриці 0
                continue;
            }
            if (index != i) {
                // обмін діагонального та index-ного елементу
                for (int j = 0; j < n; j++) {
                    swap(matrix, index, j, i, j);
                }

                det = (float) (det * Math.pow(-1, index - i));
            }

            // збереження діагональних елементів
            for (int j = 0; j < n; j++) {
                temp[j] = matrix[i][j];
            }

            for (int j = i + 1; j < n; j++) {
                num1 = temp[i];
                num2 = matrix[j][i];


                for (int k = 0; k < n; k++) {
                    matrix[j][k] = (num1 * matrix[j][k])
                            - (num2 * temp[k]);
                }
                total = total * num1; // Det(kA)=kDet(A);
            }
        }

        for (int i = 0; i < n; i++) {
            det = det * matrix[i][i];
        }
        double result =  (det / total); // Det(kA)/k=Det(A);
        if(result == -0.0) result = 0;
        return result;
    }
}
