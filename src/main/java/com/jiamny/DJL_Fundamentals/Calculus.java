package com.jiamny.DJL_Fundamentals;
//# Calculus
//## Derivatives and Differentiation

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import com.jiamny.Utils.PlotFigure;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;

import java.util.function.Function;

public class Calculus {

    public static Double numericalLim(Function<Double, Double> f, double x, double h) {
        return (f.apply(x + h) - f.apply(x)) / h;
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
        Function<Double, Double> f = x -> (3 * Math.pow(x, 2) - 4 * x);

        double h = 0.1;
        for (int i = 0; i < 5; i++) {
            System.out.println("h=" + String.format("%.5f", h) + ", numerical limit="
                    + String.format("%.5f", numericalLim(f, 1, h)));
            h *= 0.1;
        }

        NDArray X = manager.arange(0f, 3f, 0.1f, DataType.FLOAT64);
        double[] x = X.toDoubleArray();

        double[] fx = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            fx[i] = f.apply(x[i]);
        }

        double[] fg = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            fg[i] = 2 * x[i] - 3;
        }

        Figure fig = PlotFigure.plotLineAndSegment(x, fx, fg, "f(x)",
                "Tangent line(x=1)", "x", "f(x)", 700, 500);
        Plot.show(fig);

        System.exit(0);
    }
}


