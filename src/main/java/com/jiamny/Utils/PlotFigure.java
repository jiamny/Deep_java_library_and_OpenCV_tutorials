package com.jiamny.Utils;

import tech.tablesaw.plotly.components.Axis;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.ScatterTrace;

public class PlotFigure {
    public static Figure plotLineAndSegment(double[] x, double[] y, double[] segment,
                                            String trace1Name, String trace2Name,
                                            String xLabel, String yLabel,
                                            int width, int height) {
        ScatterTrace trace = ScatterTrace.builder(x, y)
                .mode(ScatterTrace.Mode.LINE)
                .name(trace1Name)
                .build();

        ScatterTrace trace2 = ScatterTrace.builder(x, segment)
                .mode(ScatterTrace.Mode.LINE)
                .name(trace2Name)
                .build();

        Layout layout = Layout.builder()
                .height(height)
                .width(width)
                .showLegend(true)
                .xAxis(Axis.builder().title(xLabel).build())
                .yAxis(Axis.builder().title(yLabel).build())
                .build();

        return new Figure(layout, trace, trace2);
    }
}
