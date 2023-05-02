package com.jiamny.DJL_Fundamentals;

//# Data Preprocessing
//## Adding tablesaw dependencies to Jupyter notebook

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import tech.tablesaw.api.BooleanColumn;
import tech.tablesaw.api.DoubleColumn;
import tech.tablesaw.api.StringColumn;
import tech.tablesaw.api.Table;
import tech.tablesaw.columns.Column;

import java.io.File;
import java.io.FileWriter;
import java.util.List;

public class TablesawExp {

    public static void main(String[] args) {
        //## Reading the Dataset
        File file = new File("./data/");
        file.mkdir();

        String dataFile = "./data/house_tiny.csv";

        try {
            // Create file
            File f = new File(dataFile);
            f.createNewFile();

            // Write to file
            FileWriter writer = new FileWriter(dataFile);
            writer.write("NumRooms,Alley,Price\n"); // Column names
            writer.write("NA,Pave,127500\n");       // Each row represents a data example
            writer.write("2,NA,106000\n");
            writer.write("4,NA,178100\n");
            writer.write("NA,NA,140000\n");
            writer.close();

            Table data = Table.read().file("./data/house_tiny.csv");
            System.out.println("data: \n" + data);

            //## Handling Missing Data
            Table inputs = data.create(data.columns());
            inputs.removeColumns("Price");
            Table outputs = data.selectColumns("Price");

            Column col = inputs.column("NumRooms");
            col.set(col.isMissing(), (int) inputs.nCol("NumRooms").mean());
            System.out.println("inputs: \n" + inputs);

            StringColumn scol = (StringColumn) inputs.column("Alley");
            List<BooleanColumn> dummies = scol.getDummies();
            inputs.removeColumns(scol);
            inputs.addColumns(DoubleColumn.create("Alley_Pave", dummies.get(0).asDoubleArray()),
                    DoubleColumn.create("Alley_nan", dummies.get(1).asDoubleArray())
            );

            System.out.println("inputs: \n" + inputs);

            //## Conversion to the NDArray Format
            NDManager nd = NDManager.newBaseManager();
            NDArray x = nd.create(inputs.as().doubleMatrix());
            NDArray y = nd.create(outputs.as().intMatrix());
            System.out.println("x: \n" + x);

            System.out.println("y: \n" + y);
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.exit(0);
    }
}

