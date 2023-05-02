package com.jiamny.Utils;

import ai.djl.ndarray.NDArray;
import ai.djl.training.dataset.ArrayDataset;

public class UtilFunctions {

    // Saved in the utils file for later use
    public static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle){
        return new ArrayDataset.Builder()
                .setData(features) // set the features
                .optLabels(labels) // set the labels
                .setSampling(batchSize,shuffle) // set the batch size and random sampling
                .build();
    }
}
