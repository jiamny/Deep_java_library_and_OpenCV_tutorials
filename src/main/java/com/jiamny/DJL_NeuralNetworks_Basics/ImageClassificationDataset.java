package com.jiamny.DJL_NeuralNetworks_Basics;

//# The Image Classification Dataset

import ai.djl.Device;
import ai.djl.basicdataset.cv.classification.*;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.Record;
import com.jiamny.Utils.ImageUtils;
import org.apache.commons.lang3.time.StopWatch;

import java.awt.image.BufferedImage;
import java.awt.Graphics2D;
import java.awt.Color;
import java.util.concurrent.TimeUnit;

public class ImageClassificationDataset {

    // Saved in the FashionMnist class for later use
    public static String[] getFashionMnistLabels(int[] labelIndices) {
        String[] textLabels = {"t-shirt", "trouser", "pullover", "dress", "coat",
                "sandal", "shirt", "sneaker", "bag", "ankle boot"};
        String[] convertedLabels = new String[labelIndices.length];
        for (int i = 0; i < labelIndices.length; i++) {
            convertedLabels[i] = textLabels[labelIndices[i]];
        }
        return convertedLabels;
    }

    public static String getFashionMnistLabel(int labelIndice) {
        String[] textLabels = {"t-shirt", "trouser", "pullover", "dress", "coat",
                "sandal", "shirt", "sneaker", "bag", "ankle boot"};
        return textLabels[labelIndice];
    }

    // Saved in the FashionMnistUtils class for later use
    public static BufferedImage showImages(
            ArrayDataset dataset, int number, int width, int height, int scale, NDManager manager) {
        // Plot a list of images
        BufferedImage[] images = new BufferedImage[number];
        String[] labels = new String[number];
        for (int i = 0; i < number; i++) {
            Record record = dataset.get(manager, i);
            NDArray array = record.getData().get(0).squeeze(-1).toDevice(Device.cpu(), true);
            int y = (int) record.getLabels().get(0).toDevice(Device.cpu(), true).getFloat();
            images[i] = toImage(array, width, height);
            labels[i] = getFashionMnistLabel(y);
            System.out.println("labels[i]: " + labels[i]);
        }
        int w = images[0].getWidth() * scale;
        int h = images[0].getHeight() * scale;

        System.out.println("w: " + w + " h: " + h);

        return ImageUtils.showImages(images, labels, w, h);
    }

    private static BufferedImage toImage(NDArray array, int width, int height) {
        System.setProperty("apple.awt.UIElement", "true");
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g = (Graphics2D) img.getGraphics();
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                float c = array.getFloat(j, i) / 255; // scale down to between 0 and 1
                g.setColor(new Color(c, c, c)); // set as a gray color
                g.fillRect(i, j, 1, 1);
            }
        }
        g.dispose();
        return img;
    }

    public static void main(String[] args) {

        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        try {
            //## Getting the Dataset
            int batchSize = 256;
            boolean randomShuffle = true;

            FashionMnist mnistTrain = FashionMnist.builder()
                    .optUsage(Dataset.Usage.TRAIN)
                    .setSampling(batchSize, randomShuffle)
                    .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                    .build();

            FashionMnist mnistTest = FashionMnist.builder()
                    .optUsage(Dataset.Usage.TEST)
                    .setSampling(batchSize, randomShuffle)
                    .optLimit(Long.getLong("DATASET_LIMIT", Long.MAX_VALUE))
                    .build();

            mnistTrain.prepare();
            mnistTest.prepare();

            NDManager manager = NDManager.newBaseManager();
            System.out.println(mnistTrain.size());
            System.out.println(mnistTest.size());

            final int SCALE = 4;
            final int WIDTH = 28;
            final int HEIGHT = 28;

            showImages(mnistTrain, 6, WIDTH, HEIGHT, SCALE, manager);

            //## Reading a Minibatch
            StopWatch stopWatch = new StopWatch();
            stopWatch.start();
            for (Batch batch : mnistTrain.getData(manager)) {
                NDArray x = batch.getData().head();
                NDArray y = batch.getLabels().head();
            }
            stopWatch.stop();
            System.out.println(String.format("%.2f sec", stopWatch.getTime(TimeUnit.SECONDS) * 1.0f));
        } catch (Exception e) {
            e.printStackTrace();
        }

        System.exit(0);
    }
}

