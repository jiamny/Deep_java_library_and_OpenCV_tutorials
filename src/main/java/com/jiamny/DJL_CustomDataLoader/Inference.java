/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package com.jiamny.DJL_CustomDataLoader;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import static com.jiamny.Utils.HelperFunctions.argSort;
import static com.jiamny.Utils.ImageHelper.addTextToBox;
import static com.jiamny.Utils.ImageHelper.ndarrayToMat;
import static com.jiamny.Utils.ImageViewer.show;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;
import static org.opencv.imgproc.Imgproc.cvtColor;

/** Uses the model to generate a prediction called an inference */
public class Inference {
    static {
        System.load("/usr/local/share/java/opencv4/libopencv_java480.so");
    }

    public static void main(String[] args) throws ModelException, TranslateException, IOException {
        // ----------------------------------------------------------------------
        // set specific version of torch & CUDA
        // ----------------------------------------------------------------------
        System.setProperty("PYTORCH_VERSION", "1.13.1");
        System.setProperty("PYTORCH_FLAVOR", "cu117");
        System.out.println(Engine.getDefaultEngineName());
        System.out.println(Engine.getInstance().defaultDevice());

        NDManager manager = NDManager.newBaseManager();
        // the location where the model is saved
        Path modelDir = Paths.get("models");

        // the path of image to classify
        String imageFilePath;
        if (args.length == 0) {
            imageFilePath = "data/images/rose.jpg";
        } else {
            imageFilePath = args[0];
        }

        // Load the image file from the path
        Image img = ImageFactory.getInstance().fromFile(Paths.get(imageFilePath));

        try (Model model = Models.getModel()) { // empty model instance
            // load the model
            model.load(modelDir, Models.MODEL_NAME);

            // define a translator for pre and post processing
            // out of the box this translator converts images to ResNet friendly ResNet 18 shape
            Translator<Image, Classifications> translator =
                    ImageClassificationTranslator.builder()
                            .addTransform(new Resize(Models.IMAGE_WIDTH, Models.IMAGE_HEIGHT))
                            .addTransform(new ToTensor())
                            .optApplySoftmax(true)
                            .build();

            // run the inference using a Predictor
            try (Predictor<Image, Classifications> predictor = model.newPredictor(translator)) {
                // holds the probability score per label
                Classifications predictResult = predictor.predict(img);
                System.out.println(predictResult);
                int [] idx = argSort(
                        predictResult.getProbabilities().stream().mapToDouble(x -> x).toArray());
                Classifications.Classification t = predictResult.item(idx[0]);
                String clsname = t.getClassName();
                System.out.println(clsname);

                Mat cvimg = ndarrayToMat(img.toNDArray(manager));

                int x = 0, y = 0;
                Scalar boxcolor = new Scalar(255, 0, 0);
                Scalar txtcolor = new Scalar(255, 255, 255);
                cvtColor(cvimg, cvimg, COLOR_BGR2RGB);
                addTextToBox(cvimg, clsname, txtcolor, Imgproc.FONT_HERSHEY_COMPLEX, boxcolor,
                        new Point(x+50, y + 15), 0.75, 1);

                // show predicted results
                HighGui.imshow("predicted", cvimg);
                HighGui.waitKey(0);
                HighGui.destroyAllWindows();

                System.exit(0);
            }
        }
    }
}
