# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import io
from keras.models import load_model
import argparse
from voxel_classifier.features_extract import FeaturesExtract
from voxel_classifier.prediction import Prediction

cwd = os.getcwd()


def runPred(img_path, model_paths):
    output_path = os.path.join(os.path.dirname(img_path), "predictions")
    os.makedirs(output_path, exist_ok=True)
    for target, model_path in model_paths.items():
        if target == 'Mitochondria':
            featureClass = FeaturesExtract(img_path, model_path)
            features, img_info_dict = featureClass.run_extract()
            pred = Prediction(SMAType='ML', riPath=img_path, modelPath=model_path,
                              img_info_dict=img_info_dict, features=features)
            mask = pred.run_prediction()
            pred = mask[0]
        else:
            model = load_model(model_path, compile=False)
            img = io.imread(img_path)
            img = (img - 1.3375) / 0.001
            img = img[np.newaxis, :, :, np.newaxis]
            pred = model.predict(img)
            if target == "Cell":
                pred = pred[0, :, :, 0] * 255
            else:
                pred = (pred[0, :, :, 0] > 0.4) * 255
        io.imsave(output_path + "\\" + target + ".tiff", pred.astype(np.uint8))


def main(filename):

    model_paths = {"Cell": os.path.join(cwd, "models", "cell_model.hdf5"),
                   "Nuclei": os.path.join(cwd, "models", "nuclei_model.hdf5"),
                   "Nucleoli": os.path.join(cwd, "models", "nucleoli_model.hdf5"),
                   "Mitochondria": os.path.join(cwd, "models", "mitochondria_model.sav")}
    runPred(filename, model_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    filename = args.filename
    if not os.path.isfile(filename):
        print("The file doesn't exist.")
        raise SystemExit(1)
    filename = args.filename
    main(filename)
