# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 16:25:50 2022

@authors: agathe.marguier, lorenzo.archetti, alexandre.jeanne, mathieu.frechin

This module contains the prediction class.
"""

import cv2
import logging
import logging.config
import numpy as np
import pickle
import time

from datetime import datetime
from os import makedirs, listdir
from os.path import join, isdir, basename, dirname, realpath
from skimage.io import imread
from skimage.morphology import erosion, square
from skimage.measure import label
from typing import List, Dict
from vigra.filters import hessianOfGaussian, tensorEigenvalues


log_file_path = join(realpath('.'),'voxel_classifier','logging.conf')
logging.config.fileConfig(log_file_path)
logger = logging.getLogger('training')

def singleThreadPrediction(clf, subX, idx, totidx):
    """
    Single-threaded prediction.

    Args:
        clf (sklearn classifier): Trained classifier.
        subX (np.array): Subset of image(s) to classify.
        idx (int): Prediction number.
        totidx (int): Total predictions.
    Returns:
        
        pred (np.array): Prediction (as probabilities) for the input image(s).

    """
    
    logger.debug('PRED-THREADING: starting prediction {}/{}'.format(idx, totidx))
    time_start = time.perf_counter()
    pred = clf.predict_proba(subX)
    time_elapsed = round((time.perf_counter() - time_start),1)
    outString = 'PRED-THREADING: predictions {}/{} took {} sec'.format(
        idx, totidx, time_elapsed)
    logger.debug(outString)
    
    return pred


def core_smart_stain_ML(x, glImSz, clf):
    """
    Performs the classification.

    Args:
        x (list): List corresponding to each timepoint and (n,m) array as
                  values whers n is the number of pixels and m the number of features.
        glImSz (list): List of image sizes at each timepoint.
        clf (sklearn classifier): Trained classifier.

    Returns:
        prob_map (numpy.ndarray): (n,m) array where n is the number of pixels
                                  of all images and m the number of class
                                  prediction (in our case 2, mitochondria and
                                  background)

    """
    
    
    if len(glImSz[0]) == 2:
        sz2cut = int(glImSz[0][0]*(glImSz[0][1]/4))
    if len(glImSz[0]) == 3:
        sz2cut = int(glImSz[0][1]*(glImSz[0][2]/4))

    tot_len = 0
    for size in glImSz:
        if len(size) == 2:
            tot_len += size[0]*size[1]
        elif len(size) == 3:
            tot_len += size[0]*size[1]*size[2]

    xtogether = np.zeros((tot_len, x[0].shape[1]), dtype=np.float32)

    p1 = 0
    p2 = 0
    for iXn in x:
        p2 += iXn.shape[0]
        xtogether[p1:p2, :] = iXn
        p1 = p2
    del x

    how_to_cut = int(xtogether.shape[0]/sz2cut)
    xlist = []
    all_processed = False
    tmp_check = list(range(how_to_cut))
    for i in range(how_to_cut):
        pos1 = i*sz2cut
        pos2 = pos1+sz2cut
        if i == tmp_check[-1]:
            pos2 = 0
            for iimtmp in glImSz:
                tmppos = 1
                for isztmp in iimtmp:
                    tmppos *= isztmp
                pos2 += tmppos
        if pos2 == xtogether.shape[0]:
            all_processed = True
        if pos2 > xtogether.shape[0]:
            pos2 = xtogether.shape[0]
            all_processed = True
        xlist.append(xtogether[pos1:pos2, :])
        if all_processed:
            break
    xtogether = None

    idx = 0
    totidx = len(xlist)
    
    
    probs= []
    for subX in xlist:
        idx += 1
        x = singleThreadPrediction(clf, subX, idx, totidx)
        probs.append(x)
    idx = 0
    for p in probs:
        idx += 1
        if idx == 1:
            prob_map = p
        else:
            prob_map = np.vstack((prob_map, p))

    return prob_map

def smart_stain_run_ML_new(x, glImSz, clf):
    '''
    Calls the function that performs the classification and from the output reconstitutes the original image shape.

    Args:
        x (dict): N keys corresponding to each timepoint and (n,m) array as
                  values whers n is the number of pixels and m the number of features..
        glImSz (list): List of images size.
        clf (sklearn classifier): Trained classifier to use for classification.

    Returns:
        prob_img (list): List of probability maps where each element correspond
                         to each image in exp_img.

    '''
    prob_img = []
    results = {}
    
    # Run model
    tmp = core_smart_stain_ML(x=x, glImSz=glImSz, clf=clf)
    del x
    
    # Split the model output array for each image, each pixels are on one column.
    results = []
    p1 = 0
    p2 = 0
    for i in range(len(glImSz)):
        tot_len = 1
        for ii in glImSz[i]:
            tot_len *= ii
        p2 += tot_len
        results.append(tmp[p1:p2, :])
        p1 += tot_len
    del tmp
    
    # Reshape results like input images to create the probability map
    prob_img = []
    for i_mages in range(len(results)):
        tmp1 = np.reshape(results[i_mages][:, 0], glImSz[i_mages])
        prob_img.append(tmp1)
    prob_img=np.asarray(prob_img, dtype=np.float32)

    return prob_img

class Prediction:
    """
    Apply a trained classifier to segment mitochondria.

    Args:
        riPath (str): Path of images to predict.
        features(list, optional): List of features as numpy array, each element
                                  in the list corresponds to one feature-extracted
                                  flattened RI image. Defaults to None.
        models (dict, optional): previously loaded model (for testing only).
                                 Defaults to None.
        model_path (str, optional): Path to a saved model. Defaults to "".
        outputPath (str): Path where the masks and or probability maps are saved.

    Attributes:
        area (int): Post processing parameter for minimum area.
        adpt_t (int): Post processing parameter for adaptative threshold.
        dim (str): Dataset dimension.
        erod (int): Post processing parameter for erosion.
        exp_img (List[np.array]): List of all input images.
        features(list): List of features as numpy array, each element in the
                        list corresponds to one RI image.
        img_info_dict(dict): Dictionnary with the size of images use during features extraction.
        ml_model (sklearn.ensemble): Model used for prediction.
        outputPath (str): Path where the masks and or probability maps are saved.
        pm_list (List[np.array]): List of all probability maps for each TP.
        riPath (str): Path of image to predict.
        target (str): Cell structure to classify.
        thresh (int): Post processing parameter for threshold.
    """

    def __init__(self, SMAType, riPath, features=None, img_info_dict=None, modelPath="", models=None, outputPath=""):

        if SMAType == "ML":
            assert models or modelPath, 'Prediction requires either models or a path to the models.'
            
            if models is None:
                with open(modelPath, 'rb') as file:
                    models = pickle.load(file)
    
            self.ml_model = models.get("Classifier")
    
            self.dim = models.get("Dimensions", "2d")
            self.target = models.get("Target", "Mitochondria")
    
            postprocKwargs = models["Postprocessing_params"]
    
            self.thresh = postprocKwargs.get('Threshold', 0)
            self.adpt_t = postprocKwargs.get('Adpt_threshold', 0)
            self.erod = postprocKwargs.get('Erosion', 1)
            self.area = postprocKwargs.get('Min_area', 1)
        
        else:
            self.ml_model = None
            self.dim = "2d"
            self.target = "Mitochondria"
            self.modelPath = modelPath
            self.thresh = 0
            self.adpt_t = 0
            self.erod = 1
            self.area = 1
            
        self.SMAType = SMAType
        self.outputPath = outputPath
        self.riPath = riPath
        self.features = features
        self.img_info_dict = img_info_dict
        self.pm_list = []

        

    def run_prediction(self, save_output=True):
        '''
        Orchestrate the prediction.

        Args:
            save_output (bool, optional): True if mask and maps are saved.
                                          Defaults to True.

        '''
        now=datetime.now()
        stringNow=now.strftime("%Y%m%d_%H%M") 

        if not self.outputPath:
            if self.riPath.endswith('.tiff') or self.riPath.endswith('.tif'):
                self.outputPath = join(dirname(dirname(dirname(self.riPath))),"SMA_output")
            elif 'RI' in basename(self.riPath) or 'ri' in basename(self.riPath):
                self.outputPath = join(dirname(dirname(self.riPath)),"SMA_output")
            elif isdir(self.riPath):
                self.outputPath = join(dirname(self.riPath),"SMA_output")
            else:
                raise Exception('Changing in the SMA: it is mandatory to enter the folder containing the images(riPath)')

        if self.dim == '2d':
            if not self.features:
                if self.riPath.endswith('.tiff') or self.riPath.endswith('.tif'):
                    featuresPath = dirname(dirname(self.riPath))
                elif 'RI' in basename(self.riPath) or 'ri' in basename(self.riPath):
                    featuresPath = dirname(self.riPath)
                elif isdir(self.riPath):
                    featuresPath = self.riPath
                else:
                    raise Exception('Changing in the SMA: it is mandatory to enter the folder containing the images(riPath)')
                print("Loading existing features...")
                self.features = []
                correctFeaturesFolder = [x for x in listdir(featuresPath) if isdir(
                    join(featuresPath, x)) and x.startswith("features")][-1]
                for y in listdir(join(featuresPath, correctFeaturesFolder)):
                    self.features.append(pickle.load(
                        open(join(featuresPath, correctFeaturesFolder, y), 'rb')))
                print("Features loaded")

                with open(join(featuresPath, 'img_info.sav'), 'rb') as openfile:
                    # Load the dictionary from the file
                    self.img_info_dict = pickle.load(openfile)

            glImSz = []
            for value in self.img_info_dict.values():
                if value[2] == 0:
                    glImSz.append(value[:2])
                else:
                    for i in range(value[2]):
                        glImSz.append(value[:2])

            print('Prediction start')
            self.pm_list = list()
            prediction_masks = smart_stain_run_ML_new(self.features, glImSz, self.ml_model)
            self.pm_list = prediction_masks

            if save_output:
                img_name = 'bulk_prediction'
                if not isinstance(self.riPath, List) and not isinstance(self.riPath, Dict):
                    img_name = basename(self.riPath).split('.')[0]
                path = self.outputPath + '/' + 'prediction_' + img_name + "_" + stringNow
                if not isdir(path):
                    makedirs(path)
            self.run_pp()

            if save_output:
                mask_list = self.save_mask(path, ind_2_save=save_output if isinstance(save_output, List) else None)
                return mask_list     

    def run_pp(self, proba_maps_path=''):
        '''
        Post-Processes the probability maps to make masks.

        Args:
            proba_maps_path (str, optional): Folder path where probability maps
                                             are saved from a previous prediction.
                                             Defaults to ''.
        '''
        if proba_maps_path:
            self.pm_list = []
            print('Probability maps loading')
            for file in listdir(proba_maps_path):
                if file.endswith('.sav'):
                    pm_path = join(proba_maps_path, file)
                    maps = pickle.load(open(pm_path, 'rb'))
                    prediction = maps[:, :]
                if file.endswith('.tiff'):
                    pm_path = join(proba_maps_path, file)
                    prediction = imread(pm_path)

                self.pm_list.append(prediction)
            print('Loading done')

        self.img_2_save = []
        self.tmp_img_2_save =[]
        print('Starts post processing thresh: {}, adpt_thresh: {}, erosion{}, area_filtering{}'.format(
            self.thresh, self.adpt_t, self.erod, self.area))

        for prediction_map in self.pm_list:
            
            if self.SMAType == "ML":
                thresholding = np.copy(prediction_map)
                thresholding = thresholding * 255
                thresholding = thresholding.astype('uint8')
    
                if self.adpt_t > 1:
                    rmv = -1
                    mask = cv2.adaptiveThreshold(
                        thresholding, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.adpt_t, rmv)
                    mask = np.where(mask,prediction_map,0)
                else:
                    mask = thresholding
    
                mask = np.where(mask >= self.thresh, 255, 0).astype(np.uint8)
                
            elif self.SMAType == "UNet":
                mask = np.where(prediction_map >= self.thresh, 255, 0).astype(np.uint8)
                
            erosion_map = erosion(mask, square(self.erod))
            labels = label(erosion_map, background = 0, connectivity=2)
            classes, objects_size = np.unique(labels, return_counts=True)

            goodlabels=[]
            for currLabel, npx in zip(classes, objects_size):
                if npx > self.area and currLabel != 0:
                    goodlabels.append(currLabel)
            mask = np.isin(labels, goodlabels)
            mask = np.where(mask, 1, 0)
            
            enhanceMasks = False
            if enhanceMasks:
                mask = mask.astype(np.float32)
                mask = hessianOfGaussian(mask, 1)
                mask = tensorEigenvalues(mask)
                amax = mask.max(axis = 2)
                amin = mask.min(axis = 2)
                mask = np.where(-amin > amax, amin, amax) * -1
                mask = np.clip(mask, 0, 1)
                binarizeEM = True
                if binarizeEM:
                    mask = np.where(mask >= 0.1, 255, 0).astype(np.uint8)
                else:
                    mask = np.asarray(mask, dtype=np.float32)
            else:
                mask = np.asarray(mask, dtype='uint8') * 255

            self.img_2_save.append(mask)

        print('Post-processing done')

    def save_mask(self, path, mask=True, raw_data=False, ind_2_save=None):
        '''
        Saves the post-processing results.

        Args:
            path (str): Output path where maps and masks are saved.
            mask (bool, optional): If True, saves binary masks. Defaults to True.
            raw_data (bool, optional): If True, saves the probability mask as
                                       png. Defaults to False.
            ind_2_save (List, optional): Only used with binary masks, non-overlay,
                                         non-raw_data. Only saves images at the
                                         specified indices, or all images if none
                                         are specified.
        '''
        if mask:
            print('Saving masks')
            mask_list = list()
            for tp, mask in enumerate(self.img_2_save):
                assert ind_2_save is None or isinstance(ind_2_save, List)
                if ind_2_save is None or len(ind_2_save) == 0 or tp in ind_2_save:
                    if not isdir(join(path, 'Binary_masks')):
                        makedirs(join(path, 'Binary_masks'))
                    mask_list.append(mask)
            return mask_list
            print('Masks saved')

