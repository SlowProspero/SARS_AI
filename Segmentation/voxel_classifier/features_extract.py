# -*- coding: utf-8 -*-
"""
@authors: agathe.marguier, lorenzo.archetti, alexandre.jeanne, mathieu.frechin

This module contains the FeaturesExtract class.
"""

import cv2
import logging
import logging.config
import numpy as np
import pickle
import time
import vigra

from os import makedirs, listdir, path
from os.path import join, isdir, dirname, isfile, basename
from skimage.feature import local_binary_pattern
from skimage.io import imread
from typing import List


log_file_path = path.join(path.realpath('.'),'voxel_classifier','logging.conf')
print(log_file_path)
logging.config.fileConfig(log_file_path)
logger = logging.getLogger('training')


       
def runSingleThread(i_filt_list, sigma, deltaSigma, inOutScaleDiff, exp_img, rad, return_timing=False, lbprad=3):
    """
    Run single-threaded feature extraction.

    Args:
        i_filt_list (dict): Dictionary containing filters to use.
        sigma (float): Sigma value for filters.
        deltaSigma (float): Delta sigma for difference of gaussian filter.
        inOutScaleDiff (float): Inner-outer scale difference for structure tensor calculation.
        exp_img (TYPE): Image to filter.
        rad (int): radius for kuwuhara filter (if method == mean).
        return_timing (bool, optional): Whether to return time needed for feature extraction. Defaults to False.
        lbprad (int, optional): Radius for local binary pattern filter. Defaults to 3.

    Returns:
        fimg (dict): Dictionary containing the filtered images with the name of the filter as keys.

    """
    exp_img = exp_img.astype(np.float32)
    fimg = {}
    filter_timing = {}
    hess=None
    grad=None
    structTens=None
    s = np.asarray(0.6)
    s = np.array(s, dtype='float32')
    
    for i_filt in i_filt_list:      
        start_time = time.perf_counter()
        if i_filt == 'baseImg':
            fimg[i_filt] = exp_img
            
        elif i_filt == 'ggradient':
            if grad is not None:
                fimg[i_filt+'-sgm' + str(sigma)] = np.sum(grad, axis=2)
            else:
                if sigma > s:
                    grad = vigra.filters.gaussianGradient(exp_img, sigma)
                    fimg[i_filt+'-sgm' + str(sigma)] = np.sum(grad, axis=2)
      
        elif i_filt == 'ggradientmag':
            if grad is not None:
                fimg[i_filt+'-sgm' + str(sigma)] = np.sqrt(grad[:,:,0]**2+grad[:,:,1]**2)
            else:
                if sigma > s:
                    grad = vigra.filters.gaussianGradient(exp_img, sigma)
                    fimg[i_filt+'-sgm' + str(sigma)] = np.sqrt(grad[:,:,0]**2+grad[:,:,1]**2)
            
        elif i_filt == 'hessianOfG':
            if hess is not None:
                fimg[i_filt+'-sgm' + str(sigma)] = np.stack((hess[:,:,0]+hess[:,:,2],hess[:,:,1]),axis=2)
            else:
                if sigma > s:
                    hess = vigra.filters.hessianOfGaussian(exp_img, sigma)
                    fimg[i_filt+'-sgm' + str(sigma)] = np.stack((hess[:,:,0]+hess[:,:,2],hess[:,:,1]),axis=2)
         
        elif i_filt == 'hessianOfGEigen' or i_filt == 'Laplacian':  
            if hess is not None:
                if i_filt == 'hessianOfGEigen':
                    fimg[i_filt+'-sgm' + str(sigma)] = vigra.filters.tensorEigenvalues(hess)
                elif i_filt == 'Laplacian':
                    fimg[i_filt+'-sgm' + str(sigma)] = hess[:,:,0] + hess[:,:,2]
            else:
                if sigma > s:
                    hess = vigra.filters.hessianOfGaussian(exp_img, sigma)
                    if i_filt == 'hessianOfGEigen':
                        fimg[i_filt+'-sgm' + str(sigma)] = vigra.filters.tensorEigenvalues(hess)
                    elif i_filt == 'Laplacian':
                        fimg[i_filt+'-sgm' + str(sigma)] = hess[:,:,0] + hess[:,:,2]
    
        elif i_filt == 'gsmoothing':
            if sigma > 0:
                fimg[i_filt+'-sgm' + str(sigma)] = vigra.filters.gaussianSmoothing(exp_img, sigma)
            
        elif i_filt == 'gsharpening2D':
            if sigma > 0:
                fimg[i_filt+'-sgm' + str(sigma)] = vigra.filters.gaussianSharpening2D(exp_img, sigma)
            
        elif i_filt == 'difOfGauss':
            if sigma > s:
                if isinstance(deltaSigma, List):
                    for dsig in deltaSigma:
                        fimg[i_filt+'-sgm' + str(sigma) + '-dsg' + str(dsig)] = gauss_diff(exp_img, sigma, sigma+dsig)
                else:
                    fimg[i_filt+'-sgm' + str(sigma)] = gauss_diff(exp_img, sigma, sigma+deltaSigma)
    
        elif i_filt == 'StrctTen' or i_filt == 'TensorDeterm' or i_filt == 'TensorEigenV' or i_filt == 'TensorTrace':
            if structTens is not None:
                if i_filt == 'StrctTen':
                    fimg[i_filt+'-sgm' + str(sigma)] = np.stack((structTens[:,:,0]+structTens[:,:,2],structTens[:,:,1]),axis=2)
                elif i_filt == 'TensorDeterm':
                    fimg[i_filt+'-sgm' + str(sigma)] = vigra.filters.tensorDeterminant(structTens)
                elif i_filt == 'TensorEigenV':
                    fimg[i_filt+'-sgm' + str(sigma)] = vigra.filters.tensorEigenvalues(structTens)
                elif i_filt == 'TensorTrace':
                    fimg[i_filt+'-sgm' + str(sigma)] = vigra.filters.tensorTrace(structTens)
            else:
                if sigma > s:
                    structTens=vigra.filters.structureTensor(exp_img, sigma, sigma+inOutScaleDiff)
                    if i_filt == 'StrctTen':
                        fimg[i_filt+'-sgm' + str(sigma)] = np.stack((structTens[:,:,0]+structTens[:,:,2],structTens[:,:,1]),axis=2)
                    elif i_filt == 'TensorDeterm':
                        fimg[i_filt+'-sgm' + str(sigma)] = vigra.filters.tensorDeterminant(structTens)
                    elif i_filt == 'TensorEigenV':
                        fimg[i_filt+'-sgm' + str(sigma)] = vigra.filters.tensorEigenvalues(structTens)
                    elif i_filt == 'TensorTrace':
                        fimg[i_filt+'-sgm' + str(sigma)] = vigra.filters.tensorTrace(structTens)
                        
        elif i_filt == 'kuwahara':
            fimg[i_filt+'-rad' + str(rad)] = kuwahara(exp_img, radius=rad)
            
        elif i_filt == 'LocalBinaryPattern':
            lbp_input = np.copy(exp_img)
            if sigma > s:
                    lbp_input = vigra.filters.gaussianSmoothing(exp_img, sigma)
            if isinstance(lbprad, List):
                for lbprad_val in lbprad:
                    fimg[i_filt+'-rad' + str(lbprad_val)+'-sgm' + str(sigma)] = np.asarray(local_binary_pattern(lbp_input, P=min(31, int(lbprad_val*4)), R=lbprad_val, method='uniform'), dtype=np.float32)
            else:
                fimg[i_filt+'-rad' + str(lbprad)] = np.asarray(local_binary_pattern(lbp_input, P=min(int(lbprad*4), 31), R=lbprad, method='uniform'), dtype=np.float32)

        end_time = time.perf_counter()
        filter_timing[i_filt] = end_time-start_time

    # if len(fimg) == 0:
    #     fimg['base_img'] = exp_img

    if return_timing:
        return fimg, filter_timing
    return fimg

def gauss_diff(img, sig, highSig):
    """
    Apply difference of gaussian with the high and low sigmas as parameters.

    Args:
        img (numpy.ndarray): (w,h) array which contains all pixel's values for
                             a timepoint before filter.
        sig (float): low sigma's value to use for the image to subtract from the high sigma image.
        highSig (float): high sigma value used for the image from which the low sigma value image is subtracted.

    Returns:
        img_out (numpy.ndarray): (w,h) filtered array.
    """

    diffH = vigra.filters.gaussianSmoothing(img, sigma=highSig)
    diffL = vigra.filters.gaussianSmoothing(img, sigma=sig)
    img_out = diffH - diffL
    return img_out


def vigra_feat_extrctn(sigmas, deltaSigma, inOutScaleDiff, filters, includeSeparateChannels, exp_img, return_timing=False, lbprad=3, fnames=None):
    """
    Call the vigra's functions and other filters one by one and run them with
    the *MyOwnThreading* module, before to merge the results.

    Args:
        sigmas (numpy.ndarray): sigma's values to parameter the filter.
        deltaSigma (float)
        inOutScaleDiff (float)       
        filters (dict): {str:bool} dictionary with names of filters as keys and
                        boolean as values.
        includeSeparateChannels (bool): Whether to also include separate channels of filters whose output has more than one channel (ignoring those dependent on directionality).
        exp_img (numpy.ndarray): (w,h) array which contains all pixel's values
                                 for a timepoint.
        return_timing (bool, optional): Whether to return the time needed for feature extraction. Defaults to False.
        lbprad (int, optional): radius for the local binary pattern filter. Defaults to 3.

    Returns:
        X (numpy.ndarray): (n,m) array where n is the number of pixels and m
                           the number of features.

    """
    
    #Set radii for kuwuhara filter proportional to chosen sigmas
    radii = []
    tmpSigmas=np.sort(sigmas)
    for sigma in tmpSigmas:
        if sigma<1:
            radius=1
        else:
            radius=int(sigma)
        if radius not in radii:
            radii.append(radius)
        else:
            notAdded=True
            while notAdded:
                if radius not in radii:
                    notAdded=False
                    radii.append(radius)
                else:
                    radius+=1

    # Select filters used for features extraction
    filt_list = []
    possibleFilters = ['baseImg',
                       'gsmoothing', 
                       'gsharpening2D',
                       'difOfGauss',
                       'ggradient',
                       'ggradientmag',
                       'hessianOfG',
                       'hessianOfGEigen',
                       'Laplacian',
                       'HessianDeterm',
                       'LocalBinaryPattern',
                       'StrctTen',
                       'TensorTrace',
                       'TensorDeterm',
                       'TensorEigenV',
                       'kuwahara']
    
    for filt in filters.keys():
        if filters[filt] == True and filt in possibleFilters:
            filt_list.append(filt)
            
    # filt_list_comp_time = dict()  # to have computation time

    # Apply each filter
    fimg = {}
    filter_timings = []
    for i_ksize, sigma in enumerate(sigmas):
        if return_timing:
            currfimg, filter_timing = runSingleThread(filt_list, sigma, deltaSigma, inOutScaleDiff, exp_img, radii[i_ksize], return_timing, lbprad)
            filter_timings.append(filter_timing)
        else:
            currfimg=runSingleThread(filt_list, sigma, deltaSigma, inOutScaleDiff, exp_img, radii[i_ksize], return_timing=False, lbprad=lbprad)
        fimg.update(currfimg)
       # filt_list_comp_time[i_filt] = (time.perf_counter()-time_start)
    
    
    logger.debug('VIGRAFEATEXTRCT: merging feature dictionnaries OK')

    """
    the goal now is to organize the data, on one side the labels, on the other
    the feature space
    """
    x=[]
    if fnames is None: 
        fnames=[]
    fnames += sorted(fimg.keys())
    for idx1, i_filt in enumerate(sorted(fimg.keys())):
        tmp = fimg[i_filt]
        tdim = np.asarray(tmp.shape)
        if tdim.shape[0] == 2:
            x.append(tmp.flatten())
        elif tdim.shape[0] > 2:
            if includeSeparateChannels:
                tmpS = np.sum(tmp, axis=2)
                x.append(tmpS.flatten())
                for chn in range(tmp.shape[2]):
                    x.append(tmp[:,:,chn].flatten())
            else:
                tmp = np.sum(tmp, axis=2)
                x.append(tmp.flatten())
                
    x=np.transpose(x)
    
    if return_timing:
        return x, filter_timings
    return x


def kuwahara(orig_img, method='mean', radius=3, sigma=None):
    """
    Kuwuhara filter implementation. The Kuwahara filter is a non-linear smoothing filter used for adaptive noise reduction.

    Args:
        orig_img (TYPE): Image to filter.
        method (str, optional): Kuwuhara method to use, either mean or gaussian. Defaults to 'mean'.
        radius (int, optional): Radius to use if method == mean. Defaults to 3.
        sigma (float, optional): Sigma to use if method == gaussian. Defaults to None.

    Raises:
        NotImplementedError: Method needs to be either mean or gaussian.

    Returns:
        filtered (numpy.ndarray): filtered array

    """

    if method not in ('mean', 'gaussian'):
        raise NotImplementedError('unsupported method %s' % method)
    if method == 'gaussian' and sigma is None:
        sigma = -1
    image = orig_img.astype(np.float32, copy=False,)
    avgs = np.empty((4, *image.shape), dtype=image.dtype)
    stddevs = np.empty((4, *image.shape[:2]), dtype=image.dtype)
    squared_img = image ** 2
    if method == 'mean':
        kxy = np.ones(radius + 1, dtype=image.dtype) / (radius + 1)    # kernelX and kernelY (same)
    elif method == 'gaussian':
        kxy = cv2.getGaussianKernel(2 * radius + 1, sigma, ktype=cv2.CV_32F)
        kxy /= kxy[radius:].sum()   # normalize the semi-kernels
        klr = np.array([kxy[:radius+1], kxy[radius:]])
        kindexes = [[1, 1], [1, 0], [0, 1], [0, 0]]
    shift = [(0, 0), (0,  radius), (radius, 0), (radius, radius)]
    for k in range(4):
        if method == 'mean':
            kx = ky = kxy
        elif method == 'gaussian':
            kx, ky = klr[kindexes[k]]
        cv2.sepFilter2D(image, -1, kx, ky, avgs[k], shift[k])
        cv2.sepFilter2D(squared_img, -1, kx, ky, stddevs[k], shift[k])
        # compute the final variance on subwindow
        stddevs[k] = stddevs[k] - avgs[k] ** 2
    indices = np.argmin(stddevs, axis=0)
    if image.ndim == 2:
        filtered = np.take_along_axis(
            avgs, indices[None, ...], 0).reshape(image.shape)
    else:   # then avgs.ndim == 4
        filtered = np.take_along_axis(
            avgs, indices[None, ..., None], 0).reshape(image.shape)

    return filtered.astype(orig_img.dtype)



class FeaturesExtract:
    """
    Extract features from given ri thanks to chosen filters.

    Args:
        riPath (str or list): Path of the tiff image or list of numpy images.
        model (str): Path to exisiting model or model structure. Defaults to "".
        models (dict, optional): Previously loaded model (for testing only). Defaults to None.
        saveFeatures (str, optional): Whether to save features as files or not. Defaults to False.
        return_timing (bool, optional): Whether to return the timing for feature extraction. Defaults to False.
        kwargs(dict, optional): Advanced parameters that can be changed when training. Defaults to {"Feature_extraction_params": {"Used_projections": {'max':True, 'mean':False, 'median':False, 'min':False, 'std':False, 'sum':False},
                                                                                                                                  "Sigmas": [0], 
                                                                                                                                  "Delta_sigma": 0,
                                                                                                                                  "Inner_outer_scale_difference": 0,
                                                                                                                                  "includeSeparateChannels": False,
                                                                                                                                  "Lbp_radius": 0,
                                                                                                                                  "Filters": {"max": {'baseImg': True, 'gsmoothing': True, 'difOfGauss': True, 'ggradient': True, 'ggradientmag': True,
                                                                                                                                                      'hessianOfG': True, 'hessianOfGEigen': True, 'Laplacian': True, 'StrctTen': True,
                                                                                                                                                      'TensorDeterm': True, 'TensorEigenV': True, 'kuwahara': True,'LocalBinaryPattern': True},
                                                                                                                                              "mean": {'baseImg': False, 'gsmoothing': False, 'difOfGauss': False, 'ggradient': False, 'ggradientmag': False,
                                                                                                                                                       'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': False, 'StrctTen': False,
                                                                                                                                                       'TensorDeterm': False, 'TensorEigenV': False, 'kuwahara': False,'LocalBinaryPattern': False},
                                                                                                                                              "median": {'baseImg': False, 'gsmoothing': False, 'difOfGauss': False, 'ggradient': False, 'ggradientmag': False,
                                                                                                                                                         'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': False, 'StrctTen': False,
                                                                                                                                                         'TensorDeterm': False, 'TensorEigenV': False, 'kuwahara': False,'LocalBinaryPattern': False},
                                                                                                                                              "min": {'baseImg': False, 'gsmoothing': False, 'difOfGauss': False, 'ggradient': False, 'ggradientmag': False,
                                                                                                                                                      'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': False, 'StrctTen': False,
                                                                                                                                                      'TensorDeterm': False, 'TensorEigenV': False, 'kuwahara': False,'LocalBinaryPattern': False},
                                                                                                                                              "std": {'baseImg': False, 'gsmoothing': False, 'difOfGauss': False, 'ggradient': False, 'ggradientmag': False,
                                                                                                                                                      'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': False, 'StrctTen': False,
                                                                                                                                                      'TensorDeterm': False, 'TensorEigenV': False, 'kuwahara': False,'LocalBinaryPattern': False},
                                                                                                                                              "sum": {'baseImg': False, 'gsmoothing': False, 'difOfGauss': False, 'ggradient': False, 'ggradientmag': False,
                                                                                                                                                      'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': False, 'StrctTen': False,
                                                                                                                                                      'TensorDeterm': False, 'TensorEigenV': False, 'kuwahara': False,'LocalBinaryPattern': False}},
                                                                                                                                  }
                                                                      
                                                                                                    "Preprocessing_params": {"Normalize_on_first": True},
                                                    
                                                                                                    "Postprocessing_params": {"Threshold": 0,
                                                                                                                              "Adpt_threshold": 0,
                                                                                                                              "Erosion": 1,
                                                                                                                              "Min_area": 1}
                                                                                                    }

    Attributes:
        deltaSigma (float or list[float]): difference of sigmas to use for difference of gaussian filter.
        dim (str): Dimensions of the input image (only 2d is currently supported).
        erod (int): Post processing parameter for erosion.
        exp_img (List[np.array]): List of all input images.
        filters (dict): Filter dictionary for all projections, with each projection containing a dictionary with all the filter names as keys and booleans as
                        values. The boolean determines if the filter is used during the processing or not.
        img_info_dict(dict): Dictionnary with the size of images use during features extraction.
        includeSeparateChannels (bool): Whether to also include separate channels of filters whose output has more than one channel.
        inOutScaleDiff(int): Difference between inner and outer scale for structure tensor calculation.
        lbprad(int): Radius for local binary pattern filter.
        normOnFirst (bool): When training, tells if the preprocessing pipeline is fitted based only on the first image or all.
        ppPipeline (sklearn.pipeline): Preprocessing pipeline (either already fitted or to fit).
        predicting(bool): Tells if the features are being calculated for predicting or training.
        projectionUsed (dict): Dictionary to define which other projections are used.
        return_timing (bool): Whether to return the timing for feature extraction.
        ri (str or list): Path of the tiff image or list of numpy images.
        savePath (str): Path where features are saved.
        saveFeatures (bool): If True, array with features for each TP are saved.
        sigmas(List[float]): List of sigmas to apply as filter parameters.
        target (str): Cell structure for downstream prediction (used only to label saved features).
    """

    def __init__(self, riPath, model="", models=None, saveFeatures=False, return_timing=False, **kwargs):

        self.return_timing = return_timing
        logger.debug('Initialization...')
        if isinstance(model, str):
            self.predicting = True
            if models is None:
                with open(model, 'rb') as file:
                    models = pickle.load(file)

            featExtKwargs = models["Feature_extraction_params"]
            preprocKwargs = models["Preprocessing_params"]

            self.ppPipeline = models.get("Preprocessing_pipeline")

        else:
            self.predicting = False

            featExtKwargs = kwargs["Feature_extraction_params"]
            preprocKwargs = kwargs["Preprocessing_params"]

            self.ppPipeline = model[:-1]

        self.projectionUsed = featExtKwargs.get('Used_projections', {'max':True, 'mean':False, 'median':False, 'min':False, 'std':False, 'sum':False})
        self.filters = featExtKwargs.get('Filters', {"max": {'baseImg': True, 'gsmoothing': False, 'difOfGauss': True, 'ggradient': True, 'ggradientmag': False,
                                                             'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': True, 'StrctTen': True,
                                                             'TensorDeterm': True, 'TensorEigenV': False, 'kuwahara': True,'LocalBinaryPattern': True},
                                                     "mean": {'baseImg': False, 'gsmoothing': False, 'difOfGauss': False, 'ggradient': False, 'ggradientmag': False,
                                                              'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': False, 'StrctTen': False,
                                                              'TensorDeterm': False, 'TensorEigenV': False, 'kuwahara': False,'LocalBinaryPattern': False},
                                                     "median": {'baseImg': False, 'gsmoothing': False, 'difOfGauss': False, 'ggradient': False, 'ggradientmag': False,
                                                                'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': False, 'StrctTen': False,
                                                                'TensorDeterm': False, 'TensorEigenV': False, 'kuwahara': False,'LocalBinaryPattern': False},
                                                     "min": {'baseImg': False, 'gsmoothing': False, 'difOfGauss': False, 'ggradient': False, 'ggradientmag': False,
                                                             'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': False, 'StrctTen': False,
                                                             'TensorDeterm': False, 'TensorEigenV': False, 'kuwahara': False,'LocalBinaryPattern': False},
                                                     "std": {'baseImg': False, 'gsmoothing': False, 'difOfGauss': False, 'ggradient': False, 'ggradientmag': False,
                                                             'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': False, 'StrctTen': False,
                                                             'TensorDeterm': False, 'TensorEigenV': False, 'kuwahara': False,'LocalBinaryPattern': False},
                                                     "sum": {'baseImg': False, 'gsmoothing': False, 'difOfGauss': False, 'ggradient': False, 'ggradientmag': False,
                                                             'hessianOfG': False, 'hessianOfGEigen': False, 'Laplacian': False, 'StrctTen': False,
                                                             'TensorDeterm': False, 'TensorEigenV': False, 'kuwahara': False,'LocalBinaryPattern': False}})
        if all(isinstance(v, bool) for v in self.filters.values()):
            self.filters = {"max": self.filters,
                            "mean": self.filters,
                            "median": self.filters,
                            "min": self.filters,
                            "std": self.filters,
                            "sum": self.filters}
                    
        for proj in self.projectionUsed.keys():
            if not any(self.filters[proj].values()):
                self.projectionUsed[proj]=False
                
        self.sigmas = featExtKwargs.get('Sigmas', [0])
        self.deltaSigma = featExtKwargs.get('Delta_sigma', 0)
        self.inOutScaleDiff = featExtKwargs.get('Inner_outer_scale_difference', 0)
        self.includeSeparateChannels = featExtKwargs.get('Include_separate_channels', False)
        
        self.lbprad = featExtKwargs.get('Lbprad', 0)        

        self.normOnFirst = preprocKwargs.get('Normalize_on_first', True)

        self.dim = kwargs.get('Dimensions', "2d")
        self.target = kwargs.get('Target', "Mitochondria")

        self.ri = riPath
        self.saveFeatures = saveFeatures
    
    def load_images(self):
        self.img_info_dict = {}
        self.exp_img = {}

        # self.__class__.load_images_ is identical to FeaturesExtract.load_images_, 
        # but more in line with the DRY principle.
        savePath = self.__class__.load_images_(ri=self.ri, img_info_dict=self.img_info_dict, exp_img=self.exp_img, img_names=None, projectionUsed=self.projectionUsed, dim=self.dim, saveFeatures=self.saveFeatures)

        # Set savePath / saveFeatures
        if savePath is None:
            self.saveFeatures = False
        else:
            self.savePath = savePath

    @staticmethod
    def load_images_(ri, img_info_dict: dict = {}, exp_img: dict = {}, img_names: dict = None, projectionUsed: dict = {'max':True, 'mean':False, 'median':False, 'min':False, 'std':False, 'sum':False}, dim='2d', saveFeatures=False, SMAType = "UNet"):
        if img_names is None: # Optimization functionality. Useful for if you want to keep only specific images
            img_names = {}
        '''Create self.exp_img in function of riPath type.'''
        if type(ri) == str:  # if from path:
            projections = []
            for key, value in projectionUsed.items():
                if value:
                    projections.append(key)
                    exp_img[key] = []
                    img_names[key] = []
            if (ri.endswith('.tiff') or ri.endswith('.tif')) and dim == '2d':
                # If tiff path
                if len(projections) == 1:
                    proj = projections[0]
                    savePath = dirname(dirname(ri))
                    currRi = imread(ri)
                    tiff_name = basename(ri)
                    if tiff_name.startswith(proj): 
                        key_name = tiff_name[len(proj)+1:]
                    else:
                        key_name = tiff_name
                    
                    if len(currRi.shape) == 2:
                        exp_img[proj].append(np.array(currRi))
                        img_names[proj].append(tiff_name)
                        sX = int(currRi.shape[0])
                        sY = int(currRi.shape[1])
                        img_info_dict[key_name] = [sX, sY, 0]
                    else:
                        if currRi.shape[2]==3 or currRi.shape[2]==4:
                            currRi = np.transpose(currRi,(2,0,1))
                        sX = int(currRi[0, :, :].shape[0])
                        sY = int(currRi[0, :, :].shape[1])
                        img_info_dict[key_name] = [sX, sY, currRi.shape[0]]
                        for i in range(0, currRi.shape[0]):
                            exp_img[proj].append(np.array(currRi[i, :, :]))
                            img_names[proj].append(tiff_name)

                else:
                    raise Exception('You give a tiff but you want to use differente projections ; rather give folder path as input')
            
            elif ri.endswith('.vol') or dim == '3d':
                # If vol path or 3D image
                logger.warning('3D not allowed yet')
                saveFeatures = False

            elif isdir(ri) and dim == '2d':
                # If folder path
                tiff_list = [f for f in listdir(ri) if f.endswith('.tiff') or f.endswith('.tif')]
                if tiff_list and len(projections) == 1:
                    savePath = dirname(ri)
                    proj = projections[0]
                    for ri_name in tiff_list:
                        currRi = imread(join(ri, ri_name))
                        if ri_name.startswith(proj): 
                            key_name = ri_name[len(proj)+1:]
                        else:
                            key_name = ri_name
                        if len(currRi.shape) == 2:
                            exp_img[proj].append(np.array(currRi))
                            img_names[proj].append(ri_name)
                            sX = int(currRi.shape[0])
                            sY = int(currRi.shape[1])
                            img_info_dict[key_name] = [sX, sY, 0]
                        else:
                            if currRi.shape[2]==3 or currRi.shape[2]==4:
                                currRi = np.transpose(currRi,(2,0,1))
                            sX = int(currRi[0, :, :].shape[0])
                            sY = int(currRi[0, :, :].shape[1])
                            img_info_dict[key_name] = [sX, sY, currRi.shape[0]]
                            for i in range(0, currRi.shape[0]):
                                exp_img[proj].append(np.array(currRi[i, :, :]))
                                img_names[proj].append(ri_name)
            
                else:
                    savePath = ri
                    tiff_dict = {}
                    folder_path_dict = {}
                    for proj in projections:
                        folder_path = None
                        if proj=='max':
                            if isdir(join(ri, 'RI')):
                                folder_path = join(ri, 'RI')
                            elif isdir(join(ri, 'ri')):
                                folder_path = join(ri, 'ri')
                            elif isdir(join(ri, 'ri_max')):
                                folder_path = join(ri, 'ri_max')
                            elif isdir(join(ri, 'RI_max')):
                                folder_path = join(ri, 'RI_max')
                            elif isdir(join(ri, 'max')):
                                folder_path = join(ri, 'max')
                        else:
                            if isdir(join(ri, 'RI_' + proj)):
                                folder_path = join(ri, 'RI_' + proj)
                            elif isdir(join(ri, 'ri_' + proj)):
                                folder_path = join(ri, 'ri_' + proj)
                            elif isdir(join(ri, proj)):
                                folder_path = join(ri, proj)
                            
                        if folder_path:
                            tiff_dict[proj] = [f for f in listdir(folder_path) if f.endswith('.tiff') or f.endswith('.tif')]
                            folder_path_dict[proj] = folder_path
                        else:
                            raise Exception('Folder which contains ' + proj + \
                                            ' projection not found.'\
                                            '\n The syntax for the folder is'\
                                            ' `RI_' + proj + '` or `ri_' + proj + '` or `' + proj + '`')
                            
                    if len(projections) == 1:
                        proj = projections[0]
                        for ri_name in tiff_dict[proj]:
                            if ri_name.startswith(proj):
                                key_name = ri_name[len(proj)+1:]
                            else:
                                key_name = ri_name
                            currRi = imread(join(folder_path_dict[proj], ri_name))
                            if len(currRi.shape) == 2:
                                exp_img[proj].append(np.array(currRi))
                                img_names[proj].append(ri_name)
                                sX = int(currRi.shape[0])
                                sY = int(currRi.shape[1])
                                img_info_dict[key_name] = [sX, sY, 0]
                            else:
                                if currRi.shape[2]==3 or currRi.shape[2]==4:
                                    currRi = np.transpose(currRi,(2,0,1))
                                sX = int(currRi[0, :, :].shape[0])
                                sY = int(currRi[0, :, :].shape[1])
                                img_info_dict[key_name] = [sX, sY, currRi.shape[0]]
                                for i in range(0, currRi.shape[0]):
                                    exp_img[proj].append(np.array(currRi[i, :, :]))
                                    img_names[proj].append(ri_name)
                        
                    else:
                        # Check that all folders contain the same number of images
                        init = False
                        for key, value in tiff_dict.items():
                            if not init:
                                nb_img = len(value)
                                init = True
                            else:
                                if nb_img != len(value):
                                    raise Exception('The number of images is not the same for each type of projection')

                        first_proj = projections[0]                        
                        first_list_path = tiff_dict[first_proj]
                        first_folder_path = folder_path_dict[first_proj]
                        for tiff_name in first_list_path:
                            if tiff_name.startswith(first_proj):
                                key_name = tiff_name[len(first_proj)+1:]
                            else:
                                key_name = tiff_name
                            currRi = imread(join(first_folder_path, tiff_name))
                            if len(currRi.shape) == 2:
                                exp_img[first_proj].append(np.array(currRi))
                                img_names[first_proj].append(tiff_name)
                                sX = int(currRi.shape[0])
                                sY = int(currRi.shape[1])
                                img_info_dict[key_name] = [sX, sY, 0]
                            else:
                                if currRi.shape[2]==3 or currRi.shape[2]==4:
                                    currRi = np.transpose(currRi,(2,0,1))
                                sX = int(currRi[0, :, :].shape[0])
                                sY = int(currRi[0, :, :].shape[1])
                                img_info_dict[key_name] = [sX, sY, currRi.shape[0]]
                                for i in range(0, currRi.shape[0]):
                                    exp_img[first_proj].append(np.array(currRi[i, :, :]))
                                    img_names[first_proj].append(tiff_name)
                            
                            for i in range(1, len(projections)):
                                proj = projections[i]
                                folder_path = folder_path_dict[proj]
                                if isfile(join(folder_path, key_name)): 
                                    tiff_path = join(folder_path, key_name)
                                elif isfile(join(folder_path, (proj + '_' + key_name))):
                                    tiff_path = join(folder_path, (proj + '_' + key_name))
                                else:
                                    raise Exception('No ' + proj + ' projection for image ' + key_name)
                                
                                currRi = imread(tiff_path)
                                if len(currRi.shape) == 2:
                                    exp_img[proj].append(np.array(currRi))
                                    img_names[proj].append(basename(tiff_path))
                                else:
                                    if currRi.shape[2]==3 or currRi.shape[2]==4:
                                        currRi = np.transpose(currRi,(2,0,1))
                                    for i in range(0, currRi.shape[0]):
                                        exp_img[proj].append(np.array(currRi[i, :, :]))
                                        img_names[proj].append(basename(tiff_path))
                            
                        
        elif type(ri) == list:  # if list of images = on type of projection
            saveFeatures = False
            for idx, img in enumerate(ri):
                img_info_dict[str(idx)] = [img.shape[0],img.shape[1],0]
                
            for key, value in projectionUsed.items():
                if value:
                    exp_img[key] = ri
                    break
        
        elif type(ri) == dict:

            for key, value in ri.items():
                if value:
                    if key not in projectionUsed:
                        raise Exception('You try to use an unknown projection: ' + key)
            
            for idx, img in enumerate(list(ri.values())[0]):
                img_info_dict[str(idx)] = [img.shape[0],img.shape[1],0]
            
            saveFeatures = False
            exp_img.clear()
            exp_img.update(ri)

        # Example of what not to do:
        # elif type(ri) == SomeNewType:
        #    exp_img = something
        # This would overwrite exp_img, rather than setting it with a value.
        # You can only modify exp_img, as seen above. To set exp_img with a new dict, instead do:
        # my_dict = {key: value}
        # exp_img.clear()
        # exp_img.update(my_dict)

        else:
            raise TypeError(
                'Can only use list of time points or str as the path to load the probability maps')
        if saveFeatures:
            return savePath
            
        
    def run_extract(self):
        '''
        Starts the features extraction.

        Returns:
            features(list): List containing, for each timepoint, extracted features after PCA.
            ppPipeline(sklearn.pipeline): Preprocessing pipeline.

        '''
        self.load_images()
        localtime = time.localtime()
        date = '{}.{}.{}_{}.{}'.format(
            localtime[2], localtime[1], localtime[0], localtime[3], localtime[4])
        self.sigmas = np.asarray(self.sigmas)
        sigmas_str = [str(s) for s in self.sigmas]
        sigmas_str = '_'.join(sigmas_str)
        
        features_dict = {}
        filter_timings_per_img_dict = {}
        for proj, img_list in self.exp_img.items():
            features = []
            filter_timings_per_img = []
        
            for countag, img in enumerate(img_list):
                
                start = time.perf_counter()
                if self.return_timing:
                    X, filter_timings = self.extract_features(img, proj)
                    filter_timings_per_img.append(filter_timings)
                else:
                    X = self.extract_features(img, proj)
                features.append(X)
                logger.debug('Extraction Timepoint {}: Done {}s'.format(
                    countag, round(time.perf_counter()-start,1)))
            
            features_dict[proj] = features
            if self.return_timing:
                filter_timings_per_img_dict[proj] = filter_timings_per_img
          
        features = []
        for proj, ftrs in features_dict.items():
            if len(features) == 0:
                features = ftrs
            else:
                features = [np.hstack((imgProj1, imgProj2)) for imgProj1, imgProj2 in zip(features, ftrs)]
        
        if self.return_timing:
            filter_timings_per_img = []
            for proj, tm in filter_timings_per_img_dict.items():
                if len(filter_timings_per_img) == 0:
                    filter_timings_per_img = tm
                else:
                    filter_timings_per_img = [np.hstack((tmProj1, tmProj2)) for tmProj1, tmProj2 in zip(filter_timings_per_img, tm)]
        
            
        if not self.predicting and not self.normOnFirst:
            if self.return_timing:
                pca_timing = self.fitAndApplyPreprocessing(np.concatenate(features), fitting=True)
            else:
                self.fitAndApplyPreprocessing(np.concatenate(features), fitting=True)
        
        
        for countag in range(len(features)):
            start = time.perf_counter()
            if countag == 0 and self.normOnFirst and not self.predicting:
                if self.return_timing:
                    features[countag], pca_timing = self.fitAndApplyPreprocessing(features[countag], fitting=True)
                    filter_timings_per_img[countag] = filter_timings_per_img[countag].tolist() + [pca_timing]
                else:
                    features[countag] = self.fitAndApplyPreprocessing(features[countag], fitting=True)
            else:
                features[countag] = self.fitAndApplyPreprocessing(features[countag], fitting=False)
              
            if self.saveFeatures:
                path_to_save = join(self.savePath, 'features_{}'.format(date))
                if not isdir(path_to_save):
                    makedirs(path_to_save)
                pickle.dump(features[countag], open(join(path_to_save, 'features_{}_{}_{}_imgTag_{:05}.sav'.format(
                    sigmas_str, self.target, date, countag)), 'wb'))
                logger.debug('Features saved')
                
                path_to_save_img_info = join(path_to_save, 'img_info.sav')
                with open(path_to_save_img_info, "wb") as outfile:
                    pickle.dump(self.img_info_dict, outfile)

            logger.debug('Preprocessing Timepoint {}: Done {}s'.format(
                countag, round(time.perf_counter()-start,1)))
        
        if self.predicting:
            if self.return_timing:
                return features, self.img_info_dict, filter_timings_per_img
            else:
                return features, self.img_info_dict
        else:
            if self.return_timing:
                return features, self.ppPipeline, self.img_info_dict, filter_timings_per_img
            else:
                return features, self.ppPipeline, self.img_info_dict


    def extract_features(self, image, proj):
        """
        Extract features from an image by applying filters chosen by the user.

        Args:
            image (numpy.ndarray): (w,h) array containing the pixel's values
                                    for an image.
            proj (str): current projection to select correct filter set.
        Returns:
            x (numpy.ndarray): (n,m) array where n is the number of pixels
                                    and m the number of features.

        """
        
        if self.return_timing:
            x, filter_timings = vigra_feat_extrctn(self.sigmas, self.deltaSigma, self.inOutScaleDiff, self.filters[proj], self.includeSeparateChannels, image, self.return_timing, self.lbprad)
        else:
            x = vigra_feat_extrctn(self.sigmas, self.deltaSigma, self.inOutScaleDiff, self.filters[proj], self.includeSeparateChannels, image, self.return_timing, self.lbprad)
        
        if self.return_timing:
            timing = {'filter': filter_timings, 'pca': None}
            return x, timing
        else:
            return x
        
    
    def fitAndApplyPreprocessing(self, features, fitting):
        """
        Fit and/or apply feature processing, consisting of scaling and/or PCA

        Args:
            features (numpy.ndarray): (n,m) array where n is the number of pixels and m the number of features.
            fitting (bool): whether the preprocessing pipeline needs to be fitted.

        Returns:
            features (numpy.ndarray): (n,m) array where n is the number of pixels and m the number of preprocessed features.

        """
        
        if fitting:
            logger.debug("Calculating normalization and/or PCA")
            if len(features.shape) < 2:
                features = np.expand_dims(features,0)
            if "PCA" in self.ppPipeline.named_steps:
                self.ppPipeline["PCA"].n_components=min(features.shape[1],self.ppPipeline["PCA"].n_components)
            start_time = time.perf_counter()
            if self.normOnFirst:
                features = self.ppPipeline.fit_transform(features)
            else:
                self.ppPipeline.fit(features)
            end_time = time.perf_counter()
    
        else:
            logger.debug("Applying normalization and/or PCA")
            features = self.ppPipeline.transform(features)
            
        features = features.astype(np.float32)
    
        logger.debug('Normalization/PCA done')

        if self.return_timing and fitting:
            if self.normOnFirst:
                return features, end_time-start_time
            else:
                return end_time-start_time
        else:
            return features

