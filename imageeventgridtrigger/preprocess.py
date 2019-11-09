import json
import logging
import azure.functions as func
import numpy as np
import os
import cv2
import scipy
import scipy.signal as ss
import skimage
from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
from skimage.restoration import denoise_nl_means, estimate_sigma
import uuid
import tempfile
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from imageeventgridtrigger.utils import writedebugimage


def PreProcessImage(image, container):

    ## TODO searchwindow size based on image size
    ## Step 0 Non local means denoising
    
    ## nonLocalMeans needs nvidia gpu
    #x = cv2.cuda.nonLocalMeans(image.copy(),h=7,search_window=35,block_size=7)
    #writedebugimage(x,'slownlmeans-0')
    
    ## scikit has slower version of nonlocalmeans but running time is very high even for 1mb image
    """
    sigma_est = np.mean(estimate_sigma(image.copy(), multichannel=False))
    print(f"estimated noise standard deviation = {sigma_est}")

    patch_kw = dict(patch_size=7,      # 5x5 patches
                    patch_distance=13, # 35x35 search area
                    multichannel=False)

    # slow algorithm
    image = denoise_nl_means(image.copy(), h=0.6 * sigma_est, fast_mode=False, **patch_kw)
    writedebugimage(image, 'slow-nlmeans-0')
    """    

    image = cv2.fastNlMeansDenoisingColored(image,None,h=7,hColor=7,templateWindowSize=7,searchWindowSize=35)
    writedebugimage(image, container, 'nlmeans-0')
       
    ## Step 1 bilateral with conservative params, but multiple times
    for i in range(100):
        image = cv2.bilateralFilter(image,2,10,10)
    writedebugimage(image, container, 'bilateral-1')
    
    ## Step 2 median blur with conservative params
    image = cv2.medianBlur(image,3)
    writedebugimage(image, container, 'medianblurred-2')
    
    ## Step 3 convert to grey scale. Decolor algo does much better job of contrast preservation vs plain vanilla
    ## cvtColor COLOR_BGR2GRAY method. The histogram shifts closer to bimodal
    image,_ = cv2.decolor(image.copy())
    writedebugimage(image, container, 'decolor-3')
    
    #hist = cv2.calcHist([image],[0],None,[256],[0,256]) 
    #plt.plot(hist)
    #plt.xlabel('decolor')
    #plt.show()
    
    ## Step 3 convert to grey scale
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #writedebugimage(image, 'grayimage-3')
    #hist = cv2.calcHist([image],[0],None,[256],[0,256]) 
    #plt.plot(hist)
    #plt.xlabel('regular')
    #plt.show()
   
    return image
