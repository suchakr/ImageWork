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

# %%

def ThresholdWithOTSU(image, container, orig_img):
    thirdH = int(image.shape[0]/3)
    thirdW = int(image.shape[1]/3)
    roi = image[thirdH:2*thirdH, thirdW:2*thirdW] 
    
    #histr = cv2.calcHist([image],[0],None,[256],[0,256]) 
    #plt.plot(histr)
    #plt.show()

    histr = cv2.calcHist([roi],[0],None,[256],[0,256]) 
    #plt.plot(histr)
    #plt.show()
          
    img2 = image.copy()
    ot, roit = cv2.threshold(roi, 0, 255, cv2.THRESH_OTSU)    
    _, ti = cv2.threshold(img2, ot, 255, cv2.THRESH_BINARY_INV)
    writedebugimage(ti, container, 'raw-otsu')
    contours, _ = cv2.findContours(ti,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    
    mask_remove = orig_img.copy()
    mask_retain = orig_img.copy()
 
    # loop over the contours
    for c in contours:
        # if the contour is bad, draw it on the mask
        if is_good_contour_for_line_segmentation(c, ti)==True:
            r = cv2.boundingRect(c)
            cv2.rectangle(mask, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), color=0, thickness=-1)
            cv2.drawContours(mask_retain, [c], 0, color=(0,255,0), thickness=1)
        else:
            #r = cv2.boundingRect(c)
            cv2.drawContours(mask_remove, [c], 0, color=(0,0,255), thickness=-1)
    
    writedebugimage(mask_retain, container, 'retaining-this')
    writedebugimage(mask_remove, container, 'removing-this')

    # remove the contours from the image and show the resulting images
    writedebugimage(255-mask, container, 'contour-removal-mask')
    ## AARGH! someone please simplify this! X is the correct form, it has already been inverted so text is white.
    x = cv2.bitwise_and(ti,ti, mask=255-mask)
    return x
    
def is_good_contour_for_line_segmentation(c, image):
    ## boundingRect is x,y,w,h. Note w>4h and 5px limit is a postprocessing condition too. 
    ## minRect is ( center (x,y), (width, height), angle of rotation ) w=r[1][0], h=r[1][1]
    mask = np.zeros(image.shape,np.uint8)
    cv2.drawContours(mask,[c],0,255,-1)
    avg_intensity = cv2.mean(image, mask=mask)
    
    ## TODO this removes straight lines like "1" in 1049
    r = cv2.minAreaRect(c)
    if (r[1][0]<=5 or r[1][1]<=5 or (r[1][0]<=10 and r[1][0]>4*r[1][1])):
        #print('reporting bad contour rect, intensity', r, avg_intensity)
        return False
    
    ## remove very large contours 
    if (r[1][1]>image.shape[0]/2 or r[1][0]>image.shape[1]/2):
        print('reporting large bad contour rect', r, avg_intensity)
        return False
    
    return True

def ThresholdSauvola(img, container, orig_img):
    ## based on observation using these values of k,r work best (default k=0.2)
    thresh_sauvola = threshold_sauvola(img.copy(), window_size=31, k=0.1, r=45)
    binary_sauvola = img > thresh_sauvola
    image = binary_sauvola.astype('uint8')*255

    ## Sauvola is edge sensitive and prone to creating thick borders around image. Try remove them based on size
    contours, _ = cv2.findContours(255-image.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    mask = np.ones(img.shape[:2], dtype="uint8") * 255
 
    mask_remove = orig_img.copy()
    mask_retain = orig_img.copy()
 
    # loop over the contours
    for c in contours:
        # if the contour is bad, draw it on the mask
        if is_good_contour_for_line_segmentation(c, 255-image.copy())==True:
            r = cv2.boundingRect(c)
            cv2.rectangle(mask, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), color=0, thickness=-1)
            cv2.drawContours(mask_retain, [c], 0, color=(0,255,0), thickness=1)
        else:
            #r = cv2.boundingRect(c)
            cv2.drawContours(mask_remove, [c], 0, color=(0,0,255), thickness=-1)
    
    writedebugimage(mask_retain, container, 'retaining-this')
    writedebugimage(mask_remove, container, 'removing-this')


    # remove the contours from the image and show the resulting images
    writedebugimage(255-mask, container, 'contour-removal-mask')
    ## AARGH! someone please simplify this! X is the correct form, it has already been inverted so text is white.
    x = cv2.bitwise_and(255-image.copy(),255-mask, mask=255-mask)
    return x

## Analyse intensity histogram of gray scale image. Ideally it should be bimodal. Examine the top two peaks, if we
## have narrow well separated peaks that indicates good contrast. A broad peak that spans many intensity values
## indicates spread and low contrast
def IsLowContrast(image):
    hist = cv2.calcHist([image],[0],None,[256],[0,256]) 
    hist = hist/image.size
    
    x = np.reshape(hist, len(hist))
    window_size = 21
    hist_s = ss.savgol_filter(x, window_size, 5) 
    maxval = np.max(hist_s)
    peaks, info = ss.find_peaks(hist_s, prominence=maxval/10, width=5)
    widths = np.sort(info['widths'])
    print(widths)
    
    ## !!width is sensitive to window size. If window size is too high the graph will have wide curves and 
    ## bad approximation
    ## with current params width of good peaks seems to be about 10-20 range and spread peaks > 30.
    ## we expect bimodal distribution. Check if top two widths exceed 30
    if (len(widths)>=2):
        if (widths[-1]>30 or widths[-2]>30):
            return True
    
    return False
