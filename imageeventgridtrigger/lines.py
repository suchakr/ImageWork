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
from imageeventgridtrigger.utils import writedebugimage, getlinewidth, findpeaks

def DoStraightLines(image, left_margin, right_margin):
    
    ## by default, specify distance of 10 pixels to avoid peaks that are too close
    intensity_hist, peaks, intensity_hist_s = findpeaks(image, 10)
    
    """
    plt.figure(figsize=(10,5))
    plt.xticks(range(0, img.shape[0], int(img.shape[0]/10))) 
    plt.xlim([0, img.shape[0]])
    plt.plot(intensity_hist)
    plt.plot(peaks, intensity_hist[peaks], "x")
    plt.show()
    """
    
    ## peaks is a simple list [y1 y2 y3...yn] where n=number of lines. Convert to contour from x=lm to rm
    contours = []
    for i, y in enumerate(peaks):
        contour = []
        for x in range(left_margin, right_margin, 1):
            contour.append([x,y])
        
        ## convert list to array  
        contour = np.array(contour).reshape((-1,1,2)).astype(np.int32)
        contours.append(contour)
        
    lw = getlinewidth(peaks)
    print('peaks {0}, linewidth {1}'.format(peaks, lw))
    return peaks, lw, contours

## smudge image from left to right to do more precise line determination
def DoHorizontalSmudge(image):
    for i in range(1,100,1):
        image = cv2.GaussianBlur(image,(5,5),sigmaX=1,sigmaY=0.01)
    return image

## This runs through vertical moving window of image from left margin to right margin, accumulating intensity profile 
## information - #peaks and linewidth of each slice. Returns back median of #peaks and average linewidth, std dev linewidth
def EstimateLines(image, left_margin, right_margin, min_peak_distance, orig_img, container):
    
    line_slices = [] 
    window = 250
    lw_sizes = []
    peak_sizes = []

    ## debug output    
    imgLines = orig_img.copy()
    
    for i in range(left_margin+1,right_margin):
        roi = image[:,max(left_margin,i-window):i]   
        intensity_hist, peaks, intensity_hist_s = findpeaks(roi, min_peak_distance)
        
        """if (i%500 == 0):
            plt.figure(figsize=(10,5))
            plt.xticks(range(0, image.shape[0], int(image.shape[0]/10))) 
            plt.xlim([0, image.shape[0]])

            plt.plot(intensity_hist)
            plt.plot(peaks, intensity_hist[peaks], "x")
            plt.show()
        """

        ## start recording vertical slice peaks, later used for threading
        peak_sizes.append(len(peaks))
        if (len(peaks)>1):
            lw = getlinewidth(peaks)
            lw_sizes.append(lw)
        else:
            print('### no peaks! y=', i)
        line_slices.append(peaks)
        
        for _, p in enumerate(peaks):
            ## draw a broad band 9px wide
            imgLines[p-4,i] = [255,0,0]
            imgLines[p-3,i] = [255,0,0]
            imgLines[p-2,i] = [255,0,0]
            imgLines[p-1,i] = [255,0,0]
            imgLines[p,i] = [255,0,0]
            imgLines[p+1,i] = [255,0,0]
            imgLines[p+2,i] = [255,0,0]
            imgLines[p+3,i] = [255,0,0]
            imgLines[p+4,i] = [255,0,0] 
        
    writedebugimage(imgLines, container, 'lines-with-smudge')
    logging.info(f'length of peak_sizes : {len(peak_sizes)}')
    return np.max(peak_sizes), np.median(peak_sizes), np.mean(lw_sizes), np.std(lw_sizes), np.min(lw_sizes), line_slices

## Given there exist maximum num_straight_lines, draw them across the whole image also in areas where there are no
## letters
def CreateMissingLines(num_straight_lines, curr_slice, prev_slice, horizontal_index, debug=0):
    if (debug == 1):
        print('previou peaks {0}'.format(prev_slice))
        print('current peaks {0}'.format(curr_slice))
            
    updated_slice = np.zeros(num_straight_lines, dtype='uint')

    ## iterate current slice and find each index is closest to which index of right slice and map them.
    for i, c in enumerate(curr_slice):
        d = 1000000 # some very high number
        best_match_index = -1
        for j, p in enumerate(prev_slice):
            if (abs(c-p)<d):
                d = abs(c-p)
                ## current best match for i in curr slice is j in right slice
                best_match_index = j
        if (debug==1):
            print('index {0} of curr is closest to index {1} of prev'.format(i, best_match_index))
        if (updated_slice[best_match_index] == 0):   
            updated_slice[best_match_index] = c
        else:
            ## TODO need better algo here?
            print('oops found two lines mapping to previous line, skipping index {0} for y={1}'.format(best_match_index, c))

    ## after all indexes in current slice are mapped to closest index in previous slice fill holes
    for i, y in enumerate(updated_slice):
        if (y==0):
            ## for first index no option but pick from neighbor, all others can utilize local linewidth
            if (i==0):
                updated_slice[i] = prev_slice[i]
            else:
                updated_slice[i] = updated_slice[i-1] + prev_slice[i] - prev_slice[i-1] 

    return updated_slice
       
## thread together line positions in each vertical slice. If a slice has less than num_straight_line peaks, extrapolate
## from neighboring slices
def CreateLineContours(line_data, num_straight_lines, left_margin, right_margin):
    ## there are rm-lm slices. Lines (i,j) records for column i, horizontal position j the y position of the intensity peak
    lines = np.zeros((right_margin-left_margin,num_straight_lines), dtype="uint")
    ## going from left to right, find the first slice that has num_straight_lines peaks
    horizontal_index = 0
    horizontal_index_back = 0
    for i,p in enumerate(line_data):
        if (len(p) != num_straight_lines):
            continue
        print('starting threading at {0}'.format(left_margin+i))
        horizontal_index = i
        horizontal_index_back = i-1
        break
    
    while (horizontal_index_back >= 0):
        #print('working on index {0} relative to lm'.format(horizontal_index_back))
        curr_slice = line_data[horizontal_index_back]
        prev_slice = line_data[horizontal_index_back+1]
        updated_slice = CreateMissingLines(num_straight_lines, curr_slice, prev_slice, horizontal_index_back)
           
        line_data[horizontal_index_back] = updated_slice
        lines[horizontal_index_back] = updated_slice
        #print('updated slice for index {0} - {1}'.format(horizontal_index_back, updated_slice))
            
        horizontal_index_back = horizontal_index_back-1
    
    while (horizontal_index < len(line_data)):
        #print('working on index {0}'.format(horizontal_index+lm))
        curr_slice = line_data[horizontal_index]
        prev_slice = line_data[horizontal_index-1]
        updated_slice = CreateMissingLines(num_straight_lines, curr_slice, prev_slice, horizontal_index)
        
        line_data[horizontal_index] = updated_slice
        lines[horizontal_index] = updated_slice
        #print('updated slice for index {0} - {1}'.format(horizontal_index+lm, updated_slice))
            
        horizontal_index = horizontal_index+1
        
    ## at this point the lines 2D array is ready for horizontal threading. Visualize it
    ## there is an indexing bug where lines has a last row of all 0. Trim it
    lines = lines[:lines.shape[0]-1, :]
    horizontal_lines = lines.T
    
    ## contours is a list of np array
    contours = []
    for i, line in enumerate(horizontal_lines):
        n = len(line)
        contour = np.zeros((n,1,2), dtype="int")
        ## contour points are in (x,y) format!
        for x, z in enumerate(line):
            contour[x] = np.array([[x+left_margin,z]])
    
        contours.append(contour)
        
    return contours     

## assign each element of cdlist (contour, rect, hull, centroid) to closest line contour. Return each line sorted by x
def ArrangeLineByLine(cdlist, line_contours):
    
    unassigned = cdlist.copy()
    charlines = [[] for x in range(0,len(line_contours))]
    
    for (i, cd) in enumerate(unassigned):
        r = cd[1]
        centroid = cd[3]
        dist = np.zeros((len(line_contours),1))
        for (j, lc) in enumerate(line_contours):
            dist[j] = abs(cv2.pointPolygonTest(lc, centroid, True))
        closest_contour_index = np.argmin(dist)
        charlines[closest_contour_index].append(r)
    
    
    for (i, l) in enumerate(charlines):
        # append sorted by x position
        l.sort(key = lambda x: x[0]) 
        logging.debug(f'chars by line:, {i}, {len(l)}')
    
    return charlines

