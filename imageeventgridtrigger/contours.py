import json
import logging
import azure.functions as func
import numpy as np
import os
import cv2

from imageeventgridtrigger.utils import writedebugimage


def GetContours(image):
    img1 = image.copy()
    contours, _ = cv2.findContours(img1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)    
    return contours

def EmitLinesAndRects(line_contours, cdlist, oi, bi, rectcolor, linecolor):
    for (i, cd) in enumerate(cdlist):
        r = cd[1]
        cv2.rectangle(oi, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), rectcolor, 1)
        cv2.rectangle(bi, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), rectcolor, 1)
    cv2.polylines(oi, line_contours, False, linecolor)
    cv2.polylines(bi, line_contours, False, linecolor)
    
    
def EmitLinesAndConvexHulls(line_contours, cdlist, oi, bi, hullcolor, linecolor):
    for _,cd in enumerate(cdlist):
        hull = cd[2]
        cv2.drawContours(oi, [hull], 0, hullcolor, 1)
        cv2.drawContours(bi, [hull], 0, hullcolor, 1)
    
    cv2.polylines(oi, line_contours, False, linecolor)
    cv2.polylines(bi, line_contours, False, linecolor)

def ShowBoundingRects(cdlist, oi, title, container):
    for (j, cd) in enumerate(cdlist):
        r = cd[1]
        cv2.rectangle(oi, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), (255,0,255), 1)
    writedebugimage(oi, container, title)
        
def GetImageForConvexHull(hull, bw_img):
    # create a blank image of size w,h. br is (x,y,w,h)
    ### absolute coordinates cause an issue with hull rendering.
    mask = np.zeros(bw_img.shape, np.uint8)
    cv2.drawContours(mask,[hull],-1,(255,255,255),-1)
    mask = cv2.bitwise_and(bw_img, bw_img, mask=mask)
    return mask
    
def BreakMergedCharsViaHull(mask, cd, outputdata):
    # split image approx around mid point where intensity projection is weakest
    hist = cv2.reduce(mask, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    #plt.plot(hist)
    #plt.xlim([0, mask.shape[0]])
    #plt.show()
    
    max_y = np.max(hist)
    (x,y,w,h) = cd[1]
    # some sensible default
    currentmin = max_y
    currentminindex = y + int(h/2)
    
    # first y rows should be empty, hull starts around y'th row
    # look for minima in the 35-65% band in the middle
    #print('checking histogram range y, y+h', y, y+h)
    ## TODO fix indexing of for loop to take range (int(y+0.35h)) etc
    for i in range(y, y+h):
        if (i>0.35*h+y and i<0.65*h+y):
            ## if intensity is below average start tracking
            if (hist[i]<currentmin):
                currentmin = hist[i]
                currentminindex = i
                # print('updating minindex to', i)
    
    # print('x,y,currentminindex,h,w', x,y,currentminindex,h,w)
    
    # now redo contouring in each subpart and report the bounding rectangles.
    mask1 = mask.copy()
    mask2 = mask.copy()
    
    mask1[currentminindex:y+h, x:x+w]=0
    mask2[0:currentminindex, x:x+w]=0
    
    c1,_ = cv2.findContours(mask1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    c2,_ = cv2.findContours(mask2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for (_, c) in enumerate(c1):
        # compute bounding box of contour
        # print('upper contour', c)
        r = cv2.boundingRect(c)
        h = cv2.convexHull(c)
        m = cv2.moments(c)
        if (m["m00"] != 0):
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])

            outputdata.append((c,r,h,(cx,cy)))
            # print('upper broken rect', r)
    for (_, c) in enumerate(c2):
        # compute bounding box of contour
        # print('lower contour', c)
        r = cv2.boundingRect(c)
        h = cv2.convexHull(c)
        m = cv2.moments(c)
        if (m["m00"] != 0):
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])

            outputdata.append((c,r,h,(cx,cy)))
            # print('lower broken rect', r)
    

def PreProcessBoundingRects1(contourdata, linewidth, bw_img, orig_img, container):
    # contourdata has contour, rect, hull, centroid
    cdlist = contourdata.copy()
    simplecontourdata = []
    compoundcontourdata = []
    discardedcontourdata = []
    
    for (i, cd) in enumerate(cdlist):
        rect = cd[1]
        #h is rect[3]
        if (rect[3]/linewidth < 1.2):
            simplecontourdata.append(cd)
        elif (rect[3] <= 2.4*linewidth):
            compoundcontourdata.append(cd)
        else:
            print('discarding large contour', cd)
            discardedcontourdata.append(cd)
            
    s = len(simplecontourdata)
    c = len(compoundcontourdata)
    d = len(discardedcontourdata)
    
    oi = orig_img.copy()
    ShowBoundingRects(compoundcontourdata, oi, 'compound-bounding-rects', container)
    
    
    outputdata = []
    
    for (_, cd) in enumerate(compoundcontourdata):
        # Hull break method
        hull = cd[2]
        mask = GetImageForConvexHull(hull, bw_img)
        # showimage(mask)
        BreakMergedCharsViaHull(mask, cd, outputdata)
        #print('broken rects len', len(output))
        
    oi = orig_img.copy()
    ShowBoundingRects(outputdata, oi, 'compound-rects-broken', container)
    
    for (_, cd) in enumerate(outputdata):
        simplecontourdata.append(cd)
    return simplecontourdata, s, c, d

def contains(r1, r2):
    if (r1 != r2):
        return r1[0] <= r2[0] and r2[0]+r2[2] <= r1[0]+r1[2] and r1[1] <= r2[1] and r2[1]+r2[3] <= r1[1]+r1[3]
    return False
    
def PreProcessBoundingRects2(contourdatalist):
    areas = []
    cdlist = []
    for (_, smallcd) in enumerate(contourdatalist):
        smallr = smallcd[1]
        a = smallr[2]*smallr[3]
        areas.append(a)
        if (a>4):
            # check this is not contained fully in another other rect
            contained = False
            for (_, bigcd) in enumerate(contourdatalist):
                bigr = bigcd[1]
                if contains(bigr,smallr):
                    contained = True
                    break
            if (contained == False):
                cdlist.append(smallcd)
            #else:
            #    print('small contained in big', small, big)
        #else:
            #print('too small', small)
            
    ##np.savetxt("boundingrectsAreas.csv", areas, fmt="%s", delimiter=",")
    return cdlist

def ShouldMerge(r1, r2, linewidth):
    ## need additional criteria here. e.g. overlap with a narrow vertical or horizontal char should be ok.
    a1 = r1[2]*r1[3]
    a2 = r2[2]*r2[3]
    
    xOverlap = (r1[0]<=r2[0] and r1[0]+r1[2]>=r2[0]) or (r1[0]>=r2[0] and r1[0]<=r2[0]+r2[2])
    yOverlap = (r1[1]<=r2[1] and r1[1]+r1[3]>=r2[1]) or (r1[1]>=r2[1] and r1[1]<=r2[1]+r2[3])
    y_r1_CloseAbove_r2 = r2[1]>r1[1]+r1[3] and r2[1]<r1[1]+r1[3]+int(0.3*linewidth)
    y_r1_CloseBelow_r2 = r1[1]>r2[1]+r2[3] and r1[1]<r2[1]+r2[3]+int(0.3*linewidth)
    yCloseEnough = y_r1_CloseAbove_r2 or y_r1_CloseBelow_r2
    
    #if (xOverlap and yOverlap):
    #    print('r1 overlap r2', r1, r2)
    #if (xOverlap and yOverlap==False and y_r1_CloseAbove_r2):
    #    print('r1 close above r2', r1, r2)
    #if (xOverlap and yOverlap==False and y_r1_CloseBelow_r2):
    #    print('r1 close below r2', r1, r2)
    
    ## merge only large left r1 with small right r2, this simplifies bookkeeping at expense of some more iterations
    ## character height vs linewidth is a heuristic. If one of the characters has height less than 0.3 of line width
    ## consider merging. If the lines are quite apart then median char height could be used instead.
    if (xOverlap and (yOverlap or yCloseEnough)):
        #if (a1/a2 >2):
        if (r1[3]<=int(0.3*linewidth) or r2[3]<=int(0.3*linewidth)):
            ## on the same line x overlap is enough to merge. On the top line add a qualifier for y overlap
            #print('a1/a2>2', r1, r2)
            return True
        else:
            ## rectangles are closer in size. Proceed carefully. Merge in vertical is safer than merge in horizontal
            ## 1) if extent of x overlap is close to r2 width, merge r2 into r1. r2 is to right of r1. This is either
            ##    vertical stack case or matras that are narrow slivers to the right. 
            ## 2) if extent of y overlap is fully above or below the center line of r1. r2 is to right of r1. This is 
            ##    some parts of r1 are fragmented and to the right
            if (r2[0]>=r1[0]):
                overlap = min(r2[0]+r2[2], r1[0]+r1[2])-max(r1[0], r2[0])
                if (overlap > int(0.8*r2[2])):
                    print('r2 significant overlap with r1', r1, r2, overlap)
                    return True
                if ((r2[1]>r1[1]+int(0.5*r1[3])) and r2[1]<r1[1]+r1[3]):
                    ## r2 is between middle and bottom of r1  |=
                    print('r2 more than midway down r1', r1, r2)
                    return True
                if ((r1[1]>r2[1]+int(0.5*r2[3])) and r1[1]<r2[1]+r2[3]):
                    ## r1 is between middle and bottom of r2 =|
                    print('r1 more than midsway down r2', r1, r2)
                    return True
        #else:
        #    print('r1 not bigger than r2 to right, no merge', r1, r2)
    return False
    
def PreProcessBR3(linerects, linewidth):
    ## call the recursive routine few times. Because merging can create more overlaps instead of complicated
    ## logic just run this routine multiple times and it should result in convergence till count doesnt decrease
    dr = []
    currentLineRects = linerects.copy()
    
    for i in range(1,3):
        print('merge iteration', i)
        currentLineRects = PreProcessBoundingRects3(currentLineRects, linewidth, dr)
        
    return currentLineRects

def PreProcessBoundingRects3(linerects, linewidth, discardedrects=[]):
    ## find overlapped rectangles in a line and see if they can be merged
    ## There are two conditions:
    ## 1) There must be an overlap or touching of borders. Look in window of 5 rects before/after
    ## 2) The areas of rectangles must be dissimilar. Current magic ratio is 2
    
    mergeOccurred = False
    mergedlinerects = []
    
    for (i, r1) in enumerate(linerects):
            
        if (r1 in discardedrects):
            continue
        lookback=max(0,i-5)
        lookahead=min(i+5,len(linerects))
        for (j, r2) in enumerate(linerects[lookback:lookahead]):
            
                
            if (r1 == r2):
                continue
            if (r2 in discardedrects):
                continue
            
            #print('checking r1 and r2', r1, r2)
            if (ShouldMerge(r1, r2, linewidth)):
                #print('merging r1 and r2', r1, r2)
                x1 = min(r1[0], r2[0])
                y1 = min(r1[1], r2[1]) 
                x2 = max(r1[0]+r1[2], r2[0]+r2[2])
                y2 = max(r1[1]+r1[3], r2[1]+r2[3])
                w = x2-x1
                h = y2-y1
                r1 = (x1,y1,w,h)
                #print('merged **r1** and r2', r1, r2)
                mergeOccurred = True
                discardedrects.append(r2)
            #else:
             #   print('r1 r2 dont satisfy criteria for merge', r1, r2)
            
        # at end of inner loop, r1 didnt overlap, or it did and has new values. Either way add to output
        mergedlinerects.append(r1)
    
    ## debug
    #print('new series:', mergedlinerects)
    #print('discarded series', discardedrects)
    
    if (mergeOccurred):
        #print('atleast one merge happened, processing again')
        mergedlinerects = PreProcessBoundingRects3(mergedlinerects, linewidth, discardedrects)
        
    #print('merged rects', mergedlinerects)
    #print('discarded rects', discardedrects)
    return mergedlinerects


def PostProcessBoundingRects4(linerects, linewidth, left_margin, right_margin):
    ## after all processing remove rects that dont have good content, or are outside the left/right margins
    ## for now remove rects with very skewed aspect ratio
    updatedlinerects=[]
    
    for (i, rc) in enumerate(linerects):
        outputline = []
        for (j, r) in enumerate(rc):
            if (r[0]+r[2]<left_margin or r[0]>right_margin):
                print('removing rectange outside margins', r)
                continue
                
            if (r[3]>int(linewidth/5) or r[2]<4*r[3]):
                if (r[2]>5 and r[3]>5):
                    outputline.append(r)
                else:
                    print('removing very small rect', r)
            else:
                print('removing junk content rectange', r)
                
        updatedlinerects.append(outputline)
    return updatedlinerects


