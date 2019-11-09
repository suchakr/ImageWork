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
from imageeventgridtrigger.preprocess import PreProcessImage
from imageeventgridtrigger.utils import parse_input, upload_blob, upload_csv, download_image, writedebugimage, getlinewidth, findpeaks, FindMargins
from imageeventgridtrigger.thresholding import ThresholdSauvola, ThresholdWithOTSU, IsLowContrast
from imageeventgridtrigger.lines import ArrangeLineByLine, DoStraightLines, DoHorizontalSmudge, CreateLineContours, CreateMissingLines, EstimateLines
from imageeventgridtrigger.contours import EmitLinesAndRects, EmitLinesAndConvexHulls, GetContours, PreProcessBR3, PreProcessBoundingRects1, PreProcessBoundingRects2, PostProcessBoundingRects4


def main(event: func.EventGridEvent, context: func.Context):
    
    cwd = context.function_directory
    session_id = str(context.invocation_id)
    logging.info(f'session_id: {session_id}')

    if (event.event_type != "Microsoft.Storage.BlobCreated"):
        logging.info('Received delete event. Skip processing %s', event.subject)
        return
    
    container, filename, manuscriptid, file_url = parse_input(event)

    if (container == None):
        logging.info('Error parsing event. Skip processing %s', event.get_json())
        return
    if (filename.endswith(".jpg") != True):
        logging.info('Received non jpg file. Skip processing %s', event.get_json())
        return
    
    ## download the file locally.
    logging.info('Received container %s, filename %s, file_url %s', container, filename, file_url)
    orig_img = download_image(container, filename, session_id, manuscriptid)
    
    ## Steps 0-4 are preprocessing - nlmeans denoising, bilateral, medianblur, gray scale
    grey_img = PreProcessImage(orig_img, container)

    if (IsLowContrast(grey_img)):
        bw_img = ThresholdSauvola(grey_img, container, orig_img)
        writedebugimage(bw_img, container, 'sauvola-4')
    else:
        bw_img = ThresholdWithOTSU(grey_img, container, orig_img)
        writedebugimage(bw_img, container, 'otsu-4')


    # %%
    ## find approx left and right margins.
    left_margin, right_margin = FindMargins(bw_img)
    logging.info('left margin=%s, right margin=%s', left_margin, right_margin)

    ## find approx linewidth assuming straight lines. This does not account for skew but creates baseline for more precise algo
    ## later
    peakpositions, linewidth, straight_line_contours = DoStraightLines(bw_img, left_margin, right_margin) 
    num_straight_lines = len(peakpositions)

    logging.info('#straightlines=%s, linewidth estimate=%s', num_straight_lines, linewidth)


    # %%
    ## Try to account for line skew. If all goes well then line_contours has accurage peak info else peakpositions has peak info
    img2 = DoHorizontalSmudge(bw_img.copy())
    writedebugimage(img2, container, 'horizontal-smudge-45')

    ## get intensity profile of moving window starting from left margin
    min_peak_distance = int(0.75*linewidth)
    num_lines, num_lines_median, mean_linewidth, min_linewidth, std_linewidth, line_data = EstimateLines(img2.copy(), left_margin, right_margin, min_peak_distance, orig_img, container)
    logging.info('Approx1: #maxlines=%s, mean linewidth=%s, min linewidth=%s, stddev linewidth=%s, #medianlines=%s'.format(num_lines, mean_linewidth, min_linewidth, std_linewidth, num_lines_median))


    # %%
    line_contours = []

    #lines should not differ much from straight line estimate - if it does fall back to using straight lines!
    if (abs(num_lines-num_straight_lines)<=5):
        num_straight_lines=int(num_lines)
        line_contours = CreateLineContours(line_data, num_straight_lines, left_margin, right_margin) 
        
    else:
        print('Using straight line algo!')
        line_contours = straight_line_contours

    img3 = orig_img.copy()
    cv2.polylines(img3, line_contours, False, (255,0,0))
    writedebugimage(img3, container, 'line-contours-5')

    ## Generate approx line bottoms for front end to draw 
    approx_lines = np.array([np.percentile(np.array([x[0][1] for x in m]), 75) for m in line_contours])
    aw = getlinewidth(approx_lines)
    approx_line_bottoms = approx_lines.astype('uint')+int(aw/2)
    #print(approx_line_bottoms)
    lines_filename = 'lines-' + str(manuscriptid) + '.csv'
    upload_csv(approx_line_bottoms, container, lines_filename)
    logging.info('uploaded approx lines for FE')

    ## Line determination done, we are ready to run contouring to detect blocks of text
    contours = GetContours(bw_img)
    logging.info(f'total contours: {len(contours)}')

    ## hold contour, rect, hull, centroid in one place
    contourdatalist = []

    for (i, c) in enumerate(contours):
        # compute bounding box of contour
        rect = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        m = cv2.moments(c)
        if (m["m00"] != 0):
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            contourdatalist.append((c,rect,hull,(cx,cy)))
        else:
            ## TODO debug why these werent caught earlier? minAreaRect issue or external vs all issue?
            print('zero moment rect!', rect)
    
    ## split any rect spanning lines into smaller rects
    cdlist, s, c, d = PreProcessBoundingRects1(contourdatalist, linewidth, bw_img, orig_img, container)
    logging.info('Done PreprocessingBR1')

    ## remove very small rects or fully contained in another
    cdlist = PreProcessBoundingRects2(cdlist)
    logging.info('Done PreprocessingBR2')

    oi1 = orig_img.copy()
    bi1 = orig_img.copy()
    bi1[:,:,:] = 255

    EmitLinesAndRects(line_contours, cdlist, oi1, bi1, (255,0,0), (0,255,0))
    writedebugimage(oi1, container, 'lines-rects')
    writedebugimage(bi1, container, 'only-lines-rects')

    oi2 = orig_img.copy()
    bi2 = orig_img.copy()
    bi2[:,:,:] = 255
    EmitLinesAndConvexHulls(line_contours, cdlist, oi2, bi2, (255,0,0), (0,255,0))
    writedebugimage(oi2, container, 'lines-hulls')
    writedebugimage(bi2, container, 'only-lines-hulls')
    
    ## assign each rectangle to a line
    linerects = ArrangeLineByLine(cdlist, line_contours)
    logging.info('Done line assignments')

    ## merge certain rectangles after line numbering assigned
    logging.info(f'Calling BR3 with linewidth {linewidth}')
    updatedlinerects = []
    for _,lr in enumerate(linerects):
        mergedlr = PreProcessBR3(lr, linewidth)
        updatedlinerects.append(mergedlr)

    ## final cleanup to trim rects with skewed aspect ratio or lying outside margins
    updatedlinerects = PostProcessBoundingRects4(updatedlinerects, linewidth, left_margin, right_margin)
    logging.info('Done PreprocessingBR4')
        
    oi3 = orig_img.copy()
    bi3 = orig_img.copy()
    bi3[:,:,:]=255

    numchar = 0
    for (i, rc) in enumerate(updatedlinerects):
        for (j, r) in enumerate(rc):
            cv2.rectangle(oi3, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), ((i%2)*255,((i+1)%2)*255,0), 1)
            cv2.rectangle(bi3, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), ((i%2)*255,((i+1)%2)*255,0), 1)
            numchar = numchar+1
                        

    cv2.polylines(oi3, line_contours, False, (127,0,127))
    cv2.polylines(bi3, line_contours, False, (127,0,127))

    logging.info(f'Stats:Contours {len(contours)}, good rects {s}, compound rects {c}, chars {numchar}')
    writedebugimage(oi3, container, 'final')
    writedebugimage(bi3, container, 'final-rects')

    ## This prints the bounding rect coordinates. 2D array so csv print needs some more parsing.
    rects_filename = 'rectangles-' + str(manuscriptid) + '.csv'
    upload_csv(updatedlinerects, container, rects_filename)
    logging.debug('Uploaded bounding rects. Done!')

    return