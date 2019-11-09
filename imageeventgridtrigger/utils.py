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


def parse_input(event):
    
    try:
        payload = event.get_json()
        file_url = payload["url"]
        logging.info(f'file_url {file_url}')
        split_path = file_url.split("/")
        file_fullname = split_path[-1]
        container = split_path[-2]
        logging.info('container %s, file_fullname %s, file_url %s', container, file_fullname, file_url)

        split_filename=file_fullname.split(".")
        filename = split_filename[0]
        extension = split_filename[1]

        manuscriptid = filename.split("-")[-1]

        logging.info(f'manuscriptid : {manuscriptid}')
        return container, file_fullname, manuscriptid, file_url
    except Exception as ex:
        logging.error(f'Exception parsing event: {ex}')

    return None, None, None, None

def upload_blob(filename, container, filepath, mode='rb'):
    try:
        logging.info('uploading file: %s from path %s to container %s', filename, filepath, container)
        connect_str = os.getenv('CONNECT_STR')
        logging.info('connect_str: %s', connect_str)
        
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        blob_client = blob_service_client.get_blob_client(container=container, blob=filename)

        ## Upload the created file
        with open(filepath, mode) as data:
            blob_client.upload_blob(data, overwrite=True)
    except Exception as ex:
        logging.error('Error uploading file %s to container %s: %s', filename, container, ex)

def writedebugimage(image, container, name='image_'):
    ## avoid race with two functions trying to write same file name. On upload to container obfuscate the extension
    ## so that debug images dont cause a processing loop.
    tmp = tempfile.gettempdir()
    localfilename = name+str(uuid.uuid4())+'.jpg'
    upload_file_path = os.path.join(tmp, localfilename)
    cv2.imwrite(upload_file_path, image)

    upload_blob(name+'.jpgbak', container, upload_file_path, 'rb')
    return

def upload_csv(line_bottoms, container, filename):
    tmp = tempfile.gettempdir()
    localfilename = filename+str(uuid.uuid4())+'.csv'
    upload_file_path = os.path.join(tmp, localfilename)
    np.savetxt(upload_file_path,line_bottoms, fmt='%s', delimiter=',')
    upload_blob(filename, container, upload_file_path, mode='rb')

def download_image(container, filename, session_id, manuscriptid):
    
    connect_str = os.getenv('CONNECT_STR')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container, blob=filename)

    # Download the blob to a local file
    tmp = tempfile.gettempdir()
    local_filename = session_id+filename
    download_file_path = os.path.join(tmp, local_filename)
    logging.info('Downloading blob %s to %s', filename, download_file_path)

    try:
        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        logging.info('downloaded blob succeeded')

        img = cv2.imread(download_file_path, cv2.IMREAD_ANYCOLOR)
        logging.info(f'Image size {img.size} and shape {img.shape}')
        ## Shape=(y,x)

        ## set height and width on image metadata - required by front end.
        logging.info('Creating metadata object')
        metadata = {'height':str(img.shape[0]), 'width':str(img.shape[1]), 'id':str(manuscriptid)}
        logging.info(f'Updating image metadata with height and width: {metadata}')
        blob_client.set_blob_metadata(metadata)
        logging.info('Updated image metadata with height and width')

        return img
    except Exception as ex:
        logging.error('Error downloading file %s: %s', filename, ex)
    return None

def getlinewidth(lp):
    ## multiple ways to calculate avg line width. Can take median of h values. Or if variance is high,
    ## can take diff of line positions in that vicinity which will be more accurate?
    lpdiff = np.diff(lp)
    return int(np.sum(lpdiff)/len(lpdiff))

def findpeaks(image, dist, horizontal_profile=True):
    if (horizontal_profile):
        intensity_hist = cv2.reduce(image, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)
    else:
        intensity_hist = cv2.reduce(image, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).T
    
    ## smoothen the intensity histogram using savgol to avoid noise
    x = np.reshape(intensity_hist, len(intensity_hist))
    window_size = 5
    if (horizontal_profile):
        window_size = int(image.shape[0]/22)  ## magic num!
    else:
        window_size = int(image.shape[1]/22)  ## magic num!
    if (window_size%2 == 0):
        window_size = window_size+1
    intensity_hist_s = ss.savgol_filter(x, window_size, 5) 
    intensity_hist_s = intensity_hist_s.astype(int)

    ## find peaks in smooth plot. Prominence is via observation - if a line has just
    ## a few letters it will create a small peak. We enforce roughly 1/10 of the line
    ## should be occupied with letters
    ## distance is to avoid quirks. We use 5 degree poly to smoothen so it leads to dual
    ## peaks at times close together. Distance specification is a heuristic to avoid
    ## getting too close peaks. Ideally we should skip that, but use this routine recursively
    ## by plugging in the median of linewidths back into the find_peaks as distance till
    ## convergence
    maxval = np.max(intensity_hist_s)
    peaks, _ = ss.find_peaks(intensity_hist_s, prominence=maxval/10, distance=dist)
    return intensity_hist, peaks, intensity_hist_s

def FindMargins(img):
    intensity_hist = cv2.reduce(img, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S).T
    ## smoothen the intensity histogram using savgol to avoid noise
    x = np.reshape(intensity_hist, len(intensity_hist))
    window_size = int(img.shape[1]/22)  ## magic num!
    if (window_size%2 == 0):
        window_size = window_size+1
    intensity_hist_s = ss.savgol_filter(x, window_size, 5) 
    intensity_hist_s = intensity_hist_s.astype(int)

    ## TODO reexamine prominence here - not same as for horizontal lines can be a stricter bound
    maxval = np.max(intensity_hist_s)
    peaks, _ = ss.find_peaks(intensity_hist_s, prominence=maxval/10, distance=10)
    
    """
    plt.figure(figsize=(10,5))
    plt.xticks(range(0, img.shape[1], int(img.shape[1]/10))) 
    plt.xlim([0, img.shape[1]])
    plt.plot(intensity_hist)
    plt.plot(intensity_hist_s, color='green')
    plt.plot(peaks, intensity_hist[peaks], "x")
    plt.show()
    """
    
    data = ss.peak_widths(intensity_hist_s, peaks, rel_height=1)
    #print('peak width data', data)
    widest_peak_index = np.argmax(data[0])
    #print('widest peak index', widest_peak_index)
    left_index = data[2][widest_peak_index]
    right_index = data[3][widest_peak_index]
    return int(left_index), int(right_index)

