
import numpy as np

import numpy as np
import os
import shutil
from scipy.ndimage import zoom, convolve, label
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.filters import window
from os import listdir
from os.path import isfile, join
import math


from projection_io import image_importer, directory_images_importer
from projection_visualization import viewer

def gain_correction(image,gain_map,offset_map):
    corrected = (image-offset_map+ 2e-8)/(gain_map-offset_map+ 2e-8)
    return corrected

def bad_pixel_correction(image,bad_pixel_map,passes=2):
    uncorrected = image*bad_pixel_map
    size = 3

    current_pass = np.copy(uncorrected)
    new_bad_pixel_map = np.copy(bad_pixel_map)

    while size <=9:

        kernel = np.zeros([size,size])
        kernel[:,:] = 1
        kernel[math.floor(size/2),math.floor(size/2)] = 0


        adj_bad_pixel = convolve(new_bad_pixel_map,kernel)
        adj_sum = convolve(current_pass,kernel)

        valid_mask = (new_bad_pixel_map == 0) & (adj_bad_pixel != 0)

        new_bad_pixel_map[valid_mask] = 1
        current_pass[valid_mask] = adj_sum[valid_mask] / adj_bad_pixel[valid_mask]

        size+=2

    return current_pass

def normalize(image):
    image = (image-image.min()+ 2e-8)/(image.max()-image.min()+ 2e-8)
    return image

def generate_gain_map(path, height, width, dtype=np.uint16):

    if not isfile(path):
        gain_stack = directory_images_importer(path, height, width, dtype=np.uint16)
        gain_map = np.zeros(gain_stack.shape[1:]).astype(np.float32)
        for i in range(gain_stack.shape[0]):
            gain_map += gain_stack[i,:,:]
            i+=1
        return gain_map
    else:
        gain_map = image_importer(path, height, width, dtype=np.uint16)

    return gain_map

def generate_offset_map(path, height, width, dtype=np.uint16):

    if not isfile(path):
        offset_stack = directory_images_importer(path, height, width, dtype=np.uint16)
        offset_map = np.zeros(offset_stack.shape[1:]).astype(np.float32)
        for i in range(offset_stack.shape[0]):
            offset_map += offset_stack[i,:,:]
            i+=1
        return offset_map
    else:
        offset_map = image_importer(path, height, width, dtype=np.uint16)

    return offset_map

def generate_bad_pixel_map(path, def projection_correction(images,gain_map,offset_map,bad_pixel_map):
    corrected_frames = np.zeros(images.shape)
    averaged = np.zeros(images.shape[1:])
    for i in range(images.shape[0]):
        print(i)
        frame = gain_correction(images[i,:,:],gain_map,offset_map)
        # normalized = normalize(frame)
        corrected_frames[i,:,:]=bad_pixel_correction(frame,bad_pixel_map)
        averaged+=corrected_frames[i,:,:]


    return corrected_frames,averaged
height, width, dtype=np.uint16):

    if not isfile(path):
        bad_pixel_stack = directory_images_importer(path, height, width, dtype=np.uint16)
        bad_pixel_map = np.zeros(bad_pixel_stack.shape[1:]).astype(np.float32)
        for i in range(bad_pixel_stack.shape[0]):
            bad_pixel_map += bad_pixel_stack[i,:,:]
            i+=1
        return bad_pixel_map
    else:
        bad_pixel_map = image_importer(path, height, width, dtype=np.uint16)

    bad_pixel_map = np.round(-(normalize(bad_pixel_map))+1)
    return bad_pixel_map

def projection_correction(images,gain_map,offset_map,bad_pixel_map):
    corrected_frames = np.zeros(images.shape)
    averaged = np.zeros(images.shape[1:])
    for i in range(images.shape[0]):
        print(i)
        frame = gain_correction(images[i,:,:],gain_map,offset_map)
        # normalized = normalize(frame)
        corrected_frames[i,:,:]=bad_pixel_correction(frame,bad_pixel_map)
        averaged+=corrected_frames[i,:,:]


    return corrected_frames,averaged

images = directory_images_importer("../test_images/projections/",2048,4096)
comparison = directory_images_importer("../test_images/comparison/",2048,4096)
gain_map = generate_gain_map("../test_images/calibration/GainMap.raw",2048,4096)
offset_map = generate_gain_map("../test_images/calibration/OffsetMap.raw",2048,4096)
bad_pixel_map = generate_bad_pixel_map("../test_images/calibration/BPMap.raw",2048,4096)


corrected,averaged = projection_correction(images,gain_map,offset_map,bad_pixel_map)
viewer(averaged)

viewer(corrected)
viewer(comparison)
