
import numpy as np
import napari
import numpy as np
import os
import shutil
from scipy.ndimage import zoom, convolve, label
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.filters import window
from os import listdir
from os.path import isfile, join
from projection_io import viewer, image_importer, directory_images_importer

def gain_correction(image,gain_map,offset_map):
    image = image + 1
    image = (image-offset_map)/(gain_map-offset_map)

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
