
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
    image = (image-offset_map+ 2e-8)/(gain_map-offset_map+ 2e-8)
    return image

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

def projection_correction(images,gain_map,offset_map,bad_pixel_map=None):
    corrected_frames = np.zeros(images.shape)
    averaged = np.zeros(images.shape[1:])
    for i in range(images.shape[0]):
        frame = gain_correction(images[i,:,:],gain_map,offset_map)
        normalized = normalize(frame)
        corrected_frames[i,:,:]+=normalized
        averaged+=corrected_frames[i,:,:]
    averaged/=images.shape[0]
    return averaged

images = directory_images_importer("../test_images/projections/",2048,4096)
comparison = directory_images_importer("../test_images/comparison/",2048,4096)
gain_map = generate_gain_map("../test_images/calibration/GainMap.raw",2048,4096)
offset_map = generate_gain_map("../test_images/calibration/OffsetMap.raw",2048,4096)
corrected = projection_correction(images,gain_map,offset_map)
viewer(corrected)
viewer(comparison)
