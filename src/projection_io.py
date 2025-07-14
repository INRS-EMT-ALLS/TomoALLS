
import numpy as np
import os
import shutil
from scipy.ndimage import zoom, convolve, label
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.filters import window
from os import listdir
from os.path import isfile, join

def image_exporter(image, path, dtype=np.uint16):

    image = np.clip(image - 1, 0, np.iinfo(dtype).max).astype(dtype)
    image.tofile(path)

def image_importer(path, height, width, dtype=np.uint16):
    return np.fromfile(path, dtype=dtype).reshape((height, width)).astype(np.float32)+1

def directory_images_importer(dir, height, width, dtype=np.uint16):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]

    images = np.zeros([len(files),height,width])

    for i, file in enumerate(files):
        images[i,:,:] = image_importer(join(dir,file),height,width)

    return images
