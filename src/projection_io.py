
import numpy as np
import os
import shutil
from scipy.ndimage import zoom, convolve, label
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.filters import window
from os import listdir
from os.path import isfile, join
from PIL import Image


def normalize(image):
    image = (image-image.min()+ 2e-8)/(image.max()-image.min()+ 2e-8)
    return image

def image_exporter(image, path,dtype=np.uint16):
    image = normalize(image)
    max_val = np.iinfo(np.uint16).max  # 65535 for uint16
    scaled = (image * max_val).round().astype(np.uint16)
    scaled.tofile(path)
def image_to_jpeg(image, path,dtype=np.uint16):

    return Image.fromarray((255*normalize(image)).astype(np.uint8), mode='L').save(path)

def image_importer(path, height, width, dtype=np.uint16):
    return np.fromfile(path, dtype=dtype).reshape((height, width)).astype(np.float32)

def directory_images_importer(dir, height, width, dtype=np.uint16):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]

    images = np.zeros([len(files),height,width]).astype(np.float32)

    for i, file in enumerate(files):
        images[i,:,:] = image_importer(join(dir,file),height,width)

    return images
