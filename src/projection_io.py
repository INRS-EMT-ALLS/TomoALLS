
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

def viewer(images):

    viewer, image_layer = napari.imshow(images)
    napari.run()

def image_importer(path, height, width, dtype=np.uint16):
    return np.fromfile(path, dtype=dtype).reshape((height, width)).astype(np.float32)

def directory_images_importer(dir, height, width, dtype=np.uint16):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]

    images = np.zeros([len(files),height,width])
    print(files)

    for i, file in enumerate(files):
        images[i,:,:] = image_importer(join(dir,file),height,width)

    return images
