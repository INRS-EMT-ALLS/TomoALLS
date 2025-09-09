from scipy.ndimage import zoom, convolve, label, shift,median_filter
import scipy
import numpy as np
from skopt import gp_minimize
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
from matplotlib.widgets import Slider
from scipy.ndimage import zoom, convolve, label
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output
import math
import random
import napari
import numpy as np
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.filters import window
import copy
from leapctype import *


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from projection_io import image_importer, directory_images_importer,image_exporter, image_to_jpeg
from projection_visualization import viewer, fft_viewer, pixel_histogram_viewer,fft_pixel_histogram_viewer
from projection_preprocessing import generate_gain_map, generate_offset_map, generate_bad_pixel_map,projection_correction,normalize,clip_extremes

def angles_to_vec(angles):
    return np.array([np.cos(angles[0])*np.cos(angles[1]),np.cos(angles[0])*np.sin(angles[1]),np.sin(angles[0])])

def rotate(tau,vec,theta):
  return vec * np.cos(theta) + np.cross(tau,vec)*np.sin(theta) + tau * np.dot(tau,vec)*(1-np.cos(theta))

def project(v,u):
  return (np.dot(v,u)/np.dot(u,u))*u

#Volume and Detector sizes
#
#
height,width = 2048,4096
corrected_path ="examples/corrected_projections_complete_cropped"

tomogram_cropped_min_x =  2250
tomogram_cropped_max_x =  2700

tomogram_cropped_min_y = 255
tomogram_cropped_max_y = 1770

images = directory_images_importer(corrected_path,tomogram_cropped_max_y-tomogram_cropped_min_y,tomogram_cropped_max_x-tomogram_cropped_min_x)
image_to_jpeg(images[90,:,:],"test.jpeg")
tomogram_height = height
tomogram_width = width

tomogram_angles = images.shape[0]
initial_tomograms = images

# print(initial_tomograms.shape)
# viewer(initial_tomograms)
# Image.fromarray((255*normalize(initial_tomograms[0,:,:])).astype(np.uint8), mode='L').save("after_crop.jpeg")

pixelSize = 0.008

initial_voxel_volume_x_len = 256
initial_voxel_volume_y_len = 256
initial_voxel_volume_z_len = 256
initial_voxel_size = 0.032

z_min = 0
z_max = 256
x_min = 100
x_max = 256
y_min = 0
y_max = 256

magnified_voxel_size = 0.016

# GEOMETRY

source_to_detector = np.array([0,2360,0])
source_to_volume = np.array([0,775,0])
detector_col_angles = np.array([0,0])
detector_row_angles = np.array([-np.pi/2,0])
rotation_axis_angles = np.array([0,0])

# GEOMETRY CONVERSION #

tomogram_cropped_height = tomogram_cropped_max_y - tomogram_cropped_min_y
tomogram_cropped_width = tomogram_cropped_max_x - tomogram_cropped_min_x

cropped_offset_from_center_y = ((tomogram_cropped_max_y + tomogram_cropped_min_y)/2 -tomogram_height/2)
cropped_offset_from_center_x = ((tomogram_cropped_max_x + tomogram_cropped_min_x)/2 -tomogram_width/2)

magnified_voxel_volume_x_len = int((x_max-x_min)*initial_voxel_size/magnified_voxel_size)
magnified_voxel_volume_y_len = int((y_max-y_min)*initial_voxel_size/magnified_voxel_size)
magnified_voxel_volume_z_len = int((z_max-z_min)*initial_voxel_size/magnified_voxel_size)

magnified_vol_pos_x = ((x_min+x_max)/2 - initial_voxel_volume_x_len/2)*initial_voxel_size
magnified_vol_pos_y = ((y_min+y_max)/2 - initial_voxel_volume_y_len/2)*initial_voxel_size
magnified_vol_pos_z = ((z_min+z_max)/2 - initial_voxel_volume_z_len/2)*initial_voxel_size

numCols = tomogram_cropped_width
numRows = tomogram_cropped_height

numAngles = tomogram_angles
volume_to_magnified_area = np.array([magnified_vol_pos_x,magnified_vol_pos_y,magnified_vol_pos_z])

detector_col_vec = angles_to_vec(detector_col_angles)
detector_col_vec = detector_col_vec/np.linalg.norm(detector_col_vec)

detector_row_vec = angles_to_vec(detector_row_angles)
detector_row_vec = detector_row_vec - project(detector_row_vec,detector_col_vec)
detector_row_vec = detector_row_vec/np.linalg.norm(detector_row_vec)

cropped_area_offset = detector_row_vec*pixelSize*cropped_offset_from_center_y+detector_col_vec*pixelSize*cropped_offset_from_center_x

volume_to_source = -source_to_volume
volume_to_detector = volume_to_source+source_to_detector+cropped_area_offset
magnified_area_to_volume = -volume_to_magnified_area

rotation_axis = angles_to_vec(rotation_axis_angles)
rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)

sourcePositions = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
moduleCenters = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
colVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
rowVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)

T_phi = 2.0*np.pi/float(numAngles)

for n in range(numAngles):
    phi = n*T_phi
    sourcePositions[n,:] = rotate(rotation_axis,volume_to_source,phi)+magnified_area_to_volume
    moduleCenters[n,:] = rotate(rotation_axis,volume_to_detector,phi)+magnified_area_to_volume
    rowVectors[n,:] = rotate(rotation_axis,detector_row_vec,phi)
    colVectors[n,:] = rotate(rotation_axis,detector_col_vec,phi)


leapct = tomographicModels()

leapct.set_modularbeam(numAngles, numRows, numCols, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)
leapct.set_volume(magnified_voxel_volume_x_len,magnified_voxel_volume_y_len,magnified_voxel_volume_z_len,magnified_voxel_size,magnified_voxel_size)
f = leapct.allocateVolume()
g =  np.ascontiguousarray(initial_tomograms)
startTime = time.time()
leapct.SART(g,f,4,3)
print('Reconstruction Elapsed Time: ' + str(time.time()-startTime))
print(f.shape)
image_to_jpeg(f[f.shape[0]//2,:,:],"test.jpeg")

viewer(f)
