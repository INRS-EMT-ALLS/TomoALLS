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

from projection_io import image_importer, directory_images_importer,image_exporter
from projection_visualization import viewer, fft_viewer, pixel_histogram_viewer,fft_pixel_histogram_viewer
from projection_preprocessing import generate_gain_map, generate_offset_map, generate_bad_pixel_map,projection_correction,normalize,clip_extremes


sobel_x = np.array([[[-1,-2,-1],[0,0,0],[1,2,1]],[[-2,-3,2],[0,0,0],[2,3,2]],[[-1,-2,-1],[0,0,0],[1,2,1]]])
sobel_y = np.array([[[-1,0,1],[-2,0,2],[-1,0,1]],[[-2,0,2],[-3,0,3],[-2,0,2]],[[-1,0,1],[-2,0,2],[-1,0,1]]])
sobel_z = np.array([[[-1,-2,-1],[-2,-3,-2],[-1,-2,-1]],[[0,0,0],[0,0,0],[0,0,0]],[[1,2,1],[2,3,2],[1,2,1]]])


sobel_kernels = [sobel_x,sobel_y,sobel_z]

def sharpness(volume):
    result = np.zeros(volume.shape)
    for kernel in sobel_kernels:
        result+= convolve(volume,kernel)**2
    return result

def angles_to_vec(angles):
    return np.array([np.cos(angles[0])*np.cos(angles[1]),np.cos(angles[0])*np.sin(angles[1]),np.sin(angles[0])])


def rotate(tau,vec,theta):
  return vec * np.cos(theta) + np.cross(tau,vec)*np.sin(theta) + tau * np.dot(tau,vec)*(1-np.cos(theta))

#Unchangeable reconstruction parameters

tomogram_height = 512
tomogram_width = 1024

tomogram_cropped_min_x =  0
tomogram_cropped_max_x =  1024

tomogram_cropped_min_y = 0
tomogram_cropped_max_y = 512

tomogram_angles = 180
pixelSize = 1

initial_voxel_volume_x_len = 256
initial_voxel_volume_y_len = 256
initial_voxel_volume_z_len = 256
initial_voxel_size = 1

z_min = 0
z_max = 256
x_min = 0
x_max = 256
y_min = 0
y_max = 256

magnified_voxel_size = 1


def project_initial_phantom(source_to_detector,source_to_volume,detector_col_angles,detector_row_angles,rotation_axis_angles):

    tomogram_cropped_height = tomogram_cropped_max_y - tomogram_cropped_min_y
    tomogram_cropped_width = tomogram_cropped_max_x - tomogram_cropped_min_x

    magnified_voxel_volume_x_len = int(((x_max-x_min)*initial_voxel_size)/initial_voxel_size)
    magnified_voxel_volume_y_len = int(((y_max-y_min)*initial_voxel_size)/initial_voxel_size)
    magnified_voxel_volume_z_len = int(((z_max-z_min)*initial_voxel_size)/initial_voxel_size)

    magnified_vol_pos_x = -((x_min+x_max)/2 - initial_voxel_volume_x_len/2)*initial_voxel_size
    magnified_vol_pos_y = -((y_min+y_max)/2 - initial_voxel_volume_y_len/2)*initial_voxel_size
    magnified_vol_pos_z = -((z_min+z_max)/2 - initial_voxel_volume_z_len/2)*initial_voxel_size

    numCols = tomogram_cropped_width
    numRows = tomogram_cropped_height

    numAngles = tomogram_angles

    volume_to_magnified_area = np.array([magnified_vol_pos_x,magnified_vol_pos_y,magnified_vol_pos_z])

    detector_col_vec = angles_to_vec(detector_col_angles)
    detector_col_vec = detector_col_vec/np.linalg.norm(detector_col_vec)

    detector_row_vec = angles_to_vec(detector_row_angles)

    detector_row_vec = detector_row_vec/np.linalg.norm(detector_row_vec)

    volume_to_source = -source_to_volume
    volume_to_detector = volume_to_source+source_to_detector
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
        sourcePositions[n,:] = rotate(rotation_axis,volume_to_source,phi)+volume_to_magnified_area
        moduleCenters[n,:] = rotate(rotation_axis,volume_to_detector,phi)+volume_to_magnified_area
        rowVectors[n,:] = rotate(rotation_axis,detector_row_vec,phi)
        colVectors[n,:] = rotate(rotation_axis,detector_col_vec,phi)


    leapct = tomographicModels()

    leapct.set_modularbeam(numAngles, numRows, numCols, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)
    leapct.set_volume(magnified_voxel_volume_x_len,magnified_voxel_volume_y_len,magnified_voxel_volume_z_len,magnified_voxel_size,magnified_voxel_size)
    f = leapct.allocateVolume()
    leapct.set_FORBILD(f,True)
    g = leapct.allocate_projections()
    leapct.project(g,f)
    initial_tomograms = g
    return g



def backproject(base_tomograms,source_to_detector,source_to_volume,detector_col_angles,detector_row_angles,rotation_axis_angles):

    tomogram_cropped_height = tomogram_cropped_max_y - tomogram_cropped_min_y
    tomogram_cropped_width = tomogram_cropped_max_x - tomogram_cropped_min_x

    magnified_voxel_volume_x_len = int(((x_max-x_min)*initial_voxel_size)/initial_voxel_size)
    magnified_voxel_volume_y_len = int(((y_max-y_min)*initial_voxel_size)/initial_voxel_size)
    magnified_voxel_volume_z_len = int(((z_max-z_min)*initial_voxel_size)/initial_voxel_size)

    magnified_vol_pos_x = -((x_min+x_max)/2 - initial_voxel_volume_x_len/2)*initial_voxel_size
    magnified_vol_pos_y = -((y_min+y_max)/2 - initial_voxel_volume_y_len/2)*initial_voxel_size
    magnified_vol_pos_z = -((z_min+z_max)/2 - initial_voxel_volume_z_len/2)*initial_voxel_size

    numCols = tomogram_cropped_width
    numRows = tomogram_cropped_height

    numAngles = base_tomograms.shape[0]

    volume_to_magnified_area = np.array([magnified_vol_pos_x,magnified_vol_pos_y,magnified_vol_pos_z])

    detector_col_vec = angles_to_vec(detector_col_angles)
    detector_col_vec = detector_col_vec/np.linalg.norm(detector_col_vec)

    detector_row_vec = angles_to_vec(detector_row_angles)
    detector_row_vec = detector_row_vec/np.linalg.norm(detector_row_vec)

    volume_to_source = -source_to_volume
    volume_to_detector = volume_to_source+source_to_detector
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
        sourcePositions[n,:] = rotate(rotation_axis,volume_to_source,phi)+volume_to_magnified_area
        moduleCenters[n,:] = rotate(rotation_axis,volume_to_detector,phi)+volume_to_magnified_area
        rowVectors[n,:] = rotate(rotation_axis,detector_row_vec,phi)
        colVectors[n,:] = rotate(rotation_axis,detector_col_vec,phi)


    leapct = tomographicModels()

    leapct.set_modularbeam(numAngles, numRows, numCols, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)
    leapct.set_volume(magnified_voxel_volume_x_len,magnified_voxel_volume_y_len,magnified_voxel_volume_z_len,magnified_voxel_size,magnified_voxel_size)
    f = leapct.allocateVolume()
    f[:] = 0
    g = base_tomograms
    leapct.SART(g,f,3,3)
    reconstruction = f
    reprojection = np.zeros(base_tomograms.shape).astype(np.float32)
    leapct.project(reprojection,f)
    loss = (normalize(np.abs(normalize(base_tomograms) - normalize(reprojection))))

    img1 = np.round(normalize(median_filter(clip_extremes(reprojection,0.1),3)))
    img2 = np.round(normalize(median_filter(clip_extremes(base_tomograms,0.1),3)))

    loss = ((normalize(base_tomograms) - normalize(reprojection))**2)*img2
    for i in range(loss.shape[0]):
        loss[i,:,:] = normalize(loss[i,:,:])

    loss2 = normalize(clip_extremes(normalize(np.abs(normalize(img1) - normalize(img2)))),)
    loss3 = np.sum(loss+loss2)


    return loss3


def backprojection_optimizer(parameters):
    param_1,param_2,param_3,param_4,param_5,param_6 = parameters
    param_7,param_8,param_9,param_10,param_11,param_12 = 0,0,0,0,0,0
    beta_source_to_detector = np.array([param_1,param_2,param_3])+source_to_detector
    beta_source_to_volume = np.array([param_4,param_5,param_6])+source_to_volume
    beta_detector_col_angles = np.array([param_7,param_8])+detector_col_angles
    beta_detector_row_angles = np.array([param_9,param_10])+detector_row_angles
    beta_rotation_axis_angles = np.array([param_11,param_12])+rotation_axis_angles



    print(f"------------------ Parameter attempt ------------------")
    loss = backproject(base_tomograms,beta_source_to_detector,beta_source_to_volume,beta_detector_col_angles,beta_detector_row_angles,beta_rotation_axis_angles)

    print("Source to detector: ", beta_source_to_detector)
    print("Source to volume: ", beta_source_to_volume)

    print("Detector row angles: ", beta_detector_row_angles)
    print("Detector row vector: ", angles_to_vec(beta_detector_row_angles))

    print("Detector column angles: ", beta_detector_col_angles)
    print("Detector column vector: ", angles_to_vec(beta_detector_col_angles))

    print("Rotation axis angles: ", beta_rotation_axis_angles)
    print("Rotation axis vector: ", angles_to_vec(beta_rotation_axis_angles))

    print("Loss:", loss)
    return loss

#REAL PARAMETERS
source_to_detector = np.array([0,1000,0])
source_to_volume = np.array([0,700,100])
detector_col_angles = np.array([0,0])
detector_row_angles = np.array([-np.pi/2,0])
rotation_axis_angles = np.array([np.pi/4,0])

base_tomograms = project_initial_phantom(source_to_detector,source_to_volume,detector_col_angles,detector_row_angles,rotation_axis_angles)



#PARAMETERS Optimization


#GUESS PARAMETERS
source_to_detector = np.array([0,1000,0])
source_to_volume = np.array([0,700,50])
detector_col_angles = np.array([0,0])
detector_row_angles = np.array([-np.pi/2,0])
rotation_axis_angles = np.array([np.pi/4,0])

params_range = [(-100,100),(-100,100),(-100,100),(-100,100),(-100,100),(-100,100),(-np.pi/2,np.pi/2),(0,2*np.pi),(-np.pi/2,np.pi/2),(0,2*np.pi),(-np.pi/2,np.pi/2),(0,2*np.pi)]
initial_params = [0,1000,0,0,700,50,0,0,-np.pi/2,0,np.pi/4,0]
res = gp_minimize(backprojection_optimizer,            # the function to minimize
                 [(-100,100),(-100,100),(-100,100),(-100,100),(-100,100),(-100,100)],
                  x0 = [0,0,0,0,0,0],
                  n_calls=100,         # the number of evaluations of f including at x0
                  n_random_starts=5,  # the number of random initial points
                  random_state=778)
print(res)
