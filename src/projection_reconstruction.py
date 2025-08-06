
import scipy
import numpy as np
from skopt import gp_minimize
from scipy import ndimage
import matplotlib.pyplot as plt
import glob
from matplotlib.widgets import Slider
from scipy.ndimage import zoom, convolve, label
import time
import numpy as np
from leapctype import *
leapct = tomographicModels()
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from projection_io import image_importer, directory_images_importer,image_exporter
from projection_visualization import viewer, fft_viewer, pixel_histogram_viewer,fft_pixel_histogram_viewer
from projection_preprocessing import generate_gain_map, generate_offset_map, generate_bad_pixel_map,projection_correction,normalize


sobel_x = np.array([[[-1,-2,-1],[0,0,0],[1,2,1]],[[-2,-3,2],[0,0,0],[2,3,2]],[[-1,-2,-1],[0,0,0],[1,2,1]]])
sobel_y = np.array([[[-1,0,1],[-2,0,2],[-1,0,1]],[[-2,0,2],[-3,0,3],[-2,0,2]],[[-1,0,1],[-2,0,2],[-1,0,1]]])
sobel_z = np.array([[[-1,-2,-1],[-2,-3,-2],[-1,-2,-1]],[[0,0,0],[0,0,0],[0,0,0]],[[1,2,1],[2,3,2],[1,2,1]]])

sobel_kernels = [sobel_x,sobel_y,sobel_z]

def cut(x):
    return (x-2**11)/(1+np.exp(-100*(x-2**11)))

def loss(vol):
    vol = normalize(vol)*(2**12)
    hist = scipy.ndimage.histogram(vol,bins = round(vol.max()),min=0,max=vol.max())
    hist[0]=0
    bin_edges = np.arange(start=0, stop=len(hist), step=1)
    res = hist*cut(bin_edges)
    return -np.sum(res)

def sharpness(volume):
    result = np.zeros(volume.shape)
    for kernel in sobel_kernels:
        result+= convolve(volume,kernel)**2
    return result

def rotate(tau,vec,theta):
  return vec * np.cos(theta) + np.cross(tau,vec)*np.sin(theta) + tau * np.dot(tau,vec)*(1-np.cos(theta))

height = 2048
width = 4096
g = np.zeros([180,2048,4096])

def black_box(params):

    param_x,param_y,param_z,param_row_x,param_row_y,param_row_z,param_col_x,param_col_y,param_col_z,param_rot_x,param_rot_y,param_rot_z = 0,0,0,0,0,0,0,0,0,0,0,0
    param_x,param_y,param_z,param_rot_y,param_rot_z = params

    max_i = 15
    max_j = 30
    max_k = 15
    i=12+param_x
    j=7+param_y
    k=7+param_z

    initial_volume_x_len = 1024
    initial_volume_y_len = 1024
    initial_volume_z_len = 1024
    initial_voxel_size = 0.008



    z_min = 0
    z_max = 1024

    x_min = 0
    x_max = 1024

    y_min = 0
    y_max = 1024


    voxel_size = 0.008


    vol_size_x = int(((x_max-x_min)*initial_voxel_size)/voxel_size)
    vol_size_y = int(((y_max-y_min)*initial_voxel_size)/voxel_size)
    vol_size_z = int(((z_max-z_min)*initial_voxel_size)/voxel_size)


    vol_pos_x = -((x_min+x_max)/2 - initial_volume_x_len/2)*initial_voxel_size
    vol_pos_y = -((y_min+y_max)/2 - initial_volume_y_len/2)*initial_voxel_size
    vol_pos_z = -((z_min+z_max)/2 - initial_volume_z_len/2)*initial_voxel_size

    numCols = 600
    numRows = 1530
    numAngles = 180
    pixelSize = 0.008

    offset_size = 0.1
    offset = (np.array([i-max_i/2,j-max_j/2,k-max_k/2]))* offset_size
    source_detector = np.array([int(452*0.008),2360,int(11*0.008)])

    source_center= np.array([0,775,0])+offset
    center_volume = np.array([vol_pos_x,vol_pos_y,vol_pos_z])
    detector_col_vec = np.array([1+param_col_x,0+param_col_y,0+param_col_z])
    detector_col_vec = detector_col_vec/np.linalg.norm(detector_col_vec)
    detector_row_vec = np.array([0+param_row_x,0+param_row_y,1+param_row_z])
    detector_row_vec = detector_row_vec/np.linalg.norm(detector_row_vec)

    center_source = -source_center
    center_detector = center_source+source_detector
    volume_center = -center_volume

    rotation_axis = np.array([1+param_rot_x,0+param_rot_y,0+param_rot_z])
    rotation_axis = rotation_axis/np.linalg.norm(rotation_axis)


    sourcePositions = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
    moduleCenters = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
    colVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)
    rowVectors = np.ascontiguousarray(np.zeros((numAngles,3)).astype(np.float32), dtype=np.float32)

    T_phi = 2.0*np.pi/float(numAngles)
    for n in range(numAngles):
        phi = n*T_phi
        rotation_matrix = np.array([[np.cos(phi),np.sin(phi)],[-np.sin(phi),np.cos(phi)]])
        sourcePositions[n,:] = rotate(rotation_axis,center_source,phi)+center_volume

        moduleCenters[n,:] = rotate(rotation_axis,center_detector,phi)+center_volume

        rowVectors[n,:] = rotate(rotation_axis,detector_row_vec,phi)

        colVectors[n,:] = rotate(rotation_axis,detector_col_vec,phi)

    leapct.set_modularbeam(numAngles, numRows, numCols, pixelSize, pixelSize, sourcePositions, moduleCenters, rowVectors, colVectors)
    leapct.set_volume(vol_size_x,vol_size_y,vol_size_z,voxel_size,voxel_size)
    f = leapct.allocateVolume()
    f[:] = 0.0
    leapct.SART(g,f,10,10)

1.6174963895406194, -0.10098227031500873, 0.03954907500173643, 0.013054101783921072, 0.00016173763630902838
0.01055437631387024, -0.03230519148056299, -0.008849894871733266, -0.0010193090723043634, 0.0008550679323881261
