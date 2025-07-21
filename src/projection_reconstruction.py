

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.ndimage import zoom, convolve, label
import numpy as np
from leapctype import *


import numpy as np
import matplotlib.pyplot as plt

leapct = tomographicModels()




import numpy as np
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
import pyvista as pv
import numpy as np
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.filters import window
import copy

def rotate(tau,vec,theta):
  return vec * np.cos(theta) + np.cross(tau,vec)*np.sin(theta) + tau * np.dot(tau,vec)*(1-np.cos(theta))


height = 2048
width = 4096
num_angles = 180
g = np.zeros([num_angles,height,width])

max_i = 15
max_j = 30
max_k = 15
i=7
j=0
k=7

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


voxel_size = 0.016


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
print(source_center)
center_volume = np.array([vol_pos_x,vol_pos_y,vol_pos_z])
detector_col_vec = np.array([1,0,0])
detector_row_vec = np.array([0,0,1])

center_source = -source_center
center_detector = center_source+source_detector
volume_center = -center_volume

rotation_axis = np.array([1,0,0])
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
leapct.set_diameterFOV(1000000)
f = leapct.allocateVolume()
g = a.astype(np.float32)


f[:] = 0.0
# # leapct.display(g)
# for j in range (max_j):
#     for i in range(max_i):
#         for k in range(max_k):
# Reconstruct the data
startTime = time.time()
leapct.SART(g,f,3,3)
time_elapsed = time.time()-startTime
print('Reconstruction Elapsed Time: ' + str(time_elapsed))
leapct.display(f)
mid_f = f[int(f.shape[0]/2),:,:]
norm_f = (mid_f - mid_f.min()) / (mid_f.max() - mid_f.min() + 2e-8)*255

im = Image.fromarray(norm_f.astype(np.uint8))
im.save(f"test_images/{i}_{j}_{k}.jpeg", cmap='gray')
print(f"Saved test_images/{i}_{j}_{k}.jpeg")
