
import numpy as np
import napari
import numpy as np
import os
import shutil
from scipy.ndimage import zoom, convolve, label, shift,median_filter
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.filters import window
from os import listdir
from os.path import isfile, join
import math
from skimage.registration import phase_cross_correlation
import cv2
import time
from projection_io import image_importer, directory_images_importer
from projection_visualization import viewer, fft_viewer, pixel_histogram_viewer,fft_pixel_histogram_viewer
import concurrent.futures
from tqdm import tqdm

#temporary
# import warnings
# warnings.filterwarnings("ignore")

sobel_3d_x = np.array([[[-1,-2,-1],[0,0,0],[1,2,1]],[[-2,-3,2],[0,0,0],[2,3,2]],[[-1,-2,-1],[0,0,0],[1,2,1]]])
sobel_3d_y = np.array([[[-1,0,1],[-2,0,2],[-1,0,1]],[[-2,0,2],[-3,0,3],[-2,0,2]],[[-1,0,1],[-2,0,2],[-1,0,1]]])
sobel_3d_z = np.array([[[-1,-2,-1],[-2,-3,-2],[-1,-2,-1]],[[0,0,0],[0,0,0],[0,0,0]],[[1,2,1],[2,3,2],[1,2,1]]])

sobel_2d_x =  np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
sobel_2d_y =   np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

sobel_kernels_3d = [sobel_3d_x,sobel_3d_y,sobel_3d_z]

sobel_kernels_2d = [sobel_2d_x,sobel_2d_y]

def to_iter(images):
    result = []
    for i in range(images.shape[0]):
        result.append(images[i,:,:])
    return result

def list_stack(image,size):
    result = []
    for i in range(size):
        result.append(image)
    return result

def gradient(volume):
    result = np.zeros(volume.shape)
    for kernel in sobel_kernels_2d:
        result+= convolve(volume,kernel)**4
    return result

def get_maximum_coordinates(image):
    flattened = np.ravel(image)
    index = np.argmax(flattened)
    j = index % image.shape[1]
    i = math.floor(index/image.shape[1])
    max = flattened[index]
    return i,j, max

def remove_streaks(image,max_iterations = 20):

    fft_list = []
    image_list = []
    if len(image.shape) > 2:
        image = image[0,:,:]
    height,width = image.shape[0],image.shape[1]
    center_mask_size = 201
    distribution_range = 51
    replacement_range = 11
    mean_size = 25

    offset = math.floor(distribution_range/2)
    subimage_offset = math.floor(replacement_range/2)
    fft_complex = fftshift(fft2(image))
    peak_mask = np.copy(fft_complex)
    center_x, center_y = peak_mask.shape

    center_x = math.floor(center_x/2)
    center_y = math.floor(center_y/2)

    min_x, max_x = center_x - math.floor(center_mask_size/2),center_x+math.floor(center_mask_size/2)
    min_y, max_y = center_y - math.floor(center_mask_size/2),center_y+math.floor(center_mask_size/2)

    peak_mask[min_x:max_x,min_y:max_y] = 0

    counter = 0

    while True:
        # fft_list.append(np.copy(np.log1p(np.abs(fft_complex))))
        # image_list.append(np.copy(np.real(ifft2(ifftshift(fft_complex)))))
        counter+=1

        x,y,max = get_maximum_coordinates(peak_mask)
        peak_mask[x,y] = 0
        if x <= offset or height-offset<= x or y <= offset or (width-offset)<=y:
            break
        distribution_set = clip_extremes(fft_complex[x-offset:x+offset,y-offset:y+offset])
        std = np.std(distribution_set)
        mean =np.real(np.mean(distribution_set))
        replacement_area = np.random.normal(loc=mean,scale=std,size=[replacement_range,replacement_range])
        if fft_complex[x-subimage_offset:x+subimage_offset+1,y-subimage_offset:y+subimage_offset+1].shape != replacement_area.shape:
            break
        fft_complex[x-subimage_offset:x+subimage_offset+1,y-subimage_offset:y+subimage_offset+1] = replacement_area
        peak_mask[x-subimage_offset:x+subimage_offset+1,y-subimage_offset:y+subimage_offset+1] = replacement_area
        if counter >= max_iterations:
            break


    reconstructed = np.real(ifft2(ifftshift(fft_complex)))
    # viz = np.zeros([len(image_list),fft_complex.shape[0],fft_complex.shape[1]])
    # for i in range(len(image_list)):
    #     viz[i,:,:] = image_list[i]
    # viewer(viz)
    # viz = np.zeros([len(fft_list),fft_complex.shape[0],fft_complex.shape[1]])
    # for i in range(len(fft_list)):
    #     viz[i,:,:] = fft_list[i]
    # viewer(viz)
    return reconstructed

def gain_correction(image,gain_map,offset_map):
    corrected = (image-offset_map+ 2e-8)/(gain_map-offset_map+ 2e-8)
    return corrected

def bad_pixel_correction(image,bad_pixel_map,passes=2):
    uncorrected = image*bad_pixel_map
    size = 3

    current_pass = np.copy(uncorrected)
    new_bad_pixel_map = np.copy(bad_pixel_map)

    while size <=7:

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

def clip_extremes(image, sigma = 3):

    mean = np.mean(image)
    std = np.std(image)
    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std
    filtered = np.clip(image, lower_bound, upper_bound)

    return filtered

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

def generate_bad_pixel_map(path, height, width, dtype=np.uint16,kernel_size = 33,ratio=0.3):

    if not isfile(path):
        gain_map_stack = directory_images_importer(path, height, width, dtype=np.uint16)
        gain_map = np.zeros(gain_map_stack.shape[1:]).astype(np.float32)
        for i in range(gain_map_stack.shape[0]):
            gain_map += gain_map_stack[i,:,:]
            i+=1
        gain_map = gain_map/gain_map_stack.shape[0]
        average_map = cv2.blur(gain_map,(kernel_size,kernel_size))
        ratio_map = np.abs((gain_map/average_map)-1)
        result = np.where(ratio_map<ratio,1,0)
        return result
    else:
        gain_map = image_importer(path, height, width, dtype=np.uint16)
        average_map = cv2.blur(gain_map,(kernel_size,kernel_size))
        ratio_map = np.abs((gain_map/average_map)-1)
        result = np.where(ratio_map<ratio,1,0)
        return result

def single_image_correction(image,gain_map,offset_map,bad_pixel_map):
    frame = gain_correction(image,gain_map,offset_map)
    bp_corrected = bad_pixel_correction(frame,bad_pixel_map)
    result = remove_streaks(bp_corrected)
    return result

def projection_correction(images,gain_map,offset_map,bad_pixel_map,min_x,max_x,min_y,max_y):

    if len(images.shape) == 2:
        images = np.array([images])
    corrected_frames = np.zeros(images.shape)
    averaged = np.zeros(images.shape[1:])
    layers_list = to_iter(images)
    gain_list = list_stack(gain_map,images.shape[0])
    offeset_list = list_stack(offset_map,images.shape[0])
    badpixel_list = list_stack(bad_pixel_map,images.shape[0])
    t1 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers = 24) as exe:
        result = exe.map(single_image_correction, layers_list,gain_list,offeset_list,badpixel_list)
    corrected_frames = np.array(list(result))
    corrected_list = []
    for i in range(corrected_frames.shape[0]):
        # print(f"Aligning frame {i}")
        img2 = np.round(normalize(median_filter(clip_extremes(corrected_frames[0,min_x:max_x,min_y:max_y],0.1),3)))
        img1 = np.round(normalize(median_filter(clip_extremes(corrected_frames[i,min_x:max_x,min_y:max_y],0.1),3)))
        shift, response = cv2.phaseCorrelate(img1, img2)
        dx, dy = shift
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        if np.abs(dx) <=10 and np.abs(dy)<=10:
            corrected_list.append(cv2.warpAffine(corrected_frames[i,:,:], M, (corrected_frames[i,:,:].shape[1], corrected_frames[i,:,:].shape[0])))
    corrected_frames  = np.array(corrected_list)
    for i in range(corrected_frames.shape[0]):
        averaged+=corrected_frames[i, :, :]
    t2 = time.time()
    # print(f'Correction time: {t2-t1} seconds')
    averaged = -(remove_streaks(normalize(averaged)))
    averaged+=1
    # viewer(corrected_frames)
    # viewer(averaged)

    return averaged
