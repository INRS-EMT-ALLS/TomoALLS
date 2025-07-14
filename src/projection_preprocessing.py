
import numpy as np
import napari
import numpy as np
import os
import shutil
from scipy.ndimage import zoom, convolve, label, shift
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.filters import window
from os import listdir
from os.path import isfile, join
import math
from skimage.registration import phase_cross_correlation


from projection_io import image_importer, directory_images_importer
from projection_visualization import viewer, fft_viewer, pixel_histogram_viewer,fft_pixel_histogram_viewer

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

def get_maximum_coordinates(image):
    max = image.max()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == max:
                return i,j, max


def remove_streaks(image,max_iterations = 20):
    image_list = []
    if len(image.shape) > 2:
        image = image[0,:,:]
    center_mask_size = 201
    mean_mask_size = 11
    mean_convolve_size = 11

    kernel_size = 3
    kernel = np.zeros([kernel_size,kernel_size])
    kernel[:,:] = 1/(kernel_size*kernel_size-1)
    kernel[math.floor(kernel_size/2),math.floor(kernel_size/2)] = 0
    kernel_offset = math.floor(kernel_size/2)


    offset = math.floor(mean_mask_size/2)
    subimage_offset = math.floor(mean_convolve_size/2)


    fft_complex = fftshift(fft2(image))
    peak_mask = np.copy(fft_complex)
    center_x, center_y = peak_mask.shape

    center_x = math.floor(center_x/2)
    center_y = math.floor(center_y/2)

    min_x, max_x = center_x - math.floor(center_mask_size/2),center_x+math.floor(center_mask_size/2)
    min_y, max_y = center_y - math.floor(center_mask_size/2),center_y+math.floor(center_mask_size/2)

    peak_mask[min_x:max_x,min_y:max_y] = 0

    exit = False
    counter = 0


    while True:
        # image_list.append(np.copy(np.log1p(np.abs(fft_complex))))
        # image_list.append(np.copy(np.real(ifft2(ifftshift(fft_complex)))))
        counter+=1
        print(counter)
        x,y,max = get_maximum_coordinates(peak_mask)
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if i != 0 and j != 0:
                    if peak_mask[x+i,y+j] == 0:
                        exit = True
        if exit:
            break

        peak_mask[x,y] = 0
        mean = convolve(peak_mask[x-subimage_offset-kernel_offset:x+subimage_offset+kernel_offset,y-subimage_offset-kernel_offset:y+subimage_offset+kernel_offset],kernel)
        fft_complex[x-subimage_offset:x+subimage_offset,y-subimage_offset:y+subimage_offset]= mean[kernel_offset:-kernel_offset,kernel_offset:-kernel_offset]
        peak_mask[x-offset:x+offset,y-offset:y+offset] = 0
        if counter >= max_iterations:
            break


    reconstructed = np.real(ifft2(ifftshift(fft_complex)))
    # viz = np.zeros([len(image_list),fft_complex.shape[0],fft_complex.shape[1]])
    # for i in range(len(image_list)):
    #     viz[i,:,:] = image_list[i]
    # viewer(viz)
    return reconstructed

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

def generate_bad_pixel_map(path, height, width, dtype=np.uint16):

    if not isfile(path):
        bad_pixel_stack = directory_images_importer(path, height, width, dtype=np.uint16)
        bad_pixel_map = np.zeros(bad_pixel_stack.shape[1:]).astype(np.float32)
        for i in range(bad_pixel_stack.shape[0]):
            bad_pixel_map += bad_pixel_stack[i,:,:]
            i+=1
        return bad_pixel_map
    else:
        bad_pixel_map = image_importer(path, height, width, dtype=np.uint16)

    bad_pixel_map = np.round(-(normalize(bad_pixel_map))+1)
    return bad_pixel_map

def projection_correction(images,gain_map,offset_map,bad_pixel_map,min_x,max_x,min_y,max_y):
    corrected_frames = np.zeros(images.shape)
    averaged = np.zeros(images.shape[1:])
    kernel_size = 11
    kernel = np.zeros([kernel_size,kernel_size])
    kernel[:,:] = 1/(kernel_size*kernel_size)
    for i in range(images.shape[0]):
        print(i)
        frame = gain_correction(images[i,:,:],gain_map,offset_map)
        # normalized = normalize(frame)
        bp_corrected = bad_pixel_correction(frame,bad_pixel_map)
        removed_streaks = remove_streaks(bp_corrected)
        corrected_frames[i,:,:] = removed_streaks

    for i in range(images.shape[0]):
        phase_difference, error, diffphase = phase_cross_correlation(corrected_frames[0,min_x:max_x,min_y:max_y], corrected_frames[i,min_x:max_x,min_y:max_y],upsample_factor=10)
        print(phase_difference,error,diffphase)
        corrected_frames[i, :, :] = shift(
            corrected_frames[i, :, :], shift=(-phase_difference[0], -phase_difference[1]), order=1, mode='nearest'
        )

    for i in range(images.shape[0]):
        averaged+=corrected_frames[i, :, :]

    averaged = remove_streaks(averaged)
    return corrected_frames,averaged
