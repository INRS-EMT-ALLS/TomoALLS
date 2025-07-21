import napari
import numpy as np
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt

def viewer(images):
    viewer, image_layer = napari.imshow(images)
    napari.run()

def fft_viewer(images):
    if len(images.shape) == 3:
        base = np.abs(fftshift(fft2(images[0,:,:])))
        fft_stack = np.zeros([images.shape[0],base.shape[0],base.shape[1]])
        for i in range(images.shape[0]):
            fft_stack[i,:,:] = 10* np.log1p(np.abs(fftshift(fft2(images[i,:,:]))))
        viewer, image_layer = napari.imshow(fft_stack)
        napari.run()
    else:
        fft_image = np.log1p(np.abs(fftshift(fft2(images))))
        viewer, image_layer = napari.imshow(fft_image)
        napari.run()

def pixel_histogram_viewer(images):

    if len(images.shape) == 3:
        raw = images[0,:,:]
        plt.hist(raw.ravel(), bins=512, color='gray', edgecolor='black')
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()
        plt.figure(figsize=(12, 5))
    else:
        raw = images[:,:]
        plt.hist(raw.ravel(), bins=512, color='gray', edgecolor='black')
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()
        plt.figure(figsize=(12, 5))

def fft_pixel_histogram_viewer(images):
    if len(images.shape) == 3:
        raw = np.log1p(np.abs(fftshift(fft2(images[0,:,:]))))
        plt.hist(raw.ravel(), bins=512, color='gray', edgecolor='black')
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()
        plt.figure(figsize=(12, 5))
    else:
        raw = np.log1p(np.abs(fftshift(fft2(images[:,:]))))
        plt.hist(raw.ravel(), bins=512, color='gray', edgecolor='black')
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.show()
        plt.figure(figsize=(12, 5))
