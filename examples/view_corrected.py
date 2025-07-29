import sys
import os
from datetime import datetime
from PIL import Image
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from projection_io import image_importer, directory_images_importer,image_exporter
from projection_visualization import viewer, fft_viewer, pixel_histogram_viewer,fft_pixel_histogram_viewer
from projection_preprocessing import generate_gain_map, generate_offset_map, generate_bad_pixel_map,projection_correction,normalize



height,width = 2048,4096
corrected_path ="examples/corrected_projections_complete_clip"
images = directory_images_importer(corrected_path,2048,4096)
viewer(images)
