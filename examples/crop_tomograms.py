import sys
import os
from datetime import datetime
from tqdm import tqdm
def extract_datetime(folder_name):
    timestamp = folder_name.replace(projection_folder_prefix, "")
    return datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from projection_io import image_importer, directory_images_importer,image_exporter
from projection_visualization import viewer, fft_viewer, pixel_histogram_viewer,fft_pixel_histogram_viewer
from projection_preprocessing import generate_gain_map, generate_offset_map, generate_bad_pixel_map,projection_correction

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
corrected_path ="examples/corrected_projections_complete_clipped"
images = directory_images_importer(corrected_path,2048,4096)

export_path = "examples/corrected_projections_complete_cropped/"

tomogram_cropped_min_x =  2250
tomogram_cropped_max_x =  2700

tomogram_cropped_min_y = 255
tomogram_cropped_max_y = 1770


for i in range(images.shape[0]):

    image_exporter(images[i,tomogram_cropped_min_y:tomogram_cropped_max_y,tomogram_cropped_min_x:tomogram_cropped_max_x],export_path+str(i*2)+".raw")
