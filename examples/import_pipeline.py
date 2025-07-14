import sys
import os
from datetime import datetime

def extract_datetime(folder_name):
    timestamp = folder_name.replace(projection_folder_prefix, "")
    return datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from projection_io import image_importer, directory_images_importer,image_exporter
from projection_visualization import viewer, fft_viewer, pixel_histogram_viewer,fft_pixel_histogram_viewer
from projection_preprocessing import generate_gain_map, generate_offset_map, generate_bad_pixel_map,projection_correction


height,width = 2048,4096
max_angle = 360

export_path = "examples/corrected_projections/"
raw_projection_path = "examples/raw_projections/"
calibration_path = "Calibration/"
bad_pixel_map_file_name = "BPMap.raw"
gain_map_path_file_name = "GainMap.raw"
offset_map_path_file_name = "OffsetMap.raw"

bad_pixel_map_path = raw_projection_path+calibration_path+bad_pixel_map_file_name
gain_map_path = raw_projection_path+calibration_path+gain_map_path_file_name
offset_map_path = raw_projection_path+calibration_path+offset_map_path_file_name

projection_folder_prefix = "Capture_"
raw_frames_folder = "RawFrames/"


all_folders = os.listdir(raw_projection_path)


capture_folders = [f for f in all_folders if f.startswith(projection_folder_prefix)]

sorted_folders = sorted(capture_folders, key=extract_datetime)

# images = directory_images_importer("../test_images/projections/",2048,4096)
gain_map = generate_gain_map(gain_map_path,height,width)
offset_map = generate_offset_map(offset_map_path,height,width)
bad_pixel_map = generate_bad_pixel_map(bad_pixel_map_path,height,width)
increment = round(max_angle/len(sorted_folders))
i = 0
for folder in sorted_folders:
    print(i,folder)
    images = directory_images_importer(raw_projection_path+folder+"/"+raw_frames_folder,2048,4096)
    corrected,averaged = projection_correction(images[:,:,:],gain_map,offset_map,bad_pixel_map,228,2000,364,2078)
    image_exporter(averaged,export_path+str(i)+".raw")
    i+=increment
