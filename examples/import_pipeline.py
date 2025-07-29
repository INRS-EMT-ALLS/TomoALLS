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


height,width = 2048,4096
max_angle = 360

export_path = "examples/corrected_projections_complete/"
raw_projection_path = "examples/raw_projections/"
calibration_path = "Calibration/"
gain_map_path_file_name = "GainMap.raw"
offset_map_path_file_name = "OffsetMap.raw"

gain_map_path = raw_projection_path+calibration_path+gain_map_path_file_name
offset_map_path = raw_projection_path+calibration_path+offset_map_path_file_name

projection_folder_prefix = "Capture_"
raw_frames_folder = "RawFrames/"


all_folders = os.listdir(raw_projection_path)


capture_folders = [f for f in all_folders if f.startswith(projection_folder_prefix)]

sorted_folders = sorted(capture_folders, key=extract_datetime)

gain_map = generate_gain_map(gain_map_path,height,width)
offset_map = generate_offset_map(offset_map_path,height,width)
bad_pixel_map = generate_bad_pixel_map(gain_map_path,height,width)
increment = round(max_angle/len(sorted_folders))
start_index = 0
i = start_index*increment
file_value = start_index*increment
for i in tqdm(range(start_index,len(sorted_folders)),desc="Preprocess pipeline"):
# for folder in sorted_folders[start_index:]:
    folder = sorted_folders[i]
    # print(i,folder)
    images = directory_images_importer(raw_projection_path+folder+"/"+raw_frames_folder,2048,4096)
    averaged = projection_correction(images[:,:,:],gain_map,offset_map,bad_pixel_map,512,2080,1835,2800)
    image_exporter(averaged,export_path+str(file_value)+".raw")
    file_value+=increment
