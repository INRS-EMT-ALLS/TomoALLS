import sys
import os
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from projection_io import directory_images_importer, image_exporter


def extract_datetime(folder_name):
    timestamp = folder_name.replace(projection_folder_prefix, "")
    return datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")


height, width = 2048, 4096
corrected_path = "examples/corrected_projections_complete_clipped"
images = directory_images_importer(corrected_path, 2048, 4096)

export_path = "examples/corrected_projections_complete_cropped/"

tomogram_cropped_min_x = 2250
tomogram_cropped_max_x = 2700

tomogram_cropped_min_y = 255
tomogram_cropped_max_y = 1770


for i in range(images.shape[0]):
    image_exporter(
        images[
            i,
            tomogram_cropped_min_y:tomogram_cropped_max_y,
            tomogram_cropped_min_x:tomogram_cropped_max_x,
        ],
        export_path + str(i * 2) + ".raw",
    )
