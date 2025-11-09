import sys
import os
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from projection_io import image_importer, directory_images_importer, image_exporter

from projection_visualization import (
    viewer,
    fft_viewer,
    pixel_histogram_viewer,
    fft_pixel_histogram_viewer,
)
from projection_preprocessing import (
    generate_gain_map,
    generate_offset_map,
    generate_bad_pixel_map,
    projection_correction,
)
from projection_reconstruction import (
    Reconstruction
)


## Import reconstruction class
recon = Reconstruction("examples/example_import.json")

#Generates inner geometry
recon.generate_geometry()

#Reconstructs the object
recon.reconstruct(3, 3)

#Views

recon.view_initial_projection()
recon.view_reconstruction_volume()

#Optimize

recon.optimize()
recon.export_json("examples/updated_import.json")
