import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from projection_io import directory_images_importer
from projection_visualization import viewer

height, width = 2048, 4096
corrected_path = "examples/corrected_projections_complete_clipped"
images = directory_images_importer(corrected_path, 2048, 4096)
viewer(images)
