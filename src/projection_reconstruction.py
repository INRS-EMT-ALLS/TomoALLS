import scipy
import numpy as np
from skopt import gp_minimize
from scipy import ndimage
import matplotlib.pyplot as plt
import glob
from matplotlib.widgets import Slider
from scipy.ndimage import zoom, convolve, label
import time
import numpy as np
# from leapctype import *

# leapct = tomographicModels()
import copy
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output
import math
import random
import napari
import numpy as np
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from skimage.filters import window
import copy


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
    normalize,
)

from scipy.ndimage import zoom, convolve, label, shift, median_filter
import scipy
import numpy as np
from skopt import gp_minimize
from scipy import ndimage
import matplotlib.pyplot as plt
import sys
from matplotlib.widgets import Slider
from scipy.ndimage import zoom, convolve, label
import time
import numpy as np
import copy
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import os

# from leapctype import *

import json


def angles_to_vec(angles):
    return np.array(
        [
            np.cos(angles[0]) * np.cos(angles[1]),
            np.cos(angles[0]) * np.sin(angles[1]),
            np.sin(angles[0]),
        ]
    )


def rotate(tau, vec, theta):
    return (
        vec * np.cos(theta)
        + np.cross(tau, vec) * np.sin(theta)
        + tau * np.dot(tau, vec) * (1 - np.cos(theta))
    )


def project(v, u):
    return (np.dot(v, u) / np.dot(u, u)) * u


class Reconstruction:
    def __init__(self) -> None:
        # Imported Values
        self.imported_values = False
        self.path = ""
        self.cropped = True
        self.height, self.width = 0, 0
        self.num_angles = 0
        self.tomogram_cropped_min_x = 0
        self.tomogram_cropped_max_x = 0
        self.tomogram_cropped_min_y = 0
        self.tomogram_cropped_max_y = 0
        self.images = 0
        self.pixel_size = 0
        self.initial_voxel_volume_x_len = 0
        self.initial_voxel_volume_y_len = 0
        self.initial_voxel_volume_z_len = 0
        self.initial_voxel_size = 0
        self.z_min = 0
        self.z_max = 0
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0
        self.magnified_voxel_size = 0.0
        self.initial_source_to_detector = np.array([0.0, 0.0, 0.0])
        self.initial_source_to_volume = np.array([0.0, 0.0, 0.0])
        self.initial_detector_col_angles = np.array([0.0, 0.0])
        self.initial_detector_row_angles = np.array([0.0, 0.0])
        self.initial_rotation_axis_angles = np.array([0.0, 0.0])
        self.source_to_detector_offset = np.array([0.0, 0.0, 0.0])
        self.source_to_volume_offset = np.array([0.0, 0.0, 0.0])
        self.detector_col_angles_offset = np.array([0.0, 0.0])
        self.detector_row_angles_offset = np.array([0.0, 0.0])
        self.rotation_axis_angles_offset = np.array([0.0, 0.0])
        self.source_to_detector = np.array([0.0, 0.0, 0.0])
        self.source_to_volume = np.array([0.0, 0.0, 0.0])
        self.detector_col_angles = np.array([0.0, 0.0])
        self.detector_row_angles = np.array([0.0, 0.0])
        self.rotation_axis_angles = np.array([0.0, 0.0])
        self.initial_projections = np.array([0.0, 0.0])

        self.num_angles = 0
        self.num_rows = 0
        self.num_cols = 0
        self.pixel_size = 0
        self.source_positions = np.array([0.0, 0.0])
        self.module_centers = np.array([0.0, 0.0])
        self.row_vectors = np.array([0.0, 0.0])
        self.col_vectors = np.array([0.0, 0.0])

        self.magnified_voxel_volume_x_len = 0
        self.magnified_voxel_volume_y_len = 0
        self.magnified_voxel_volume_z_len = 0

        self.reconstruction_volume = np.array([0.0, 0.0])
        self.reprojected_projections = np.array([0.0, 0.0])

        self.leapct = tomographicModels()

    def import_json(self, path):
        with open(path, "r") as f:
            # Parsing the JSON file into a Python dictionary
            params = json.load(f)
        print(params)
        self.imported_values = True

        self.path = params["path"]

        self.cropped = params["cropped"]

        self.height = params["initial_image_parameters"]["image_height"]
        self.width = params["initial_image_parameters"]["image_width"]
        self.pixel_size = params["initial_image_parameters"]["pixel_size"]
        self.num_angles = params["initial_image_parameters"]["num_angles"]

        self.tomogram_cropped_min_x = params["cropped_image_parameters"]["image_min_x"]
        self.tomogram_cropped_max_x = params["cropped_image_parameters"]["image_max_x"]
        self.tomogram_cropped_min_y = params["cropped_image_parameters"]["image_min_y"]
        self.tomogram_cropped_max_y = params["cropped_image_parameters"]["image_max_y"]

        self.initial_voxel_volume_x_len = params["initial_reconstruction_parameters"][
            "initial_voxel_volume_x_len"
        ]
        self.initial_voxel_volume_y_len = params["initial_reconstruction_parameters"][
            "initial_voxel_volume_y_len"
        ]
        self.initial_voxel_volume_z_len = params["initial_reconstruction_parameters"][
            "initial_voxel_volume_z_len"
        ]
        self.initial_voxel_size = params["initial_reconstruction_parameters"][
            "initial_voxel_size"
        ]

        self.x_min = params["magnified_reconstruction_parameters"][
            "initial_voxel_volume_x_len_min"
        ]
        self.x_max = params["magnified_reconstruction_parameters"][
            "initial_voxel_volume_x_len_max"
        ]
        self.y_min = params["magnified_reconstruction_parameters"][
            "initial_voxel_volume_y_len_min"
        ]
        self.y_max = params["magnified_reconstruction_parameters"][
            "initial_voxel_volume_y_len_max"
        ]
        self.z_min = params["magnified_reconstruction_parameters"][
            "initial_voxel_volume_z_len_min"
        ]
        self.z_max = params["magnified_reconstruction_parameters"][
            "initial_voxel_volume_z_len_max"
        ]
        self.magnified_voxel_size = params["magnified_reconstruction_parameters"][
            "magnified_voxel_size"
        ]

        self.initial_source_to_detector = np.array(
            params["initial_geometry_parameters"]["source_to_detector"]
        )
        self.initial_source_to_volume = np.array(
            params["initial_geometry_parameters"]["source_to_volume"]
        )
        self.initial_detector_col_angles = np.array(
            params["initial_geometry_parameters"]["detector_col_angles"]
        )
        self.initial_detector_row_angles = np.array(
            params["initial_geometry_parameters"]["detector_row_angles"]
        )
        self.initial_rotation_axis_angles = np.array(
            params["initial_geometry_parameters"]["rotation_axis_angles"]
        )

        self.source_to_detector_offset = np.array(
            params["geometry_parameters_offset"]["source_to_detector"]
        )
        self.source_to_volume_offset = np.array(
            params["geometry_parameters_offset"]["source_to_volume"]
        )
        self.detector_col_angles_offset = np.array(
            params["geometry_parameters_offset"]["detector_col_angles"]
        )
        self.detector_row_angles_offset = np.array(
            params["geometry_parameters_offset"]["detector_row_angles"]
        )
        self.rotation_axis_angles_offset = np.array(
            params["geometry_parameters_offset"]["rotation_axis_angles"]
        )

        self.source_to_detector = (
            self.initial_source_to_detector + self.source_to_detector_offset
        )
        self.source_to_volume = (
            self.initial_source_to_volume + self.source_to_volume_offset
        )
        self.detector_col_angles = (
            self.initial_detector_col_angles + self.detector_col_angles_offset
        )
        self.detector_row_angles = (
            self.initial_detector_row_angles + self.detector_row_angles_offset
        )
        self.rotation_axis_angles = (
            self.initial_rotation_axis_angles + self.rotation_axis_angles_offset
        )

        if self.cropped:
            self.initial_projections = directory_images_importer(
                self.path,
                self.tomogram_cropped_max_y - self.tomogram_cropped_min_y,
                self.tomogram_cropped_max_x - self.tomogram_cropped_min_x,
            )
        else:
            self.initial_projections = directory_images_importer(
                self.path,
                self.height,
                self.width,
            )
            self.initial_projections = self.initial_projections[
                self.tomogram_cropped_min_y : self.tomogram_cropped_max_y,
                self.tomogram_cropped_min_x : self.tomogram_cropped_max_x,
            ]

    def view(self):
        viewer(self.initial_projections)

    def generate_geometry(self):
        if self.imported_values:
            tomogram_cropped_height = (
                self.tomogram_cropped_max_y - self.tomogram_cropped_min_y
            )
            tomogram_cropped_width = (
                self.tomogram_cropped_max_x - self.tomogram_cropped_min_x
            )

            cropped_offset_from_center_y = (
                self.tomogram_cropped_max_y + self.tomogram_cropped_min_y
            ) / 2 - self.height / 2
            cropped_offset_from_center_x = (
                self.tomogram_cropped_max_x + self.tomogram_cropped_min_x
            ) / 2 - self.width / 2

            self.magnified_voxel_volume_x_len = int(
                (self.x_max - self.x_min)
                * self.initial_voxel_size
                / self.magnified_voxel_size
            )
            self.magnified_voxel_volume_y_len = int(
                (self.y_max - self.y_min)
                * self.initial_voxel_size
                / self.magnified_voxel_size
            )
            self.magnified_voxel_volume_z_len = int(
                (self.z_max - self.z_min)
                * self.initial_voxel_size
                / self.magnified_voxel_size
            )

            magnified_vol_pos_x = (
                (self.x_min + self.x_max) / 2 - self.initial_voxel_volume_x_len / 2
            ) * self.initial_voxel_size
            magnified_vol_pos_y = (
                (self.y_min + self.y_max) / 2 - self.initial_voxel_volume_y_len / 2
            ) * self.initial_voxel_size
            magnified_vol_pos_z = (
                (self.z_min + self.z_max) / 2 - self.initial_voxel_volume_z_len / 2
            ) * self.initial_voxel_size

            self.num_cols = tomogram_cropped_width
            self.num_rows = tomogram_cropped_height

            self.num_angles = self.num_angles

            volume_to_magnified_area = np.array(
                [magnified_vol_pos_x, magnified_vol_pos_y, magnified_vol_pos_z]
            )

            detector_col_vec = angles_to_vec(self.detector_col_angles)
            detector_col_vec = detector_col_vec / np.linalg.norm(detector_col_vec)

            detector_row_vec = angles_to_vec(self.detector_row_angles)
            detector_row_vec = detector_row_vec - project(
                detector_row_vec, detector_col_vec
            )
            detector_row_vec = detector_row_vec / np.linalg.norm(detector_row_vec)

            cropped_area_offset = (
                detector_row_vec * self.pixel_size * cropped_offset_from_center_y
                + detector_col_vec * self.pixel_size * cropped_offset_from_center_x
            )

            volume_to_source = -self.source_to_volume
            volume_to_detector = (
                volume_to_source + self.source_to_detector + cropped_area_offset
            )
            magnified_area_to_volume = -volume_to_magnified_area

            rotation_axis = angles_to_vec(self.rotation_axis_angles)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

            self.source_positions = np.ascontiguousarray(
                np.zeros((self.num_angles, 3)).astype(np.float32), dtype=np.float32
            )
            self.module_centers = np.ascontiguousarray(
                np.zeros((self.num_angles, 3)).astype(np.float32), dtype=np.float32
            )
            self.col_vectors = np.ascontiguousarray(
                np.zeros((self.num_angles, 3)).astype(np.float32), dtype=np.float32
            )
            self.row_vectors = np.ascontiguousarray(
                np.zeros((self.num_angles, 3)).astype(np.float32), dtype=np.float32
            )

            T_phi = 2.0 * np.pi / float(self.num_angles)

            for n in range(self.num_angles):
                phi = n * T_phi
                self.source_positions[n, :] = (
                    rotate(rotation_axis, volume_to_source, phi)
                    + magnified_area_to_volume
                )
                self.module_centers[n, :] = (
                    rotate(rotation_axis, volume_to_detector, phi)
                    + magnified_area_to_volume
                )
                self.row_vectors[n, :] = rotate(rotation_axis, detector_row_vec, phi)
                self.col_vectors[n, :] = rotate(rotation_axis, detector_col_vec, phi)

            self.leapct.set_modularbeam(
                self.num_angles,
                self.num_rows,
                self.num_cols,
                self.pixel_size,
                self.pixel_size,
                self.source_positions,
                self.module_centers,
                self.row_vectors,
                self.col_vectors,
            )
            self.leapct.set_volume(
                self.magnified_voxel_volume_x_len,
                self.magnified_voxel_volume_y_len,
                self.magnified_voxel_volume_z_len,
                self.magnified_voxel_size,
                self.magnified_voxel_size,
            )

            self.reconstruction_volume = leapct.allocateVolume()

    def reconstruct(self, num_iter=3, num_subset=3):
        startTime = time.time()
        self.leapct.SART(
            self.initial_projections, self.reconstruction_volume, num_iter, num_subset
        )
        print("Reconstruction Elapsed Time: " + str(time.time() - startTime))
        # print(f.shape)
        # image_to_jpeg(f[f.shape[0] // 2, :, :], "test.jpeg")

        # viewer(f)

    def reproject(self):
        self.leapct.project(
            self.reprojected_projections,
            self.reconstruction_volume,
        )

    def calculate_error(self):





recon = Reconstruction()
recon.import_json("src/tomogram_cropped.json")
recon.view()

# GEOMETRY CONVERSION #
