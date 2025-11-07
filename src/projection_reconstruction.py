import numpy as np
from skopt import gp_minimize
import copy
from scipy.ndimage import median_filter
import json

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
    clip_extremes,
)

from leapctype import *


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
    def __init__(self, path) -> None:
        # Imported Values

        self.leapct = tomographicModels()

        self.best_error = 10000000000000

        self.best_sweep = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        with open(path, "r") as f:
            # Parsing the JSON file into a Python dictionary
            self.json_values = json.load(f)

        # print(params)
        self.imported_values = True

        self.path = self.json_values["path"]

        self.cropped = self.json_values["cropped"]

        self.height = self.json_values["initial_image_parameters"]["image_height"]
        self.width = self.json_values["initial_image_parameters"]["image_width"]
        self.pixel_size = self.json_values["initial_image_parameters"]["pixel_size"]
        self.num_angles = self.json_values["initial_image_parameters"]["num_angles"]

        self.tomogram_cropped_min_x = self.json_values["cropped_image_parameters"][
            "image_min_x"
        ]
        self.tomogram_cropped_max_x = self.json_values["cropped_image_parameters"][
            "image_max_x"
        ]
        self.tomogram_cropped_min_y = self.json_values["cropped_image_parameters"][
            "image_min_y"
        ]
        self.tomogram_cropped_max_y = self.json_values["cropped_image_parameters"][
            "image_max_y"
        ]

        self.initial_voxel_volume_x_len = self.json_values[
            "initial_reconstruction_parameters"
        ]["initial_voxel_volume_x_len"]
        self.initial_voxel_volume_y_len = self.json_values[
            "initial_reconstruction_parameters"
        ]["initial_voxel_volume_y_len"]
        self.initial_voxel_volume_z_len = self.json_values[
            "initial_reconstruction_parameters"
        ]["initial_voxel_volume_z_len"]
        self.initial_voxel_size = self.json_values["initial_reconstruction_parameters"][
            "initial_voxel_size"
        ]

        self.x_min = self.json_values["magnified_reconstruction_parameters"][
            "initial_voxel_volume_x_len_min"
        ]
        self.x_max = self.json_values["magnified_reconstruction_parameters"][
            "initial_voxel_volume_x_len_max"
        ]
        self.y_min = self.json_values["magnified_reconstruction_parameters"][
            "initial_voxel_volume_y_len_min"
        ]
        self.y_max = self.json_values["magnified_reconstruction_parameters"][
            "initial_voxel_volume_y_len_max"
        ]
        self.z_min = self.json_values["magnified_reconstruction_parameters"][
            "initial_voxel_volume_z_len_min"
        ]
        self.z_max = self.json_values["magnified_reconstruction_parameters"][
            "initial_voxel_volume_z_len_max"
        ]
        self.magnified_voxel_size = self.json_values[
            "magnified_reconstruction_parameters"
        ]["magnified_voxel_size"]

        self.initial_source_to_detector = np.array(
            self.json_values["initial_geometry_parameters"]["source_to_detector"]
        )
        self.initial_source_to_volume = np.array(
            self.json_values["initial_geometry_parameters"]["source_to_volume"]
        )
        self.initial_detector_col_angles = np.array(
            self.json_values["initial_geometry_parameters"]["detector_col_angles"]
        )
        self.initial_detector_row_angles = np.array(
            self.json_values["initial_geometry_parameters"]["detector_row_angles"]
        )
        self.initial_rotation_axis_angles = np.array(
            self.json_values["initial_geometry_parameters"]["rotation_axis_angles"]
        )

        self.source_to_detector_offset = np.array(
            self.json_values["geometry_parameters_offset"]["source_to_detector"]
        )
        self.source_to_volume_offset = np.array(
            self.json_values["geometry_parameters_offset"]["source_to_volume"]
        )
        self.detector_col_angles_offset = np.array(
            self.json_values["geometry_parameters_offset"]["detector_col_angles"]
        )
        self.detector_row_angles_offset = np.array(
            self.json_values["geometry_parameters_offset"]["detector_row_angles"]
        )
        self.rotation_axis_angles_offset = np.array(
            self.json_values["geometry_parameters_offset"]["rotation_axis_angles"]
        )

        self.source_to_detector_range = np.array(
            self.json_values["parameter_search_range"]["source_to_detector"]
        )
        self.source_to_volume_range = np.array(
            self.json_values["parameter_search_range"]["source_to_volume"]
        )
        self.detector_col_angles_range = np.array(
            self.json_values["parameter_search_range"]["detector_col_angles"]
        )
        self.detector_row_angles_range = np.array(
            self.json_values["parameter_search_range"]["detector_row_angles"]
        )
        self.rotation_axis_angles_range = np.array(
            self.json_values["parameter_search_range"]["rotation_axis_angles"]
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

    def export_json(self, path):
        self.json_values["geometry_parameters_offset"]["source_to_detector"] = (
            np.ndarray.tolist(self.source_to_detector_offset)
        )
        self.json_values["geometry_parameters_offset"]["source_to_volume"] = (
            np.ndarray.tolist(self.source_to_volume_offset)
        )

        self.json_values["geometry_parameters_offset"]["detector_col_angles"] = (
            np.ndarray.tolist(self.detector_col_angles_offset)
        )
        self.json_values["geometry_parameters_offset"]["detector_row_angles"] = (
            np.ndarray.tolist(self.detector_row_angles_offset)
        )
        self.json_values["geometry_parameters_offset"]["rotation_axis_angles"] = (
            np.ndarray.tolist(self.rotation_axis_angles_offset)
        )
        print(type(self.json_values))
        with open(path, "w") as output_file:
            # Parsing the JSON file into a Python dictionary
            json.dump(self.json_values, output_file, indent=3)  # def view(self):

    def generate_geometry(self):
        if self.imported_values:
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

            volume_to_source = -(self.source_to_volume)
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
        # startTime = time.time()

        self.leapct.SART(
            self.initial_projections, self.reconstruction_volume, num_iter, num_subset
        )
        # print("Reconstruction Elapsed Time: " + str(time.time() - startTime))
        # print(f.shape)
        # image_to_jpeg(f[f.shape[0] // 2, :, :], "test.jpeg")

        # viewer(f)

        pass

    def reproject(self):
        self.leapct.project(
            self.reprojected_projections,
            self.reconstruction_volume,
        )
        pass

    def calculate_error(self):
        self.initial_projection_shape = np.round(
            normalize(median_filter(clip_extremes(self.initial_projections, 0.1), 3))
        )
        reprojected_projection_shape = np.round(
            normalize(
                median_filter(clip_extremes(self.reprojected_projections, 0.1), 3)
            )
        )

        shape_diff = np.abs(
            normalize(self.initial_projection_shape)
            - normalize(reprojected_projection_shape)
        )

        shape_loss = np.sum(shape_diff)

        feature_diff = np.zeros(shape_loss.shape)
        for i in range(feature_diff.shape[0]):
            feature_diff[i, :, :] = np.abs(
                normalize(self.initial_projections[i, :, :])
                - normalize(self.reprojected_projections[i, :, :])
            )

        feature_loss = np.sum(feature_diff)

        return feature_loss + shape_loss

    def optimizer(self, params_list):
        self.update_offsets(params_list)
        self.generate_geometry()
        self.reconstruct(4, 3)
        self.reproject()

        error = self.calculate_error()

        if error < self.best_error:
            self.best_sweep = copy.deepcopy(params_list)
            self.best_error = error

        return error

    def update_offsets(self, params_list):
        self.source_to_detector_offset = np.array(
            [params_list[0], params_list[1], params_list[2]]
        )
        self.source_to_volume_offset = np.array(
            [params_list[3], params_list[4], params_list[5]]
        )
        self.detector_col_angles_offset = np.array([params_list[6], params_list[7]])
        self.detector_row_angles_offset = np.array([params_list[8], params_list[9]])
        self.rotation_axis_angles_offset = np.array([params_list[10], params_list[11]])

    def optimize(self, n_calls=10, n_random_starts=10, random_state=4):
        initial_param_list = [
            self.source_to_detector[0],
            self.source_to_detector[1],
            self.source_to_detector[2],
            self.source_to_volume[0],
            self.source_to_volume[1],
            self.source_to_volume[2],
            self.detector_col_angles[0],
            self.detector_col_angles[1],
            self.detector_row_angles[0],
            self.detector_row_angles[1],
            self.rotation_axis_angles[0],
            self.rotation_axis_angles[1],
        ]
        range_list = [
            tuple(self.source_to_detector_range[0]),
            tuple(self.source_to_detector_range[1]),
            tuple(self.source_to_detector_range[2]),
            tuple(self.source_to_volume_range[0]),
            tuple(self.source_to_volume_range[1]),
            tuple(self.source_to_volume_range[2]),
            tuple(self.detector_col_angles_range[0]),
            tuple(self.detector_col_angles_range[1]),
            tuple(self.detector_row_angles_range[0]),
            tuple(self.detector_row_angles_range[1]),
            tuple(self.rotation_axis_angles_range[0]),
            tuple(self.rotation_axis_angles_range[1]),
        ]

        result = gp_minimize(
            self.optimizer,  # the function to minimize
            range_list,
            x0=initial_param_list,
            n_calls=n_calls,  # the number of evaluations of f including at x0
            n_random_starts=n_random_starts,  # the number of random initial points
            random_state=random_state,
        )
