import numpy as np
import cv2
from datetime import datetime
from campose import \
    estimate_marker_poses_in_camera,\
    estimate_camera_pose_in_base,\
    estimate_marker_poses_in_camera_weighted,\
    estimate_camera_pose_in_base_weighted
from utils import create_marker_mtcs


class PoseSingle:
    def __init__(self,
                 aruco_dict_type: str,
                 n_markers: int,
                 marker_step: float,
                 marker_edge_len: float,
                 matrix_coefficients: np.ndarray,
                 distortion_coefficients: np.ndarray):

        self.aruco_dict_type = aruco_dict_type
        self.marker_edge_len = marker_edge_len
        self.matrix_coefficients = matrix_coefficients
        self.distortion_coefficients = distortion_coefficients
        self.all_marker_poses = create_marker_mtcs(n_markers, marker_step)

    def inference(self, image: np.ndarray, timestamp: datetime, return_frame=False) -> tuple[
        np.ndarray,
        np.ndarray,
        datetime,
        float
    ]:
        """
        Arguments
        image:
        timestamp:
        return_frame:

        Returns:
        frame,
        camera_poses_in_base - array 4x4,
        timestamp,
        largest_marker_size
        """
        frame, detected_marker_poses, largest_marker_size = estimate_marker_poses_in_camera(
            image,
            self.aruco_dict_type,
            self.marker_edge_len,
            self.matrix_coefficients,
            self.distortion_coefficients,
            return_frame)

        # print(detected_marker_poses)

        if detected_marker_poses == {}:
            return frame, np.array(1), timestamp, largest_marker_size

        camera_poses_in_base = estimate_camera_pose_in_base(self.all_marker_poses, detected_marker_poses)

        # print(camera_poses_in_base)

        if return_frame:
            return frame, np.mean(list(camera_poses_in_base.values()), axis=0), timestamp, largest_marker_size
        else:
            return np.array(0), np.mean(list(camera_poses_in_base.values()), axis=0), timestamp, largest_marker_size

    def weighted_inference(self, image: np.ndarray, timestamp: datetime, return_frame=False) -> tuple[
            np.ndarray,
            np.ndarray,
            datetime,
            float
        ]:
            """
            Arguments
            image:
            timestamp:
            return_frame:

            Returns:
            frame,
            camera_poses_in_base - array 4x4,
            timestamp,
            largest_marker_size
            """
            frame, detected_marker_poses, weights = estimate_marker_poses_in_camera_weighted(
                image,
                self.aruco_dict_type,
                self.marker_edge_len,
                self.matrix_coefficients,
                self.distortion_coefficients,
                return_frame)

            # print(detected_marker_poses)

            if detected_marker_poses == {}:
                return frame, np.array(1), timestamp, weights

            camera_poses_in_base = estimate_camera_pose_in_base_weighted(
                self.all_marker_poses,
                detected_marker_poses,
                weights
            )

            # print(camera_poses_in_base)
            # print(weights.keys())
            # print(weights.values())

            # print(camera_poses_in_base)

            if return_frame:
                return frame, np.mean(list(camera_poses_in_base.values()), axis=0), timestamp, weights
            else:
                return np.array(0), np.mean(list(camera_poses_in_base.values()), axis=0), timestamp, weights

    def __call__(self, image, timestamp, return_frame=False):
        return self.weighted_inference(image, timestamp, return_frame)
