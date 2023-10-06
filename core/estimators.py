import numpy as np
import cv2
# from typing import
from datetime import datetime
from campose import \
    estimate_marker_poses_in_camera,\
    estimate_camera_pose_in_base,\
    estimate_marker_poses_in_camera_weighted, \
    estimate_marker_poses_in_camera_weighted_extrinsic_guess,\
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
        self.last_valid_marker_in_camera_rvec = {}  # key - marker id: value - rvec
        self.last_valid_marker_in_camera_tvec = {}  # same : tvec

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
        frame, detected_marker_poses, largest_marker_size = estimate_marker_poses_in_camera_weighted(
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
            frame, detected_marker_poses, weights = estimate_marker_poses_in_camera(
                image,
                self.aruco_dict_type,
                self.marker_edge_len,
                self.matrix_coefficients,
                self.distortion_coefficients,
                return_frame=return_frame,
                # use_extrinsic_guess=True,
                # rvec=self.last_valid_marker_in_camera_rvec,
                # tvec=self.last_valid_marker_in_camera_rvec,
            )

            # print(detected_marker_poses)

            if detected_marker_poses == {}:
                return frame, np.array(1), timestamp, weights

            camera_poses_in_base = estimate_camera_pose_in_base_weighted(
                self.all_marker_poses,
                detected_marker_poses,
                weights
            )

            # print(type(camera_poses_in_base['weighted']))
            if camera_poses_in_base['weighted'].size == 0:
                return frame, np.array(1), timestamp, weights

            if return_frame:
                return frame, np.mean(list(camera_poses_in_base.values()), axis=0), timestamp, weights
            else:
                return np.array(0), np.mean(list(camera_poses_in_base.values()), axis=0), timestamp, weights

    def __call__(self, image, timestamp, return_frame=False):
        return self.weighted_inference(image, timestamp, return_frame)


class PoseMultiple(PoseSingle):
    def __init__(self,
                 n_frames: int,
                 aruco_dict_type: str,
                 n_markers: int,
                 marker_step: float,
                 marker_edge_len: float,
                 matrix_coefficients: np.ndarray,
                 distortion_coefficients: np.ndarray,
                 timestamp_threshold: float = 0.1):

        super(PoseMultiple, self).__init__(aruco_dict_type,
                                           n_markers,
                                           marker_step,
                                           marker_edge_len,
                                           matrix_coefficients,
                                           distortion_coefficients)

        self.n_frames = n_frames
        self.timestamp_threshold = timestamp_threshold

    def inference(self,
                  images: list[np.ndarray],
                  timestamps: list[datetime],
                  return_frame=False) -> tuple[list[tuple[np.ndarray, np.ndarray, datetime, float]] | None, datetime]:

        # return_frame=False is here for consistency of method signature.
        # this method will not return a frame.
        if (np.max(timestamps) - np.min(timestamps)) < self.timestamp_threshold:
            # preds = np.array(
            #     [super(PoseMultiple, self).inference(image, timestamp, return_frame)
            #      for image, timestamp in zip(images, timestamps)]
            # )
            preds = [super(PoseMultiple, self).inference(image, timestamp, return_frame)
                 for image, timestamp in zip(images, timestamps)]
        else:
            preds = None

        return preds, timestamps[0]

    def __call__(self, images, timestamps, return_frame=False):
        # return_frame=False is here for consistency of method signature.
        # this method will not return a frame.
        return self.inference(images, timestamps, False)
