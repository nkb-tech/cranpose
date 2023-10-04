from typing import Tuple, List

import numpy as np
import cv2
from datetime import datetime

from numpy import ndarray

from campose import estimate_marker_poses_in_camera, estimate_camera_pose_in_base
from posesingle import PoseSingle


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
        # self.single_estimator = PoseSingle(aruco_dict_type,
        #                                    n_markers,
        #                                    marker_step,
        #                                    marker_edge_len,
        #                                    matrix_coefficients,
        #                                    distortion_coefficients)

        self.n_frames = n_frames
        self.timestamp_threshold = timestamp_threshold

    def inference(self,
                  images: list[np.ndarray],
                  timestamps: list[datetime],
                  return_frame=False) -> tuple[list[tuple[ndarray, ndarray, datetime, float]] | None, datetime]:

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
