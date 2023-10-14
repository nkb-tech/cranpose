import numpy as np
from copy import deepcopy
# from typing import
from datetime import datetime
from campose import \
    detect_markers, \
    estimate_camera_pose_in_markers, \
    estimate_marker_poses_in_camera, \
    estimate_camera_pose_in_base_old, \
    estimate_marker_poses_in_camera_weighted, \
    estimate_marker_poses_in_camera_extrinsic_guess, \
    estimate_camera_pose_in_base_weighted, \
    check_wrong_estimations, \
    estimate_camera_pose_in_base, \
    compute_marker_weights, compute_weighted_pose_estimation
from utils import create_marker_mtcs, draw_markers_on_frame, draw_weights_on_frame

from pykalman import KalmanFilter

class PoseSingle:
    def __init__(self,
                 aruco_dict_type: str,
                 camera_orientation: int, # -1 rear, 1 front
                 n_markers: int,
                 marker_step: float,
                 marker_edge_len: float,
                 matrix_coefficients: np.ndarray,
                 distortion_coefficients: np.ndarray,
                 apply_kf: bool = True,
                 transition_coef: float = 1,
                 observation_coef: float = 1,
                 invert_image: bool = False):

        self.aruco_dict_type = aruco_dict_type
        self.camera_orientation = camera_orientation
        self.marker_edge_len = marker_edge_len
        self.matrix_coefficients = matrix_coefficients
        self.distortion_coefficients = distortion_coefficients
        self.all_marker_poses = create_marker_mtcs(n_markers, marker_step)
        self.last_valid_marker_in_camera_rvec = {}  # key - marker id: value - rvec
        self.last_valid_marker_in_camera_tvec = {}  # same : tvec
        self.invert_image = invert_image
        self.apply_kf = apply_kf
        self.transition_coef = transition_coef
        self.observation_coef = observation_coef

        # the smaller the transition_coef the slower the filter
        # the smaller the observation_coef the faster the filter

        # Initial pose matrix
        self.camera_in_base = np.ma.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        # Kalman-Filter initialization
        self.filtered_state_mean = None
        self.filtered_state_covariance = None
        if apply_kf:
            self.camera_in_base.mask = True
            self.kf = self.init_kf()
        else:
            self.camera_in_base.mask = False
            self.kf = None

    def init_kf(self):
        # time step
        dt = 1/25

        # transition_matrix
        F = [[1, dt, 0.5 * dt * dt],
             [0, 1,  dt],
             [0, 0,  1]]

        # observation_matrix
        H = [1, 0, 0]

        # transition_covariance
        Q = np.array([
            [ 8.28688186e-02,  4.56069709e-02,  1.15608369e-03],
            [ 1.29460844e-03,  2.18108024e-02,  1.13942461e-03],
            [-2.02707405e-04,  8.94215211e-04,  7.30274151e-05]
        ])*self.transition_coef

        # observation_covariance
        R = np.array([[5000]])*self.observation_coef

        # initial_state_mean
        X0 = [0,
              0,
              0]

        # initial_state_covariance
        P0 = [[1e5, 0, 0],
              [0,   1, 0],
              [0,   0, 1]]

        self.filtered_state_mean = X0
        self.filtered_state_covariance = P0

        return KalmanFilter(
            transition_matrices=F,
            observation_matrices=H,
            transition_covariance=Q,
            observation_covariance=R,
            initial_state_mean=X0,
            initial_state_covariance=P0)

    # def inference(self, image: np.ndarray, timestamp: datetime, return_frame=False) -> tuple[
    #     np.ndarray,
    #     np.ndarray,
    #     datetime,
    #     float
    # ]:
    #     """
    #     Arguments
    #     image:
    #     timestamp:
    #     return_frame:
    #
    #     Returns:
    #     frame,
    #     camera_poses_in_base - array 4x4,
    #     timestamp,
    #     largest_marker_size
    #     """
    #     frame, detected_marker_poses, largest_marker_size = estimate_marker_poses_in_camera(
    #         image,
    #         self.aruco_dict_type,
    #         self.marker_edge_len,
    #         self.matrix_coefficients,
    #         self.distortion_coefficients,
    #         return_frame)
    #
    #     # print(detected_marker_poses)
    #
    #     if detected_marker_poses == {}:
    #         return frame, np.array(1), timestamp, largest_marker_size
    #
    #     camera_poses_in_base = estimate_camera_pose_in_base_old(self.all_marker_poses, detected_marker_poses)
    #
    #     # print(camera_poses_in_base)
    #
    #     if return_frame:
    #         return frame, np.mean(list(camera_poses_in_base.values()), axis=0), timestamp, largest_marker_size
    #     else:
    #         return np.array(0), np.mean(list(camera_poses_in_base.values()), axis=0), timestamp, largest_marker_size
    #
    # def weighted_inference(self, image: np.ndarray, timestamp: datetime, return_frame=False) -> tuple[
    #     np.ndarray,
    #     np.ndarray,
    #     datetime,
    #     float
    # ]:
    #     """
    #         Arguments
    #         image:
    #         timestamp:
    #         return_frame:
    #
    #         Returns:
    #         frame,
    #         camera_poses_in_base - array 4x4,
    #         timestamp,
    #         largest_marker_size
    #         """
    #     frame, detected_marker_poses, weights = estimate_marker_poses_in_camera_weighted(
    #         image,
    #         self.aruco_dict_type,
    #         self.marker_edge_len,
    #         self.matrix_coefficients,
    #         self.distortion_coefficients,
    #         return_frame=return_frame,
    #         # use_extrinsic_guess=True,
    #         # rvec=self.last_valid_marker_in_camera_rvec,
    #         # tvec=self.last_valid_marker_in_camera_rvec,
    #     )
    #
    #     # print(detected_marker_poses)
    #
    #     if detected_marker_poses == {}:
    #         return frame, np.array(1), timestamp, weights
    #
    #     camera_poses_in_base = estimate_camera_pose_in_base_weighted(
    #         self.all_marker_poses,
    #         detected_marker_poses,
    #         weights
    #     )
    #
    #     # print(type(camera_poses_in_base['weighted']))
    #     if camera_poses_in_base['weighted'].size == 0:
    #         return frame, np.array(1), timestamp, weights
    #
    #     if return_frame:
    #         return frame, np.mean(list(camera_poses_in_base.values()), axis=0), timestamp, weights
    #     else:
    #         return np.array(0), np.mean(list(camera_poses_in_base.values()), axis=0), timestamp, weights

    def weighted_inference_ext_guess(self, image: np.ndarray, return_frame=False) -> tuple[
        np.ndarray,
        np.ndarray,
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

        # Step 1. Detect markers
        corners, ids, rejected_img_points = detect_markers(
            frame=image,
            aruco_dict_type=self.aruco_dict_type,
            matrix_coefficients=self.matrix_coefficients,
            distortion_coefficients=self.distortion_coefficients,
            invert_image=self.invert_image)

        try:
            corners_dict = dict(zip(ids, corners))
        except TypeError:
            corners_dict = {}

        # Step 2. Estimate marker poses in camera
        mtcs, rvecs, tvecs = estimate_marker_poses_in_camera_extrinsic_guess(
            ids=ids, corners=corners,
            edge_len=self.marker_edge_len,
            matrix_coefficients=self.matrix_coefficients,
            distortion_coefficients=self.distortion_coefficients,
            use_extrinsic_guess=True,
            init_rvecs=self.last_valid_marker_in_camera_rvec,
            init_tvecs=self.last_valid_marker_in_camera_tvec)

        # Step 3. Estimate camera poses in markers
        camera_in_markers = estimate_camera_pose_in_markers(mtcs)  # here we're just turning matrices
        # print(camera_in_markers)

        # Step 4. Filter incorrect estimations
        rejected_marker_ids = check_wrong_estimations(camera_in_markers=camera_in_markers,
                                                      correct_view_dir=self.camera_orientation)

        for marker_id in rejected_marker_ids:
            corners_dict.pop(marker_id)
            mtcs.pop(marker_id)
            rvecs.pop(marker_id)
            tvecs.pop(marker_id)
            camera_in_markers.pop(marker_id)

        # Step 5. Estimate camera pose in base
        camera_in_base = estimate_camera_pose_in_base(self.all_marker_poses, camera_in_markers)
        # Step 6. Assign weights to markers
        weights = compute_marker_weights(corners_dict, *image.shape[:2])
        # Step 7. Compute weighted pose

        camera_in_base_weighted = compute_weighted_pose_estimation(camera_in_base, weights)

        # Step 8. Very important (!). Update init_rvecs and init_tvecs for correctly estimated marker poses
        self.last_valid_marker_in_camera_rvec.update(rvecs)
        self.last_valid_marker_in_camera_tvec.update(tvecs)

        if return_frame:
            res_image = deepcopy(image)
            for id, rvec, tvec in zip(weights.keys(), rvecs.values(), tvecs.values()):
                # print(rvec, tvec)
                draw_markers_on_frame(
                    frame=res_image,
                    corners=corners,
                    matrix_coefficients=self.matrix_coefficients,
                    distortion_coefficients=self.distortion_coefficients,
                    rvec=rvec,
                    tvec=tvec,
                    edge_len=self.marker_edge_len,
                )

            res_image = draw_weights_on_frame(
                res_image,
                corners_dict,
                weights,
            )

        else:
            res_image = np.array(0)

        if camera_in_base_weighted.size == 0:  # if marker is not detected
            # we set 'result' to be previous pose (basically to keep shape)
            camera_in_base_result = self.camera_in_base
            # and set mask = True, which omits all values from the matrix:
            if self.apply_kf:  # Maybe it is reasonable to mask the estimation only is we're using kf
                camera_in_base_result.mask = True

        else:  # if marker is detected
            # we update the result estimation
            camera_in_base_result = np.ma.asarray(camera_in_base_weighted)
            camera_in_base_result.mask = False
            # self.camera_in_base = camera_in_base_result

        # Apply Kalman-filter
        # IMPORTANT ! We apply Kalman-filter only to "x" coordinate of the measurements
        if self.apply_kf:
            self.filtered_state_mean, self.filtered_state_covariance = (
                self.kf.filter_update(
                    self.filtered_state_mean,
                    self.filtered_state_covariance,
                    observation=camera_in_base_result[0, 3])
            )
            camera_in_base_result.mask = False
            camera_in_base_result[0, 3] = self.filtered_state_mean[0]

        self.camera_in_base = camera_in_base_result

        return res_image, camera_in_base_result, weights

    def __call__(self, image, return_frame=False):
        return self.weighted_inference_ext_guess(image, return_frame)


# class PoseMultiple(PoseSingle):
#     def __init__(self,
#                  n_frames: int,
#                  aruco_dict_type: str,
#                  n_markers: int,
#                  marker_step: float,
#                  marker_edge_len: float,
#                  matrix_coefficients: np.ndarray,
#                  distortion_coefficients: np.ndarray,
#                  timestamp_threshold: float = 0.1):
#
#         super(PoseMultiple, self).__init__(aruco_dict_type,
#                                            n_markers,
#                                            marker_step,
#                                            marker_edge_len,
#                                            matrix_coefficients,
#                                            distortion_coefficients)
#
#         self.n_frames = n_frames
#         self.timestamp_threshold = timestamp_threshold
#
#     def inference(self,
#                   images: list[np.ndarray],
#                   timestamps: list[datetime],
#                   return_frame=False) -> tuple[list[tuple[np.ndarray, np.ndarray, datetime, float]] | None, datetime]:
#
#         # return_frame=False is here for consistency of method signature.
#         # this method will not return a frame.
#         if (np.max(timestamps) - np.min(timestamps)) < self.timestamp_threshold:
#             # preds = np.array(
#             #     [super(PoseMultiple, self).inference(image, timestamp, return_frame)
#             #      for image, timestamp in zip(images, timestamps)]
#             # )
#             preds = [super(PoseMultiple, self).inference(image, timestamp, return_frame)
#                      for image, timestamp in zip(images, timestamps)]
#         else:
#             preds = None
#
#         return preds, timestamps[0]
#
#     def __call__(self, images, timestamps, return_frame=False):
#         # return_frame=False is here for consistency of method signature.
#         # this method will not return a frame.
#         return self.inference(images, timestamps, False)

class PoseMultiple:
    def __init__(self,
                 estimators: list):

        self.estimators = estimators

    def inference(self,
                  images: list[np.ndarray],
                  ):

        """
        Takes list of images with order corresponding to estimators
        """

        preds = []
        for image, estimator in zip(images, self.estimators):

            _, pred, _ = estimator(image)
            if pred.shape:
                preds.append(pred)

        mean_pred = np.mean(preds, axis=0)

        return mean_pred

    def __call__(self, images):
        return self.inference(images)

