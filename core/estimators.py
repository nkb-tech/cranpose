from copy import deepcopy
from typing import Callable

import numpy as np
from campose import (check_wrong_estimations, compute_marker_weights,
                     compute_weighted_pose_estimation, detect_markers,
                     estimate_camera_pose_in_base,
                     estimate_camera_pose_in_markers,
                     estimate_marker_poses_in_camera_extrinsic_guess)
from pykalman import KalmanFilter
from utils import (create_marker_mtcs, draw_markers_on_frame,
                   draw_weights_on_frame, f_area, f_left_x_002, f_right_x_002)


class PoseSingle:
    """
    aruco_dict_type: str - тип Аруко маркеров (напр. DICT_4X4_50)
    camera_orientation: int, - ориентация камеры -1 rear, 1 front
    n_markers: int - максимальное число маркеров,
    marker_step: float - шаг между маркерами (м),
    marker_edge_len: float - длина стороны маркера (м),
    matrix_coefficients: np.ndarray - калибровка камеры intrinsic,
    distortion_coefficients: np.ndarray - коэффициенты дисторсии,
    apply_kf: bool = True - применять ли фильтр Калмана,
    transition_coef: float = 1 - коэффициент Калман фильтра transition_coef,
      чем больше, тем быстрее фильтр,
    observation_coef: float = 1 - коэффициент Калман фильтра observation_coef,
      чем больше, тем медленнее фильтр,
    size_weight_func: Callable - функция для взвешивания маркеров по их
      размеру (принимает float от 0 до 1)
    left_edge_weight_func: Callable - функция для взвешивания маркеров по
        положению относительно левого края кадра
        (принимает float от 0 до 1)
    right_edge_weight_func: Callable - функция для взвешивания маркеров по
        положению относительно правого края кадра
        (принимает float от 0 до 1)
    x_bias: float = 0 - сдвиг камеры по X относительно начала координат крана,
    invert_image: bool = False - инвертировать ли картинку
    debug: bool = False, влючение дебаг режима. в нем, эстимейтор будет
        возвращать и предсказание с калман фильтром
        и без него. Помимо этого, объект PoseMultiple, который инициализован
        хобя бы с одним объектом PoseSingle, у которого debug=True, будет
        возвращать не одну матрцу а кортеж из трех элементов:
        mean_pred, debug_preds, debug_preds_weights
    """

    def __init__(
        self,
        aruco_dict_type: str,
        camera_orientation: int,
        n_markers: int,
        marker_poses: float,
        marker_edge_len: float,
        matrix_coefficients: np.ndarray,
        distortion_coefficients: np.ndarray,
        apply_kf: bool = True,
        transition_coef: float = 1,
        observation_coef: float = 1,
        x_bias: float = 0,
        size_weight_func: Callable = f_area,
        left_edge_weight_func: Callable = f_left_x_002,
        right_edge_weight_func: Callable = f_right_x_002,
        invert_image: bool = False,
        debug: bool = False,
    ):
        self.aruco_dict_type = aruco_dict_type
        self.camera_orientation = camera_orientation
        self.marker_edge_len = marker_edge_len
        self.matrix_coefficients = matrix_coefficients
        self.distortion_coefficients = distortion_coefficients
        self.all_marker_poses = create_marker_mtcs(n_markers, marker_poses)
        self.last_valid_marker_in_camera_rvec = {
        }  # key - marker id: value - rvec
        self.last_valid_marker_in_camera_tvec = {}  # same : tvec
        self.invert_image = invert_image
        self.apply_kf = apply_kf
        self.transition_coef = transition_coef  # the smaller the slower the filter
        self.observation_coef = observation_coef  # the smaller the faster the filter
        self.x_bias = x_bias
        self.size_weight_func = size_weight_func
        self.left_edge_weight_func = left_edge_weight_func
        self.right_edge_weight_func = right_edge_weight_func
        self.debug = debug

        #  initial pose matrix
        self.camera_in_base = np.ma.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])

        # Kalman-Filter initialization
        self.filtered_state_mean = None
        self.filtered_state_covariance = None
        if apply_kf:
            self.camera_in_base.mask = True
            self.kf = self.init_kf()
        else:
            self.camera_in_base.mask = False
            self.kf = None

        if debug:
            self.camera_in_base_nofilter = np.ma.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]])

    def init_kf(self):
        # time step
        dt = 1 / 25

        # transition_matrix
        F = [[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]]

        # observation_matrix
        H = [1, 0, 0]

        # transition_covariance
        Q = np.array([[8.28688186e-02, 4.56069709e-02, 1.15608369e-03],
                      [1.29460844e-03, 2.18108024e-02, 1.13942461e-03],
                      [-2.02707405e-04, 8.94215211e-04, 7.30274151e-05]
                      ]) * self.transition_coef

        # observation_covariance
        R = np.array([[5000]]) * self.observation_coef

        # initial_state_mean
        X0 = [0, 0, 0]

        # initial_state_covariance
        P0 = [[1e5, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.filtered_state_mean = X0
        self.filtered_state_covariance = P0

        return KalmanFilter(transition_matrices=F,
                            observation_matrices=H,
                            transition_covariance=Q,
                            observation_covariance=R,
                            initial_state_mean=X0,
                            initial_state_covariance=P0)

    def weighted_inference_ext_guess(
        self,
        image: np.ndarray,
        return_frame: bool = False
    ) -> tuple[np.ndarray, np.ndarray, dict[float]] | tuple[
            np.ndarray, np.ndarray, np.ndarray, dict[float]]:
        """
        Takes image from camera and proceeds pose estimation 
        in base coordinate system.

        Arguments
            image: np.ndarray
            return_frame:

        Returns:
            tuple[
                frame: np.ndarray - frame with drawn marker edges and axes,
                camera_poses_in_base: np.ndarray - of shape (4, 4),
                weights: dict[np.ndarray] - weights of each marker
                ]
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
            ids=ids,
            corners=corners,
            edge_len=self.marker_edge_len,
            matrix_coefficients=self.matrix_coefficients,
            distortion_coefficients=self.distortion_coefficients,
            use_extrinsic_guess=True,
            init_rvecs=self.last_valid_marker_in_camera_rvec,
            init_tvecs=self.last_valid_marker_in_camera_tvec)

        # Step 3. Estimate camera poses in markers
        camera_in_markers = estimate_camera_pose_in_markers(
            mtcs)  # here we're just turning matrices
        # print(camera_in_markers)

        # Step 4. Filter incorrect estimations
        rejected_marker_ids = check_wrong_estimations(
            camera_in_markers=camera_in_markers,
            correct_view_dir=self.camera_orientation)

        for marker_id in rejected_marker_ids:
            corners_dict.pop(marker_id)
            mtcs.pop(marker_id)
            rvecs.pop(marker_id)
            tvecs.pop(marker_id)
            camera_in_markers.pop(marker_id)

        # Step 5. Estimate camera pose in base
        camera_in_base = estimate_camera_pose_in_base(self.all_marker_poses,
                                                      camera_in_markers)
        # Step 6. Assign weights to markers
        weights = compute_marker_weights(
            corners_dict,
            *image.shape[:2],
            weight_func_area=self.size_weight_func,
            weight_func_left_edge=self.left_edge_weight_func,
            weight_func_right_edge=self.right_edge_weight_func,
        )

        # Step 7. Compute weighted pose
        if sum(weights.values()) != 0:
            camera_in_base_weighted = compute_weighted_pose_estimation(
                camera_in_base, weights)
        else:
            camera_in_base_weighted = np.array([])

        # Step 8. Very important (!). 
        # Update init_rvecs and init_tvecs for correctly estimated marker poses
        self.last_valid_marker_in_camera_rvec.update(rvecs)
        self.last_valid_marker_in_camera_tvec.update(tvecs)

        if return_frame:
            res_image = deepcopy(image)
            for id, rvec, tvec in zip(weights.keys(), rvecs.values(),
                                      tvecs.values()):
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
            # Maybe it is reasonable to mask the estimation
            # only if we're using a kf
            if self.apply_kf:
                camera_in_base_result.mask = True

        else:  # if marker is detected
            # we update the result estimation
            camera_in_base_result = np.ma.asarray(camera_in_base_weighted)
            camera_in_base_result.mask = False
            
            # Subtract bias
            camera_in_base_result[0, 3] -= self.x_bias
            if self.debug:
                self.camera_in_base_nofilter = np.ma.asarray(
                    camera_in_base_weighted)
            # self.camera_in_base = camera_in_base_result

        # Apply Kalman-filter
        # IMPORTANT ! We apply Kalman-filter only to "x"
        # coordinate of the measurements
        if self.apply_kf:
            self.filtered_state_mean, self.filtered_state_covariance = (
                self.kf.filter_update(self.filtered_state_mean,
                                      self.filtered_state_covariance,
                                      observation=camera_in_base_result[0, 3]))
            camera_in_base_result.mask = False
            camera_in_base_result[0, 3] = self.filtered_state_mean[0]

        self.camera_in_base = camera_in_base_result

        if self.debug:
            return res_image, camera_in_base_result, \
                self.camera_in_base_nofilter, weights

        else:
            return res_image, camera_in_base_result, weights

    def __call__(self, image, return_frame=False):
        return self.weighted_inference_ext_guess(image, return_frame)


class PoseMultiple:
    """
    Class to perform pose estimation from multiple cameras
    estimators: list - objects of PoseSingle
    """
    def __init__(self, estimators: list):

        self.estimators = estimators

    def inference(
        self,
        images: list[np.ndarray],
    ) -> np.ndarray | tuple[np.ndarray, list, list]:
        """
        Takes a ndarray of images with order corresponding to estimators
        Arguments:
            images: list[np.ndarray] - images
        Return:
            mean_pred: list[np.ndarray] of shape (4, 4)

        X - coordinate of camera can be retrieved by mean_pred[0, 3] (!)
        """

        debug_preds_weights = []
        debug_preds = []
        debug = False
        preds = []
        for image, estimator in zip(images, self.estimators):

            if image is not None:

                if estimator.debug:
                    _, pred, debug_pred, weights = estimator(image)

                    debug = True
                    if pred.shape:
                        preds.append(pred)
                    debug_preds.append(pred)
                    debug_preds_weights.append(weights)

                else:
                    _, pred, weights = estimator(image)

                    if pred.shape:
                        preds.append(pred)

        if preds != []:
            mean_pred = np.mean(preds, axis=0)
        else:
            mean_pred = np.eye(4)

        if debug:
            return mean_pred, debug_preds, debug_preds_weights
        else:
            return mean_pred

    def __call__(self, images):
        return self.inference(images)


class PoseSpecial:
    """
    Определяет положение выбранного аруко маркера ОТНОСИТЕЛЬНО камеры.
    """

    def __init__(
        self,
        aruco_dict_type: str,
        marker_id: int,
        marker_edge_len: float,
        matrix_coefficients: np.ndarray,
        distortion_coefficients: np.ndarray,
        camera_orientation: int = 1,  # unused yet
        apply_kf: bool = False,
        transition_coef: float = 1,
        observation_coef: float = 1,
        z_bias: float = 0,
        invert_image: bool = False,
        debug: bool = False
    ) -> None:

        self.aruco_dict_type = aruco_dict_type
        self.camera_orientation = camera_orientation  # unused yet
        self.marker_id = marker_id
        self.marker_edge_len = marker_edge_len
        self.matrix_coefficients = matrix_coefficients
        self.distortion_coefficients = distortion_coefficients
        self.last_valid_marker_in_camera_rvec = {marker_id: None}
        self.last_valid_marker_in_camera_tvec = {marker_id: None}
        self.invert_image = invert_image
        self.apply_kf = apply_kf
        self.transition_coef = transition_coef  # the smaller the slower the filter
        self.observation_coef = observation_coef  # the smaller the faster the filter
        self.z_bias = z_bias
        self.debug = debug

        # initial pose matrix
        self.mtx = np.ma.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
        
        # Kalman-Filter initialization
        self.filtered_state_mean = None
        self.filtered_state_covariance = None
        if apply_kf:
            self.mtx.mask = True
            self.kf = self.init_kf()
        else:
            self.mtx.mask = False
            self.kf = None

        if self.debug:
            self.mtx_nofilter = np.ma.array([[1, 0, 0, 0],
                                             [0, 1, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]])

    def init_kf(self):
        # time step
        dt = 1 / 25

        # transition_matrix
        F = [[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]]

        # observation_matrix
        H = [1, 0, 0]

        # transition_covariance
        Q = np.array([[8.28688186e-02, 4.56069709e-02, 1.15608369e-03],
                      [1.29460844e-03, 2.18108024e-02, 1.13942461e-03],
                      [-2.02707405e-04, 8.94215211e-04, 7.30274151e-05]
                      ]) * self.transition_coef

        # observation_covariance
        R = np.array([[5000]]) * self.observation_coef

        # initial_state_mean
        X0 = [0, 0, 0]

        # initial_state_covariance
        P0 = [[1e5, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.filtered_state_mean = X0
        self.filtered_state_covariance = P0

        return KalmanFilter(transition_matrices=F,
                            observation_matrices=H,
                            transition_covariance=Q,
                            observation_covariance=R,
                            initial_state_mean=X0,
                            initial_state_covariance=P0)

    def inference(
        self,
        image: np.ndarray,
        return_frame: bool = False
    ) -> tuple[np.ndarray, np.ndarray, dict[float]] | tuple[
            np.ndarray, np.ndarray, np.ndarray, dict[float]]:
        """
        Estimates marker pose in camera.

        Arguments
            image: np.ndarray
            return_frame:

        Returns:
            tuple[
                frame: np.ndarray - frame with drawn marker edges and axes,
                camera_poses_in_base: np.ndarray - of shape (4, 4),
                weights: dict[np.ndarray] - weights of each marker
                ]
        """

        # Step 1. Detect markers
        corners, ids, rejected_img_points = detect_markers(
            frame=image,
            aruco_dict_type=self.aruco_dict_type,
            matrix_coefficients=self.matrix_coefficients,
            distortion_coefficients=self.distortion_coefficients,
            invert_image=self.invert_image)

        # try:
        #     corners_dict = dict(zip(ids, corners))
        # except TypeError:
        #     corners_dict = {}

        # Step 2. Estimate marker poses in camera
        mtcs, rvecs, tvecs = estimate_marker_poses_in_camera_extrinsic_guess(
            ids=ids,
            corners=corners,
            edge_len=self.marker_edge_len,
            matrix_coefficients=self.matrix_coefficients,
            distortion_coefficients=self.distortion_coefficients,
            use_extrinsic_guess=True,
            init_rvecs=self.last_valid_marker_in_camera_rvec,
            init_tvecs=self.last_valid_marker_in_camera_tvec
            )

        # print(mtcs)
        mtx = mtcs.get(self.marker_id, None)
        rvec = rvecs.get(self.marker_id, None)
        tvec = tvecs.get(self.marker_id, None)

        # Step 3. Very important (?). Update init_rvecs and init_tvecs for
        # correctly estimated marker poses
        if rvec is not None:
            self.last_valid_marker_in_camera_rvec = {
                self.marker_id: rvec}
            self.last_valid_marker_in_camera_tvec = {
                self.marker_id: tvec}

        if return_frame:
            res_image = deepcopy(image)
            if mtx is not None:
                draw_markers_on_frame(
                    frame=res_image,
                    corners=corners,
                    matrix_coefficients=self.matrix_coefficients,
                    distortion_coefficients=self.distortion_coefficients,
                    rvec=rvec,
                    tvec=tvec,
                    edge_len=self.marker_edge_len,
                )

        else:
            res_image = np.array(0)

        if mtx is None:  # if marker is not detected
            # we set 'result' to be previous pose (basically to keep shape)
            mtx_result = self.mtx
            # and set mask = True, which omits all values from the matrix:
            # Maybe it is reasonable to mask the estimation only if
            # we're using a Kalman filter
            if self.apply_kf:
                mtx_result.mask = True

        else:  # if marker is detected
            # we update the result estimation
            mtx_result = np.ma.asarray(mtx)
            mtx_result.mask = False
            if self.debug:
                self.mtx_nofilter = np.ma.asarray(mtx_result)
            # self.camera_in_base = camera_in_base_result

        # Apply Kalman-filter
        # IMPORTANT !
        # We apply Kalman-filter only to "z" coordinate of the measurements
        # !!! Hardcoded! Different from SingleEstimator!
        # FIXME make it optional, set in settings
        if self.apply_kf:
            self.filtered_state_mean, self.filtered_state_covariance = (
                self.kf.filter_update(self.filtered_state_mean,
                                      self.filtered_state_covariance,
                                      observation=mtx_result[2, 3]))
            mtx_result.mask = False
            mtx_result[2, 3] = self.filtered_state_mean[0]

        # Subtract bias
        mtx_result[2, 3] -= self.z_bias
        self.camera_in_base = mtx_result

        if self.debug:
            return res_image, mtx_result, self.mtx_nofilter

        else:
            return res_image, mtx_result

    def __call__(self, image, return_frame=False):
        return self.inference(image, return_frame)

# pass
