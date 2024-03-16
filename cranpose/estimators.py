"""
Provides classes of estimators.
"""
import os
import warnings
from copy import deepcopy
from typing import Callable, Tuple, Union
from collections import deque

import numpy as np
import torch
import yaml
from pykalman import KalmanFilter

from .cammovement import CameraMovement
from .campose import (
    check_wrong_estimations,
    compute_marker_weights,
    compute_weighted_pose_estimation,
    detect_markers_opencv,
    detect_markers_combined_v2,
    estimate_camera_pose_in_base,
    estimate_camera_pose_in_markers,
    estimate_marker_poses_in_camera_extrinsic_guess,
)

from .detectors.detector import ArUCoDetector, IdDecodeRefiner
from .detectors.deep.stag_decode.detection_engine import DetectionEngine
from .detectors.deep.deeptag_model_setting import load_deeptag_models
from .detectors.yolo.inference import YoloCranpose
from .utils import (
    aruco_codebook_by_dict_type,
    create_marker_mtcs,
    draw_markers_on_frame,
    draw_weights_on_frame,
    f_area,
    f_left_x_002,
    f_right_x_002,
    flatten_list,
)

config_filename = os.path.join(os.path.dirname(__file__), "default_config.yaml")

with open(config_filename, encoding='utf8') as f:
    DEFAULTS = yaml.load(f, Loader=yaml.FullLoader)

class PoseSingle:
    """
    Arguments
    ---------
    aruco_dict_type : str
        Тип Аруко маркеров (напр. DICT_4X4_50).
    camera_orientation : int
        Ориентация камеры -1 rear, 1 front.
    n_markers : int
        Максимальное число маркеров.
    marker_step : float
        Шаг между маркерами (м).
    marker_edge_len : float
        Длина стороны маркера (м).
    matrix_coefficients : np.ndarray
        Калибровка камеры intrinsic.
    distortion_coefficients : np.ndarray
        Коэффициенты дисторсии.
    apply_kf : bool = False - применять ли фильтр Калмана,
    transition_coef : float = 1 - коэффициент Калман фильтра transition_coef,
      чем больше, тем быстрее фильтр,
    observation_coef : float = 1 - коэффициент Калман фильтра observation_coef,
      чем больше, тем медленнее фильтр,
    size_weight_func : Callable - функция для взвешивания маркеров по их
      размеру (принимает float от 0 до 1)
    left_edge_weight_func : Callable - функция для взвешивания маркеров по
        положению относительно левого края кадра
        (принимает float от 0 до 1)
    right_edge_weight_func : Callable - функция для взвешивания маркеров по
        положению относительно правого края кадра
        (принимает float от 0 до 1)
    x_bias : float = 0
        Сдвиг камеры по X относительно начала координат крана.
    invert_image : bool = False - инвертировать ли картинку
    debug : bool = False, влючение дебаг режима. в нем, эстимейтор будет
        возвращать и предсказание с калман фильтром
        и без него. Помимо этого, объект PoseMultiple, который инициализован
        хобя бы с одним объектом PoseSingle, у которого debug=True, будет
        возвращать не одну матрцу а кортеж из трех элементов:
        mean_pred, debug_preds, debug_preds_weights
    """

    def __init__(
        self,
        aruco_dict_type: int,
        camera_orientation: int,
        n_markers: int,
        marker_poses: dict,
        marker_edge_len: float,
        matrix_coefficients: np.ndarray,
        distortion_coefficients: np.ndarray,
        apply_kf: bool = False,
        transition_coef: float = 1,
        observation_coef: float = 1,
        x_bias: float = 0,
        size_weight_func: Callable = f_area,
        left_edge_weight_func: Callable = f_left_x_002,
        right_edge_weight_func: Callable = f_right_x_002,
        camera_movement: CameraMovement = None,
        use_deep_detector_stg2: bool = False,
        detector_type = 'yolo',
        deep_detector_checkpoint_dir = os.path.join(
            os.path.dirname(__file__),'detectors','deep','models'),
        deep_detector_device = 'cpu',
        camera_movement_n_init_frames: int = 10,
        transform_imgage_in_second_classic_detector: bool = True,
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
        self.deep_detector_checkpoint_dir = deep_detector_checkpoint_dir
        self.n_init_frames = camera_movement_n_init_frames
        self.camera_movement = camera_movement
        self.debug = debug
        self.transform_imgage_in_second_classic_detector = \
            transform_imgage_in_second_classic_detector

        self.adaptiveThreshWinSizeMin = 13
        self.adaptiveThreshWinSizeMax = 33
        self.adaptiveThreshWinSizeStep = 10
        self.adaptiveThreshConstant = 3

        self.detect_function = self._detect_markers_v2

        self.is_detection = False

        # This entity will remember last seen ids from last N frames
        # TODO pass this N from outside
        # This is required for id refiner. 
        self.ids_buffer = deque(maxlen=3)

        if camera_movement is not None:
            warnings.warn("""
                          Do not use vis_movement = True,
                          when calling the created object in production.
                          Use it only for debugging camera movement settings.
                          """)

        if use_deep_detector_stg2:
            if torch.cuda.is_available():
                if type(deep_detector_device) == str:
                    deep_detector_device = torch.device(deep_detector_device)
                self.deep_detector_device = deep_detector_device
            else:
                self.deep_detector_device = 'cpu'

            self.deep_detector = self._init_deep_detector(detector_type)

        else:
            self.deep_detector = None


        self.n_frame = 0  # frame index counted until it reaches n_init_frames
        self.support_frame = None

        # initial pose matrix
        self.camera_in_base = np.ma.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0],
                                           [0, 0, 1, 0],
                                           [0, 0, 0, 1]])

        # Kalman-Filter initialization
        self.filtered_state_mean = None
        self.filtered_state_covariance = None
        if apply_kf:
            warnings.warn("""Using KF in PoseSingle object is deprecated
              and will be completely removed in the future.""")
            self.camera_in_base.mask = False
            # self.camera_in_base.mask = True
            self.kf = self._init_kf()
        else:
            self.camera_in_base.mask = False
            self.kf = None

        self.camera_in_base_nofilter = np.ma.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]])

        # Initialise IdDecodeRefiner
        self.id_refiner = IdDecodeRefiner(
            all_marker_poses=self.all_marker_poses,
            marker_edge_len = marker_edge_len,
            matrix_coefficients = matrix_coefficients,
            distortion_coefficients = distortion_coefficients,
            guess_threshold = 1, #  TODO pass this N from outside
        )

    def _init_deep_detector(self, detector_type: str = 'yolo'):

        assert detector_type in ('yolo', 'deep')

        if detector_type == 'deep':
            # Turn marker codes from aruco dicitonary into codebook
            codebook = aruco_codebook_by_dict_type(
                aruco_dict_type=self.aruco_dict_type)

            # a piece of hardcode
            checkpoint_dir = self.deep_detector_checkpoint_dir
            tag_family = "aruco"
            hamming_dist= 8

            model_detector, model_decoder, device, tag_type, grid_size_cand_list = \
                load_deeptag_models(tag_family, checkpoint_dir=checkpoint_dir,
                                    device=self.deep_detector_device) 

            deep_detector = DetectionEngine(
                model_detector, model_decoder, device, tag_type, grid_size_cand_list,
                stg2_iter_num= 2, # 1 or 2
                min_center_score=0.2, min_corner_score = 0.2, # 0.1 or 0.2 or 0.3
                batch_size_stg2 = 4, # 1 or 2 or 4
                hamming_dist= hamming_dist, # 0, 2, 4
                cameraMatrix = self.matrix_coefficients,
                distCoeffs= self.distortion_coefficients,
                codebook = codebook,
                tag_real_size_in_meter_dict = {-1:self.marker_edge_len})

        elif detector_type == 'yolo':
            deep_detector = YoloCranpose(device=self.deep_detector_device)

        return deep_detector

    def _init_kf(self):
        
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

    def init_camera_movement(self,
                             nrows=4,
                             ncols=4,
                             blur_size=5,
                             ):
        ...
        # kwargs = dict(nrows=nrows, ncols=ncols, blur_size=5, H=h, W=w)
        # diff = AbsdiffComparison(threshold = 15,
        #                          movement_perc_threshold = 0.05,
        #                          **kwargs)

    def _detect_markers(self, image: np.ndarray
                       ) -> Tuple[np.array, np.array, np.array, dict]:
        """
        Комбинированная детекция в два шага
        1. Классиеческий детектор
        2. Если ничего не обнаружено на шаге 1: 
        полноценный нейростевой детектор (только двустадийный дип детектор)
        """

        # /// Attempt to detect them with the classical method ///
        corners, ids, rejected_img_points = detect_markers_opencv(
            frame = image,
            aruco_dict_type = self.aruco_dict_type,
            matrix_coefficients = self.matrix_coefficients,
            distortion_coefficients = self.distortion_coefficients,
            adaptiveThreshWinSizeMin = self.adaptiveThreshWinSizeMin,
            adaptiveThreshWinSizeMax = self.adaptiveThreshWinSizeMax,
            adaptiveThreshWinSizeStep = self.adaptiveThreshWinSizeStep,
            adaptiveThreshConstant = self.adaptiveThreshConstant,
            invert_image = self.invert_image)
        
        # Do some postprocessing based on the result
        if ids is not None:
            mask = np.array([
                    id in self.all_marker_poses.keys() for id in ids
                ])
            ids = ids[mask]
            corners = np.array(corners)[mask]
            corners_dict = dict(zip(ids, corners))
        else:
            mask = None
            corners_dict = {}

        # /// Optional detection with deep detector ///
        # (only if nothing is detected with the function above)
        if self.deep_detector and corners_dict == {}:
            corners, ids = self.deep_detector.inference(
                image = image, detect_scale=None)
            if len(corners) > 0:
                mask = np.array([
                        id in self.all_marker_poses.keys() for id in ids
                    ])
                ids = np.array(ids)[mask]
                corners = np.array(corners)[mask]

                corners_dict = dict(zip(ids, corners))

            if len(corners) == 0:
                mask = None
                corners_dict = {}
        
        # print(corners, ids)

        return corners, ids, mask, corners_dict
    
    def _detect_markers_v2(self, image: np.ndarray
                           ) -> Tuple[np.array, np.array, np.array, dict]:
        """
        Комбинированная детекция в два шага
        1. Классиеческий детектор
        2. Если ничего не обнаружено на шаге 1: 
        нейросетевой детектор углов + классический детектор для декодинга
        (работает и ёло и дип)
        """
        # /// Step1. Attempt to detect them with the classical method ///
        corners, ids, rejected_img_points = detect_markers_opencv(
            frame=image,
            aruco_dict_type=self.aruco_dict_type,
            matrix_coefficients=self.matrix_coefficients,
            distortion_coefficients=self.distortion_coefficients,
            adaptiveThreshWinSizeMin = self.adaptiveThreshWinSizeMin,
            adaptiveThreshWinSizeMax = self.adaptiveThreshWinSizeMax,
            adaptiveThreshWinSizeStep = self.adaptiveThreshWinSizeStep,
            adaptiveThreshConstant = self.adaptiveThreshConstant,
            invert_image=self.invert_image)
        
        # Do some postprocessing based on the result
        if ids is not None:
            mask = np.array([
                    id in self.all_marker_poses.keys() for id in ids
                ])
            ids = ids[mask]
            corners = np.array(corners)[mask]
            corners_dict = dict(zip(ids, corners))
        else:
            mask = None
            corners_dict = {}

        # /// Step 2. Optional detection with deep detector ///
        # If NO markers have been detected with classical method,
        # from the whole picture, lets take advantage of a deep detector
        # and help it
        if corners_dict == {}:
            corners, ids = detect_markers_combined_v2(
                image=image,
                aruco_dict_type=self.aruco_dict_type,
                deep_detector=self.deep_detector,
                transform_imgage_in_second_classic_detector = \
                    self.transform_imgage_in_second_classic_detector)

            # Do some postprocessing based on the result
            if ids is not None:

                mask = np.array([
                        id in self.all_marker_poses.keys() for id in ids
                    ])
                ids = ids[mask]
                corners = np.array(corners)[mask]
                corners_dict = dict(zip(ids, corners))
            else:
                mask = None
                corners_dict = {}
        return corners, ids, mask, corners_dict
    
    def _detect_markers_v3(self, image: np.ndarray
                          ) -> Tuple[np.array, np.array, np.array, dict]:
        """
        Детекция только нейросетевым детектором углов +
        классический детектор для декодинга
        (работает и ёло и дип)
        """
        mask = None
        corners_dict = {}
        # /// Step 1. Detection with deep detector ///
        # from the whole picture, lets take advantage of a deep detector
        # and help it
        if corners_dict == {}:
            corners, ids = detect_markers_combined_v2(
                image=image,
                aruco_dict_type=self.aruco_dict_type,
                deep_detector=self.deep_detector,
                transform_imgage_in_second_classic_detector = \
                    self.transform_imgage_in_second_classic_detector)

            # Do some postprocessing based on the result
            if ids is not None:

                mask = np.array([
                        id in self.all_marker_poses.keys() for id in ids
                    ])
                ids = ids[mask]
                corners = np.array(corners)[mask]
                corners_dict = dict(zip(ids, corners))
            else:
                mask = None
                corners_dict = {}
        return corners, ids, mask, corners_dict
    
    def _filter_out_wrong_estimations(self,
                                     camera_in_markers,
                                     corners_dict,
                                     mtcs,
                                     rvecs, tvecs):
        """
        Filters out incorrect estimations (inplace).
        Thus, links to original objects should be given to the method
        """
        rejected_marker_ids = check_wrong_estimations(
            camera_in_markers=camera_in_markers,
            correct_view_dir=self.camera_orientation)
        try:
            for marker_id in rejected_marker_ids:
                corners_dict.pop(marker_id)
                mtcs.pop(marker_id)
                rvecs.pop(marker_id)
                tvecs.pop(marker_id)
                camera_in_markers.pop(marker_id)
        except Exception as e:
            print('A difficult to handle exception:', e)
            # print(mask)
            # print(ids)
            # print(corners)
            print(corners_dict)
            print(rejected_marker_ids)
            print(e)
            raise Exception

    def _draw_res(self, image,
                  return_frame,
                  corners,
                  corners_dict,
                  weights,
                  rvecs,
                  tvecs):
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

            # res_image = draw_weights_on_frame(
            #     res_image,
            #     corners_dict,
            #     weights,
            # )

        else:
            res_image = np.array(0)
        
        return res_image
    
    def _subtract_bias(self, camera_in_base_weighted):
            if camera_in_base_weighted.size == 0:  # if marker is not detected
                self.is_detection = False
                # we set 'result' to be previous pose (basically to keep shape)
                camera_in_base_result = self.camera_in_base
                # and set mask = True, which omits all values from the matrix:
                # Maybe it is reasonable to mask the estimation
                # only if we're using a kf
                if self.apply_kf:
                    # camera_in_base_result.mask = True
                    camera_in_base_result.mask = False

            else:  # if marker is detected
                self.is_detection = True
                # we update the result estimation
                camera_in_base_result = np.ma.asarray(camera_in_base_weighted)
                camera_in_base_result.mask = False

                # Subtract bias
                camera_in_base_result[0, 3] -= self.x_bias
                self.camera_in_base_nofilter = np.ma.asarray(
                    camera_in_base_weighted)
                # self.camera_in_base = camera_in_base_result
            return camera_in_base_result
    
    def _kf_step(self, camera_in_base_result):
        # IMPORTANT ! We apply Kalman-filter only to "x" coordinate
        if self.apply_kf:
            self.filtered_state_mean, self.filtered_state_covariance = (
                self.kf.filter_update(self.filtered_state_mean,
                                      self.filtered_state_covariance,
                                      observation=camera_in_base_result[0, 3]))
            # TODO check if OK:
            # camera_in_base_result.mask = False
            camera_in_base_result[0, 3] = self.filtered_state_mean[0]
        return camera_in_base_result
    
    def estimate_pose_with_detection(
        self,
        image: np.ndarray,
        return_frame: bool = False
    ) -> tuple[np.ndarray, np.ndarray, dict[float]] | tuple[
            np.ndarray, np.ndarray, np.ndarray, dict[float]]:
        """
        Takes image from camera and proceeds pose estimation 
        in base coordinate system.

        Logic without KF:
        If there's a detection(s):
            - estimate pose, return estimation
        If there's no detection:
            - return last estimation

        Logic with KF:
        If there's a detection(s):
            - estimate pose, return filtered prediction, update filter
        If there's no detection:
            - return filter prediction

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

        # /// Step 1. Detect markers ///

        corners, ids, mask, corners_dict = self.detect_function(image)

        # /// Step 2. Estimate marker poses in camera ///
        mtcs, rvecs, tvecs = estimate_marker_poses_in_camera_extrinsic_guess(
            ids=ids,
            corners=corners,
            edge_len=self.marker_edge_len,
            matrix_coefficients=self.matrix_coefficients,
            distortion_coefficients=self.distortion_coefficients,
            use_extrinsic_guess=True,
            init_rvecs=self.last_valid_marker_in_camera_rvec,
            init_tvecs=self.last_valid_marker_in_camera_tvec)

        # /// Step 3. Estimate camera poses in markers ///
        camera_in_markers = estimate_camera_pose_in_markers(
            mtcs)  # here we're just inverting matrices

        # /// Step 4. Filter incorrect estimations ///
        self._filter_out_wrong_estimations(
            camera_in_markers, corners_dict,
            mtcs, rvecs, tvecs)

        # /// Step 5. Estimate camera pose in base ///
        camera_in_base = estimate_camera_pose_in_base(self.all_marker_poses,
                                                      camera_in_markers)
        # /// Step 6. Assign weights to markers ///
        weights = compute_marker_weights(
            corners_dict,
            *image.shape[:2],
            weight_func_area=self.size_weight_func,
            weight_func_left_edge=self.left_edge_weight_func,
            weight_func_right_edge=self.right_edge_weight_func,
        )

        # /// Step 7. Compute weighted pose ///
        if sum(weights.values()) != 0:
            camera_in_base_weighted = compute_weighted_pose_estimation(
                camera_in_base, weights)
        else:
            camera_in_base_weighted = np.array([])

        # /// Step 8. Very important (!) ///
        # Update init_rvecs and init_tvecs for correctly estimated marker poses
        self.last_valid_marker_in_camera_rvec.update(rvecs)
        self.last_valid_marker_in_camera_tvec.update(tvecs)

        res_image = self._draw_res(
            image, return_frame, corners,
            corners_dict, weights, rvecs, tvecs)
        
        # /// Step 9. Subtract bias ///                
        camera_in_base_result = self._subtract_bias(camera_in_base_weighted)

        # /// Step 10. Apply Kalman-filter ///
        self.camera_in_base = self._kf_step(camera_in_base_result)

        if self.debug:
            return res_image, camera_in_base_result, \
                self.camera_in_base_nofilter, weights

        else:
            return res_image, camera_in_base_result, weights
        
    def estimate_pose(
        self,
        corners_dict: dict,
        image: np.array,
        return_frame: bool = False,
        corners_raw: np.ndarray = None,
        ids_raw: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[float]] | tuple[
            np.ndarray, np.ndarray, np.ndarray, dict[float]]:
        """
        Takes image from camera and proceeds pose estimation 
        in base coordinate system.

        Logic without KF:
        If there's a detection(s):
            - estimate pose, return estimation
        If there's no detection:
            - return last estimation

        Logic with KF:
        If there's a detection(s):
            - estimate pose, return filtered prediction, update filter
        If there's no detection:
            - return filter prediction

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

        # /// Step 1. Extract detections ///

        corners, ids = list(corners_dict.values()), list(corners_dict.keys())
        # print(corners_dict)
        # /// Step 1.1. Refine ids ///
        # If we pass raw corners and ids, make refinement
        # if (ids is not None) and (corners is not None) \
        #     and (self.camera_in_base_nofilter.data != np.array(np.eye(4))).all():

        #     self.camera_in_base_nofilter_nobias = \
        #         self.camera_in_base_nofilter.data.copy()
        #     self.camera_in_base_nofilter_nobias[0][3] += self.x_bias 

        #     corners_dict = self.id_refiner.refine_ids(
        #         self.self.id_refiner._guess_id,
        #         corners, ids,
        #         prev_frame_ids = flatten(list(self.ids_buffer)),
        #         estimated_coordinates_nobias = self.camera_in_base_nofilter_nobias,
        #         )

        #     corners, ids = list(corners_dict.values()), list(corners_dict.keys())
        #     print(corners_dict)
        #     self.ids_buffer.append(ids)
        # import ipdb; ipdb.set_trace()
        self.ids_buffer.append(ids)


        # /// Step 2. Estimate marker poses in camera ///
        mtcs, rvecs, tvecs = estimate_marker_poses_in_camera_extrinsic_guess(
            ids=ids,
            corners=corners,
            edge_len=self.marker_edge_len,
            matrix_coefficients=self.matrix_coefficients,
            distortion_coefficients=self.distortion_coefficients,
            use_extrinsic_guess=True,
            init_rvecs=self.last_valid_marker_in_camera_rvec,
            init_tvecs=self.last_valid_marker_in_camera_tvec)

        # /// Step 3. Estimate camera poses in markers ///
        camera_in_markers = estimate_camera_pose_in_markers(
            mtcs)  # here we're just inverting matrices

        # /// Step 4. Filter incorrect estimations ///
        self._filter_out_wrong_estimations(
            camera_in_markers, corners_dict,
            mtcs, rvecs, tvecs)

        # /// Step 5. Estimate camera pose in base ///
        camera_in_base = estimate_camera_pose_in_base(self.all_marker_poses,
                                                      camera_in_markers)
        # /// Step 6. Assign weights to markers ///
        weights = compute_marker_weights(
            corners_dict,
            *image.shape[:2],
            weight_func_area=self.size_weight_func,
            weight_func_left_edge=self.left_edge_weight_func,
            weight_func_right_edge=self.right_edge_weight_func,
        )

        # /// Step 7. Compute weighted pose ///
        if sum(weights.values()) != 0:
            camera_in_base_weighted = compute_weighted_pose_estimation(
                camera_in_base, weights)
        else:
            camera_in_base_weighted = np.array([])

        # /// Step 8. Very important (!) ///
        # Update init_rvecs and init_tvecs for correctly estimated marker poses
        self.last_valid_marker_in_camera_rvec.update(rvecs)
        self.last_valid_marker_in_camera_tvec.update(tvecs)

        res_image = self._draw_res(
            image, return_frame, corners,
            corners_dict, weights, rvecs, tvecs)
        
        # /// Step 9. Subtract bias ///                
        camera_in_base_result = self._subtract_bias(camera_in_base_weighted)

        # /// Step 10. Apply Kalman-filter ///
        self.camera_in_base = self._kf_step(camera_in_base_result)

        if self.debug:
            return res_image, camera_in_base_result, \
                self.camera_in_base_nofilter, weights

        else:
            return res_image, camera_in_base_result, weights

    def inference_with_detection(
            self, image, vis_movement=False, vis_detections=False):
        """
        Combines camera movement estimation and pose estimation
        General pipeline:
        1. Do camera movement detection
        2. Only if camera is moving do pose estimation

        Arguments
        ---------
            image : np.array 
            vis_movement : bool - visualise movement or not. if True - use it only 
                for debug/research, because when movement is visualised,
                no markers can be detected in further steps
            vis_detections: bool - visualise detections or not
        Returnes
        --------
            Tuple[return of self.estimate_pose_with_detection, is_camera_moving]
        """

        # If camera movement is off
        if self.camera_movement is None:
            return *self.estimate_pose_with_detection(image, vis_detections), True


        # If camera movement is on and initialization period is passed
        if (self.camera_movement is not None) & \
           (self.n_frame > self.n_init_frames):
            
            # To overcome possible troubles with Kalman Filter, ignore it when
            # camera is not moving:
            if self.apply_kf:
                # calculate if camera is moving
                is_moving, image = self.camera_movement(
                    image, vis_movement)
                if is_moving:
                    # if self.debug:
                    #     res_image, camera_in_base_result, \
                    #     camera_in_base_nofilter, weights = \
                    #         self.estimate_pose_with_detection(image, vis_detections)
                    # else: 
                    #     res_image, camera_in_base_result, \
                    #     weights = self.estimate_pose_with_detection(image, vis_detections)
                    return *self.estimate_pose_with_detection(image, vis_detections), True
                else:
                    if self.debug:
                        return image, self.camera_in_base, self.camera_in_base_nofilter, {}, False
                    else: 
                        return image, self.camera_in_base, {}, False
                    # return image, self.camera_in_base, {}

            # If Klaman filter is OFF for PoseSingle (IT SHOULD BE OFF) ->
            # still calculate movement, but just to return it further to
            # PoseMultiple.
            else:
                is_moving, image = self.camera_movement(
                    image, vis_movement)
                return *self.estimate_pose_with_detection(image, vis_detections), is_moving
        # For initial n_init_frames do not calculate camera movement
        else:
            self.n_frame += 1
            self.support_frame = image
            return *self.estimate_pose_with_detection(image, vis_detections), True


    def inference(self, corner_dict, image,
                  vis_movement=False, vis_detections=False):
        """
        Combines camera movement estimation and pose estimation
        General pipeline:
        1. Do camera movement detection
        2. Only if camera is moving do pose estimation

        Arguments
        ---------
            image : np.array 
            vis_movement : bool - visualise movement or not. if True - use it only 
                for debug/research, because when movement is visualised,
                no markers can be detected in further steps
            vis_detections: bool - visualise detections or not

        Returns
        --------
            Tuple[return of self.estimate_pose, is_camera_moving]
        """

        # If camera movement is off
        if self.camera_movement is None:
            return *self.estimate_pose(corner_dict, image, vis_detections), True


        # If camera movement is on and initialization period is passed
        if (self.camera_movement is not None) & \
           (self.n_frame > self.n_init_frames):
            
            # To overcome possible troubles with Kalman Filter, ignore it when
            # camera is not moving:
            if self.apply_kf:
                # calculate if camera is moving
                is_moving, image = self.camera_movement(
                    image, vis_movement)
                if is_moving:
                    # if self.debug:
                    #     res_image, camera_in_base_result, \
                    #     camera_in_base_nofilter, weights = \
                    #         self.estimate_pose(image, vis_detections)
                    # else: 
                    #     res_image, camera_in_base_result, \
                    #     weights = self.estimate_pose(image, vis_detections)
                    return *self.estimate_pose(corner_dict, image, vis_detections), True
                else:
                    if self.debug:
                        return image, self.camera_in_base, self.camera_in_base_nofilter, {}, False
                    else: 
                        return image, self.camera_in_base, {}, False
                    # return image, self.camera_in_base, {}

            # If Klaman filter is OFF for PoseSingle (IT SHOULD BE OFF) ->
            # still calculate movement, but just to return it further to
            # PoseMultiple.
            else:
                is_moving, image = self.camera_movement(
                    image, vis_movement)
                return *self.estimate_pose(corner_dict, image, vis_detections), is_moving
        # For initial n_init_frames do not calculate camera movement
        else:
            self.n_frame += 1
            self.support_frame = image
            return *self.estimate_pose(corner_dict, image, vis_detections), True

    def detect_movement(self, image, return_frame):
        ...

    def __call__(self, image, vis_movement=False, vis_detections=False):
        return self.inference_with_detection(image, vis_movement, vis_detections)


class PoseMultiple:
    """
    Update NOTE: changed return structure if debug == True!

    Class to perform pose estimation from multiple cameras.

    Fusing predictions from single estimators:
    1. If all estimators detected at least one marker ->
        take a mean of predictions
    2. If some estimators detected markers but others didn't ->
        take a mean of predictions of those which have detected markers
    3. If no estimators detected markers ->
        3.1 No estimators had any predictions in the past
        (no detection occured in the past) ->
            return a no prediction result ("---")
        3.2 Some estimators had predictions in the past, others didn't ->
            take a mean of those who had
        3.3 All estimators had predictionsin the past  ->
            NOT !!! take a mean of predictions
            BUT !!! take last valid prediction from estimator with most recent detection

    This class is capable of applying Kalman Filter (KF) to the estimation
    It is STRONGLY RECOMMENDED TO APPLY KF HERE, INSTEAD OF PoseSingle objects
    """
    def __init__(self,
                 estimators: list[PoseSingle],
                 should_apply_kf: bool = True,
                 should_refine_ids: bool = True,
                 kf_transition_covariance: np.ndarray =
                  DEFAULTS['KF']['transition_covariance'],
                 kf_observation_covariance: np.ndarray =
                  DEFAULTS['KF']['observation_covariance'],
                 ) -> None:
        """
        Arguments
        ---------
        estimators : list
            Objects of PoseSingle
        apply_kf : bool = True
            Apply Kalman filter to the estimation or not.
        kf_transition_covariance : np.ndarray
            Translation covariance of the Kalman filter.
            Default value is taken from the .yaml file. 
        kf_observation_covariance : np.ndarray
            Observation covariance of the Kalman filter.
            Default value is taken from the .yaml file.

        Returns
        -------
        None
        """

        assert len(estimators) > 0, "At least one estimator must be passed for initialisation."
        # TODO check that estimators have same marker 

        self.estimators = estimators
        self.should_apply_kf = should_apply_kf
        self.should_refine_ids = should_refine_ids

        # initial pose matrix
        self.state_mean = np.ma.array(np.eye(4))
        # Kalman-Filter initialization
        self.filtered_state_mean = None
        self.filtered_state_covariance = None
        if should_apply_kf:
            # self.filtered_state_mean.mask = True
            self.kf = self._init_kf(kf_transition_covariance, kf_observation_covariance)
        else:
            # self.filtered_state_mean.mask = False
            self.kf = None

        # We initialise an IdDecodeRefiner for each estimator because
        # in general estimators may have different camera calibration parameters,
        # and even marker edge lenghts and marker poses.
        self.id_refiners = []
        for single_estimator in self.estimators:
            self.id_refiners.append(IdDecodeRefiner(
                all_marker_poses= single_estimator.all_marker_poses,
                marker_edge_len = single_estimator.marker_edge_len,
                matrix_coefficients = single_estimator.matrix_coefficients,
                distortion_coefficients = single_estimator.distortion_coefficients,
                guess_threshold = 1, #  TODO pass this N from the outside
                )
            )

    def _init_kf(self, kf_transition_covariance, kf_observation_covariance):
        # time step
        dt = 1 / 2

        # transition_matrix
        F = [[1, dt, 0.5 * dt * dt], [0, 1, dt], [0, 0, 1]]

        # observation_matrix
        H = [1, 0, 0]

        # transition_covariance
        Q = kf_transition_covariance

        # observation_covariance
        R = kf_observation_covariance

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
    
    def _kf_step(self, observation):
        """
        does predict-update step of the Kalman filter
        returns estimation or prediction depending of observation presense
        """
        # Apply Kalman-filter
        # IMPORTANT ! We apply Kalman-filter only to "x"
        # coordinate of the measurements

        # This function returnes ESTIMATION
        # (estimation = KF(observation, prediction))
        # if observation.mask == False (observation is available)
        # otherwise, it returnes PREDICTION, if observation.mask == True
        # (no observation):

        self.filtered_state_mean, self.filtered_state_covariance = (
            self.kf.filter_update(self.filtered_state_mean,
                                    self.filtered_state_covariance,
                                    observation=observation[0, 3]))
        # TODO check if OK:
        # camera_in_base_result.mask = False
        observation[0, 3] = self.filtered_state_mean[0]
        return observation


    def inference_with_detection(
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

        # /// Collecting information from single estimators ///

        debug_preds_weights = []
        debug_preds = []
        debug = False
        preds = []
        is_detection = []
        is_moving = []
        for image, estimator in zip(images, self.estimators):

            if image is not None:

                if estimator.debug:
                    _, pred, debug_pred, weights, movement = estimator(image)

                    debug = True
                    if pred.shape:
                        preds.append(pred)
                    is_detection.append(estimator.is_detection)

                    debug_preds.append(debug_pred)
                    debug_preds_weights.append(weights)

                else:
                    _, pred, weights, movement = estimator(image)

                    if pred.shape:
                        preds.append(pred)
                    is_detection.append(estimator.is_detection)
                is_moving.append(movement)


        # /// Algorithm to fuse predictions ///

        # We have a list of predictions, and list of if there's a detection:
        is_detection = np.array(is_detection)
        # 1 and 2
        if any(is_detection):
            # Case when estimators detected somtething
            preds = np.array(preds)[is_detection]
            mean_pred = np.ma.array(np.mean(preds, axis=0))
            # This tells filter to make an ESTIMATION:
            mean_pred.mask = False
                
        # 3
        else:
            # check which estimators have already made a detection
            initialized_estimators = np.array([
                not np.all(pred.data == np.eye(4)) for pred in preds])
            # if an estimator returnes an identity matrix, it means
            # it is not initialized
            if any(initialized_estimators): #  3.2 and 3.3
                
                # TODO remove this old solution (2 lines below)
                # preds = np.array(preds)[initialized_estimators]
                # mean_pred = np.ma.array(np.mean(preds, axis=0))
                mean_pred = self.state_mean

            else:
                mean_pred = np.ma.array(np.eye(4))
            # This tells filter to make a PREDICTION:
            mean_pred.mask = False
            # mean_pred.mask = True
        if debug:
            debug_mean_pred = mean_pred
        if self.should_apply_kf:
            # If ANY cameras says is moving -> apply Kalman Filter as usual
            if any(is_moving):
                filtered_mean_pred = self._kf_step(deepcopy(mean_pred))
            # If no cameras say they're moving -> 
            # the action depends on if they detect anything
            else:
                # if at least one camera detected a marker -> trust it and
                # apply Kalman Filter as usual
                if any(is_detection):
                    filtered_mean_pred = self._kf_step(deepcopy(mean_pred))
                # otherwise, if no cameras are detecting, 
                # and there's no movement -> do neither update nor predict
                # take unfiltered estimation
                else:
                    filtered_mean_pred = mean_pred
            if debug:
                debug_mean_pred = mean_pred
            
            mean_pred = filtered_mean_pred
        self.state_mean = mean_pred

        # if preds != []:
        #     # if we have AT LEAST one detection, we use only preds with detections
        #     if any(is_detection):
        #         preds = np.array(preds)[is_detection]
        #     # else, all Kalman predictions or last valid predictions are fused
        #     mean_pred = np.mean(preds, axis=0)
        # else:  # this case is impossible in current version
        #     mean_pred = np.eye(4)

        if debug:
            return mean_pred, debug_mean_pred, debug_preds, debug_preds_weights
        else:
            return mean_pred
        
    def inference(
        self,
        corner_dicts: list[dict],
        images: list[np.ndarray],
        mask: np.ndarray = None,
        corners_raw: np.ndarray = None,
        ids_raw: np.ndarray = None,
    ) -> np.ndarray | tuple[np.ndarray, list, list]:
        """
        Takes a ndarray of images with order corresponding to estimators
        Arguments:
            images: list[np.ndarray] - images
        Return:
            mean_pred: list[np.ndarray] of shape (4, 4)

        X - coordinate of camera can be retrieved by mean_pred[0, 3] (!)
        """

        if mask is None:
            mask = np.full(len(images), True, bool)

        # /// Collecting information from single estimators ///

        debug_preds_weights = []
        debug_preds = []
        debug = False
        preds = []
        is_detection = []
        is_moving = []

        # FIXME remove after debug
        self.corner_dicts = corner_dicts.copy()

        # ID refinement
        if self.should_refine_ids:
            corner_dicts = self.refine_ids(
                corners_raw, ids_raw,
            )

        # FIXME remove after debug
        self.refined_corner_dicts = corner_dicts.copy()

        for n, (image, estimator, corner_dict, valid) in enumerate(zip(
            images, self.estimators, corner_dicts, mask)):

            if image is not None and valid:

                if estimator.debug:
                    _, pred, debug_pred, weights, movement = estimator.inference(
                        corner_dict, image)

                    debug = True
                    if pred.shape:
                        preds.append(pred)
                    is_detection.append(estimator.is_detection)

                    debug_preds.append(debug_pred)
                    debug_preds_weights.append(weights)

                else:
                    _, pred, weights, movement = estimator.inference(
                        corner_dict, image)
                    if pred.shape:
                        preds.append(pred)
                    is_detection.append(estimator.is_detection)
                is_moving.append(movement)


        # /// Algorithm to fuse predictions ///

        # We have a list of predictions, and list of if there's a detection:
        is_detection = np.array(is_detection)
        # 1 and 2
        if any(is_detection):
            # Case when estimators detected somtething
            preds = np.array(preds)[is_detection]
            mean_pred = np.ma.array(np.mean(preds, axis=0))
            # This tells filter to make an ESTIMATION:
            mean_pred.mask = False
                
        # 3
        else:
            # check which estimators have already made a detection
            initialized_estimators = np.array([
                not np.all(pred.data == np.eye(4)) for pred in preds])
            # if an estimator returnes an identity matrix, it means
            # it is not initialized
            if any(initialized_estimators): #  3.2 and 3.3
                
                # TODO remove this old solution (2 lines below)
                # preds = np.array(preds)[initialized_estimators]
                # mean_pred = np.ma.array(np.mean(preds, axis=0))
                mean_pred = self.state_mean

            else:
                mean_pred = np.ma.array(np.eye(4))
            # This tells filter to make a PREDICTION:
            mean_pred.mask = False
            # mean_pred.mask = True
        if debug:
            debug_mean_pred = mean_pred
        if self.should_apply_kf:
            # If ANY cameras says is moving -> apply Kalman Filter as usual
            if any(is_moving):
                filtered_mean_pred = self._kf_step(deepcopy(mean_pred))
            # If no cameras say they're moving -> 
            # the action depends on if they detect anything
            else:
                # if at least one camera detected a marker -> trust it and
                # apply Kalman Filter as usual
                if any(is_detection):
                    filtered_mean_pred = self._kf_step(deepcopy(mean_pred))
                # otherwise, if no cameras are detecting, 
                # and there's no movement -> do neither update nor predict
                # take unfiltered estimation
                else:
                    filtered_mean_pred = mean_pred
            if debug:
                debug_mean_pred = mean_pred
            
            mean_pred = filtered_mean_pred
        self.state_mean = mean_pred

        # if preds != []:
        #     # if we have AT LEAST one detection, we use only preds with detections
        #     if any(is_detection):
        #         preds = np.array(preds)[is_detection]
        #     # else, all Kalman predictions or last valid predictions are fused
        #     mean_pred = np.mean(preds, axis=0)
        # else:  # this case is impossible in current version
        #     mean_pred = np.eye(4)

        if debug:
            return mean_pred, debug_mean_pred, debug_preds, debug_preds_weights
        else:
            return mean_pred
    
    def refine_ids(
        self,
        corners_raw,
        ids_raw
    ) -> dict: 
        """
        Refinement of ids based on guess_id method.
        For each estimator the list of assumptions is formed from
        ids in the buffer of the current estimator and ids in buffer of
        other estimators +/- 1 dependind on their view direction
        (PoseSingle.camera_orientation).

        Arguments
        ---------
        corners : list | np.ndarray
            A list of corners in the order which corresponds to self.estimators
        """

        refined_corner_dicts = []
        for n, (image_corners_raw, image_ids_raw, estimator, refiner) in enumerate(
            zip(corners_raw, ids_raw, self.estimators, self.id_refiners)):
            
            # Ids from current estimator buffer:
            prev_frame_ids = flatten_list(list(estimator.ids_buffer))

            # Ids from other estimators +/- 1:
            # FIXME it must work even in case of unsorted ids.
            for other_estimator in self.estimators:
                if estimator != other_estimator:
                    ids_from_ohter_estimator =  [
                        other_estimator.camera_orientation + x
                        for x in flatten_list(list(other_estimator.ids_buffer))
                    ]
                    prev_frame_ids.extend(ids_from_ohter_estimator)

            coordinates_to_compare = self.state_mean.data.copy()
            coordinates_to_compare[0][3]+=estimator.x_bias
            refined_corner_dict = refiner.refine_ids(
                method_ = refiner._guess_id,
                corners = image_corners_raw, ids = image_ids_raw,
                prev_frame_ids = prev_frame_ids,
                estimated_coordinates = coordinates_to_compare,
                )
            refined_corner_dicts.append(refined_corner_dict)
        return refined_corner_dicts
        

    def __call__(self, *args, **kwargs):
        return self.inference(*args, **kwargs)


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
        camera_movement: CameraMovement = None,
        camera_movement_n_init_frames: int = 10,
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
        self.camera_movement = camera_movement
        self.n_init_frames = camera_movement_n_init_frames
        self.z_bias = z_bias
        self.debug = debug
        self.n_frame = 0
        # initial pose matrix
        self.mtx = np.ma.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
        
        # Kalman-Filter initialization
        self.filtered_state_mean = None
        self.filtered_state_covariance = None
        if apply_kf:
            self.mtx.mask = False
            # self.mtx.mask = True
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

    def estimate_pose(
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
        corners, ids, rejected_img_points = detect_markers_opencv(
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
                mtx_result.mask = False
                # mtx_result.mask = True

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
        
    def inference(self, image, vis_movement=False, vis_detections=False):
        """
        Combines camera movement estimation and pose estimation
        General pipeline:
        1. Do camera movement detection
        2. Only if camera is moving do pose estimation

        Arguments:
            image: np.array 
            vis_movement: bool - visualise movement or not. if True - use it only 
                for debug/research, because when movement is visualised,
                no markers can be detected in further steps
            vis_detections: bool - visualise detections or not
        Returnes:
            Tuple[return of self.estimate_pose, is_camera_moving]
        """

        # If camera movement is off
        if self.camera_movement is None:
            return *self.estimate_pose(image, vis_detections), True


        # If camera movement is on and initialization period is passed
        if (self.camera_movement is not None) & \
           (self.n_frame > self.n_init_frames):
            
            # To overcome possible troubles with Kalman Filter, ignore it when
            # camera is not moving:
            if self.apply_kf:
                # calculate if camera is moving
                is_moving, image = self.camera_movement(
                    image, vis_movement)
                if is_moving:
                    # if self.debug:
                    #     res_image, camera_in_base_result, \
                    #     camera_in_base_nofilter, weights = \
                    #         self.estimate_pose(image, vis_detections)
                    # else: 
                    #     res_image, camera_in_base_result, \
                    #     weights = self.estimate_pose(image, vis_detections)
                    return *self.estimate_pose(image, vis_detections), True
                else:
                    if self.debug:
                        return image, self.camera_in_base, self.camera_in_base_nofilter, {}, False
                    else: 
                        return image, self.camera_in_base, {}, False
                    # return image, self.camera_in_base, {}

            # If Klaman filter is OFF for PoseSpecial ->
            # then camera movement does no useful job. 
            # ignore it, and say camera is ALWAYS moving
            else:
                return *self.estimate_pose(image, vis_detections), True
        # For initial n_init_frames do not calculate camera movement
        else:
            self.n_frame += 1
            self.support_frame = image
            return *self.estimate_pose(image, vis_detections), True

    def __call__(self, image, vis_movement=False, vis_detections=False):
        return self.inference(image, vis_movement, vis_detections)
    

class CranePoseEstimatior:
    """
    Class for pose estimation for multiple cranes.
    It uses batch detection in order to increase speed of inference.

    ...
    """
    def __init__(self,
                 detector: ArUCoDetector,
                 estimators: dict,
                 debug: bool):
        """
        Arguments
        ---------
        detector_type : str
            A type of aruco detector. Available options: 'yolo', 'deep'
        aruco_dict_type : str
            A type of ArUCo markers (ex. DICT_4X4_50)
        estimators : dict
            A dict of PoseMultiple estimators. Its keys are indices of
            corresponding cranes. When CranePoseEstimatior estimator
            is called, it must receive a dict images with same keys as
            in "estimators".
        """
        self.detector = detector
        self.estimators = estimators
        self.debug = debug
        self.detections = None

    def detect_markers(self, images: dict):
        """
        A method to perform aruco detection.

        """
        ...

    @staticmethod
    def _get_batch(images_dict: dict):
        """
        Transforms a dict to a list of images
        """
        return [j for i in list(images_dict.values()) for j in i]
    
    @staticmethod
    def _get_dict(batch: np.array, dict_like: dict):
        """
        Transforms a np.array of predictions into
        a dict with crane indices
        """
        res = {}
        counter = 0
        for key in dict_like.keys():
            res[key] = []
            for n in range(len(dict_like[key])):
                res[key].append(batch[counter])
                counter += 1
        return res

    def estimate(self,
                 images: dict,
                 mask: np.ndarray = None,
                 return_detections = False):
        
        # /// Detect marker corners and ids ///
        self.detections = self.detector(
            self._get_batch(images))

        corners_dicts_batch, masks_batch, raw_corners_batch, raw_ids_batch = self.detections
        detections_with_ids = self._get_dict(corners_dicts_batch, images)
        raw_corners_by_cranes = self._get_dict(raw_corners_batch, images)
        raw_ids_by_cranes = self._get_dict(raw_ids_batch, images)

        # /// Estimate cordinates ///
        # (this piece of code estimates coordinates by those markers, which ids
        # were detected)
        estimates = {}
        refined_corner_dicts = {}
        for crane_id in self.estimators.keys():
            # if self.debug:
            #     mean_pred, debug_mean_pred, debug_preds, debug_preds_weights = \
            #     self.estimators[crane_id](detections[crane_id], images[crane_id])
            # else:
            #     mean_pred = self.estimators[crane_id](detections[crane_id], images[crane_id])

            estimates[crane_id] = self.estimators[crane_id](
                detections_with_ids[crane_id], images[crane_id], mask,
                raw_corners_by_cranes[crane_id], raw_ids_by_cranes[crane_id])

            refined_corner_dicts[crane_id] = self.estimators[crane_id].refined_corner_dicts

            
        if return_detections:
            result = {
                'estimates': estimates,
                'detections_with_ids': detections_with_ids, 
                'corners': raw_corners_batch, 
                'ids': raw_ids_batch,
                'detections_with_refined_ids': refined_corner_dicts}
        else:
            result = {
                'estimates': estimates}

        return result
        




    
