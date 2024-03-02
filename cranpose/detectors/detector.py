from itertools import compress
import os
from typing import Tuple, Union

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from ..campose import detect_markers_combined_v2, detect_markers_opencv
from ..utils import aruco_codebook_by_dict_type
from .utils import (detect_nn_corners_decode_classic_id, 
                    detect_nn_corners_decode_nn_id)
from .deep.deeptag_model_setting import load_deeptag_models
from .deep.stag_decode.detection_engine import DetectionEngine
from .yolo.inference import YoloCranpose

config_filename = os.path.join(
    os.path.dirname(__file__), f"..{os.sep}default_config.yaml"
)

with open(config_filename, encoding='utf8') as f:
    DEFAULTS = yaml.load(f, Loader=yaml.FullLoader)


def detect_opencv(detector, frame, invert_image = False):
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if invert_image:
        frame = 255 - frame

    corners, ids, rejected_img_points = detector.detectMarkers(frame)

    if ids is not None:
        ids = ids.reshape(1, -1)[0]
    return corners, ids, rejected_img_points

def filter_detections(detected_ids, all_ids, corners):
    # Do some postprocessing based on the result
    if detected_ids is not None:
        mask = np.array([id in all_ids for id in detected_ids])
        detected_ids = detected_ids[mask]
        corners = np.array(corners)[mask]
        corners_dict = dict(zip(detected_ids, corners))
    else:
        mask = None
        corners_dict = {}
    return mask, corners_dict


class ArUCoDetector:
    """
    ArUCo detector can detect ArUCo markers with several methods.

    Here is their description:

    1. Classic opencv-based detection
    2.
    ...

    Options to choose a neural network - based detector:
    1. 'deep' - a chinese deep model
    2. 'yolo' - yolov8 pose model

    Parameters
    ----------

    self.detector : Union[DetectionEngine, YoloCranpose]
    """

    def __init__(
        self,
        detector_type: str,
        aruco_dict_type: int,
        aruco_ids_list: list[int],
        device: Union[str, torch.device],
        matrix_coefficients: np.array,
        distortion_coefficients: np.array,
        is_opencv_step_1=True,
        is_nn_step_2=True,
        is_nn_step_2_deep_id=False,
    ):
        """
        Detector initisation.

        Arguments
        ---------
        detector_type : str
            Type of an aruco detector. Available options: 'yolo', 'deep'
        aruco_dict_type : str
            Type of ArUCo markers (ex. DICT_4X4_50)
        aruco_ids_list : list[int]
            a list of all marker ids from aruco_dict_type that should be detected
        is_opencv_step_1 : bool
            Whether to use opencv algorithm at the 1st detection step
        is_nn_step_2 : bool
            Whether to use NN model at the 2nd detection step
        is_nn_step_2_deep_id : bool
            Decode id with the NN model. Works only with
            nn_step_2_type == 'deep'
        """
        assert detector_type in ('yolo', 'deep')
        # if is_nn_step_2_deep_id:
        #     raise NotImplementedError("This is not implemented")
        # TODO add assertions for detector_type + is_nn_step_2_deep_id

        self.aruco_ids_list = aruco_ids_list
        self.is_opencv_step_1 = is_opencv_step_1
        self.is_nn_step_2 = is_nn_step_2
        self.detector_type = detector_type
        self.is_nn_step_2_deep_id = is_nn_step_2_deep_id


        if torch.cuda.is_available():
            if type(device) == str:
                device = torch.device(device)
            else:
                device = 'cpu'

        if detector_type == 'deep':
            # Turn marker codes from aruco dicitonary into codebook
            codebook = aruco_codebook_by_dict_type(
                aruco_dict_type=aruco_dict_type
            )

            # a piece of hardcode
            checkpoint_dir = DEFAULTS['deep_detector_checkpoints_dir']
            tag_family = "aruco"
            hamming_dist = 8

            model_detector, model_decoder, device, tag_type, grid_size_cand_list = \
                load_deeptag_models(tag_family, checkpoint_dir=checkpoint_dir,
                                    device=device)
            
            if not is_nn_step_2_deep_id:
            
                # By default DetectionEngine's __call__ method is used for non-batch inference
                DetectionEngine.__call__ = DetectionEngine.process_stage_1_batch

            self.nn_detector = DetectionEngine(
                model_detector,
                model_decoder,
                device,
                tag_type,
                grid_size_cand_list,
                stg2_iter_num=2,    # 1 or 2
                min_center_score=0.2,
                min_corner_score=0.2,    # 0.1 or 0.2 or 0.3
                batch_size_stg2=4,    # 1 or 2 or 4
                hamming_dist=hamming_dist,    # 0, 2, 4
                cameraMatrix=matrix_coefficients,
                distCoeffs=distortion_coefficients,
                codebook=codebook
            )

        elif detector_type == 'yolo':
            
            YoloCranpose.__call__ = YoloCranpose.predict_keypoints_batch
            self.nn_detector = YoloCranpose(
                model=YOLO(DEFAULTS['yolo_detector_checkpoint_path']),
                device=device
            )

        # Calibration coefs
        self.matrix_coefficients = matrix_coefficients
        self.distortion_coefficients = distortion_coefficients

        # /// OpenCV detector initialisation ///
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)

        if self.is_opencv_step_1:

            opencv_detector_parameters = cv2.aruco.DetectorParameters()
            opencv_detector_parameters.cornerRefinementMethod = \
                cv2.aruco.CORNER_REFINE_SUBPIX
            opencv_detector_parameters.adaptiveThreshWinSizeMin = DEFAULTS[
                'opencv_detector_adaptive_thresh_win_size_min']
            opencv_detector_parameters.adaptiveThreshWinSizeMax = DEFAULTS[
                'opencv_detector_adaptive_thresh_win_size_max']
            opencv_detector_parameters.adaptiveThreshWinSizeStep = DEFAULTS[
                'opencv_detector_adaptive_thresh_win_size_step']
            opencv_detector_parameters.adaptiveThreshConstant = DEFAULTS[
                'opencv_detector_adaptive_thresh_constant']
            self.opencv_detector = cv2.aruco.ArucoDetector(
                dictionary, opencv_detector_parameters
            )

        if not self.is_nn_step_2_deep_id:

            opencv_decoder_parameters = cv2.aruco.DetectorParameters()
            opencv_decoder_parameters.cornerRefinementMethod = \
                cv2.aruco.CORNER_REFINE_SUBPIX
            opencv_decoder_parameters.adaptiveThreshWinSizeMin = DEFAULTS[
                'opencv_decoder_adaptive_thresh_win_size_min']
            opencv_decoder_parameters.adaptiveThreshWinSizeMax = DEFAULTS[
                'opencv_decoder_adaptive_thresh_win_size_max']
            opencv_decoder_parameters.adaptiveThreshWinSizeStep = DEFAULTS[
                'opencv_decoder_adaptive_thresh_win_size_step']
            opencv_decoder_parameters.adaptiveThreshConstant = DEFAULTS[
                'opencv_decoder_adaptive_thresh_constant']
            self.opencv_decoder = cv2.aruco.ArucoDetector(
                dictionary, opencv_decoder_parameters
            )

    # def _prepare_tensor(self, images: dict):
    #     """
    #     Makes a tensor from a dict of images

    #     Arguments
    #     ---------
    #     images : dict
    #         This dictionary must have following structure:
    #         keys - indices of cranes (same as in )
    #     """

    def _detect_markers(
        self,
        images: np.ndarray,
    ) -> Tuple[list, list]:
        """
        General marker detection pipline.

        It is possible to detect only with opencv algortithm,
        only with a neural network, or combine both approaches as:
        1. Detect markers with opencv
        2. If no markers are detected -> detect with a neural network

        If neural network detection is used, there are two options:
        1. Detect corners with a network and decode marker ID with a
        classic detector.
        (both 'yolo' and 'deep' chinese detectors can be applied)
        2. Detect corners and decode ID with a network.
        (only 'deep' chinese detector can do this)

        Arguments
        ---------
        image : np.ndarray
            Image for detection

        """
        corners_dicts_batch = np.array([{} for _ in range(len(images))]) 
        masks_batch = np.array([None for _ in range(len(images))]) 
        is_detection_batch = np.full(len(images), 0).astype(bool)


        if self.is_opencv_step_1:
            # /// Attempt to detect them with the classical method ///
            for n, image in enumerate(images):
                corners_dict = {} 
                mask = None
                corners, ids, rejected_img_points = detect_opencv(
                    detector=self.opencv_detector,
                    frame=image)
                
                mask, corners_dict = filter_detections(ids, self.aruco_ids_list, corners)
                corners_dicts_batch[n] = corners_dict
                masks_batch[n] = mask
                is_detection_batch[n] = not corners_dict == {}

        if self.is_nn_step_2:
            # /// Optional detection with deep detector ///
            # (only if nothing is detected with the function above,
            # or if opencv detection is off)
            # import ipdb; ipdb.set_trace()

            if np.sum(~is_detection_batch) > 0:
                images_to_nn = list(compress(images, ~is_detection_batch))
                # import ipdb; ipdb.set_trace()
                if self.is_nn_step_2_deep_id:
                    corners_batch, ids_batch = detect_nn_corners_decode_nn_id(
                        images = images_to_nn,
                        detector=self.nn_detector)

                else:

                    corners_batch, ids_batch = detect_nn_corners_decode_classic_id(
                            images = images_to_nn,
                            # images=images[~is_detection_batch],
                            detector=self.nn_detector,
                            decoder=self.opencv_decoder)

                nn_corners_dicts_batch = np.array([{} for _ in range(len(images_to_nn))])
                nn_masks_batch = np.array([None for _ in range(len(images_to_nn))])
                for n, (corners, ids) in enumerate(zip(corners_batch, ids_batch)):
                    # import ipdb; ipdb.set_trace()
                    nn_mask, nn_corners_dict = filter_detections(ids, self.aruco_ids_list, corners)
                    nn_corners_dicts_batch[n] = nn_corners_dict
                    nn_masks_batch[n] = nn_mask
                
                corners_dicts_batch[~is_detection_batch] = nn_corners_dicts_batch
                masks_batch[~is_detection_batch] = nn_masks_batch
            
        return corners_dicts_batch, masks_batch
    
    def __call__(self, *args, **kwargs):
        return self._detect_markers(*args, **kwargs)
