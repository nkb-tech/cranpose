import os
from itertools import compress
from typing import List, Tuple, Union, Callable

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

from ..campose import (
    estimate_camera_pose_in_base,
    estimate_camera_pose_in_markers,
)
from ..utils import (
    aruco_codebook_by_dict_type,
    custom_estimatePoseSingleMarkers_use_extrinsic_guess,
)
from .deep.deeptag_model_setting import load_deeptag_models
from .deep.stag_decode.detection_engine import DetectionEngine
from .utils import (
    decode_ids_classic,
    detect_nn_corners_decode_nn_id,
)
from .yolo.inference import YoloCranpose

config_filename = os.path.join(
    os.path.dirname(__file__), f"..{os.sep}default_config.yaml"
)

with open(config_filename, encoding='utf8') as f:
    DEFAULTS = yaml.load(f, Loader=yaml.FullLoader)


def detect_opencv(detector, frame, invert_image=False):
    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if invert_image:
        frame = 255 - frame

    corners, ids, rejected_img_points = detector.detectMarkers(frame)

    if ids is not None:
        ids = ids.reshape(1, -1)[0]
    return corners, ids, rejected_img_points


def filter_ids(detected_ids, all_ids, corners):
    """
    This function removes predicitons, which ids are not in all_ids list.
    It is applied to detecitons of a single image (not a batch).
    """
    if detected_ids is not None:
        mask = np.array([id in all_ids for id in detected_ids]).astype(bool)
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
            if type(device) is str:
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

                # By default DetectionEngine's __call__ method is used for
                # non-batch inference
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

        # new
        # Here we create a list of results for each image.
        # This list will contain another list which elements will correspond
        # to an individual marker in the image. In a majority of cases,
        # each image will have one marker detected, but sometimes,
        # especially with a neural network, there can be more of them.

        # new
        corners_batch = [None for _ in range(len(images))]
        ids_batch = [None for _ in range(len(images))]

        corners_dicts_batch = np.array([{} for _ in range(len(images))])
        masks_batch = np.array([None for _ in range(len(images))])
        is_detection_batch = np.full(len(images), 0).astype(bool)

        if self.is_opencv_step_1:
            # /// Attempt to detect markers with the classical method ///
            # (the classical method yields only
            # coupled result: conrners with ids)
            for n, image in enumerate(images):
                corners_dict = {}
                mask = None
                corners, ids, rejected_img_points = detect_opencv(
                    detector=self.opencv_detector, frame=image
                )

                mask, corners_dict = filter_ids(
                    ids, self.aruco_ids_list, corners
                )
                corners_dicts_batch[n] = corners_dict
                masks_batch[n] = mask
                is_detection_batch[n] = not corners_dict == {}

                corners_batch[n] = corners
                ids_batch[n] = ids

        if self.is_nn_step_2:
            # /// Optional detection with deep detector ///
            # (This takes only images, in which the above pipeline
            # did not detect anything, or if opencv detection is off)
            # import ipdb; ipdb.set_trace()

            if np.sum(~is_detection_batch) > 0:
                images_to_nn = list(compress(images, ~is_detection_batch))
                if self.is_nn_step_2_deep_id:
                    nn_corners_batch, nn_ids_batch = detect_nn_corners_decode_nn_id(
                        images=images_to_nn, detector=self.nn_detector
                    )
                else:
                    # corners_batch, ids_batch = detect_nn_corners_decode_classic_id(
                    #         images = images_to_nn,
                    #         detector=self.nn_detector,
                    #         decoder=self.opencv_decoder)

                    nn_corners_batch = self.nn_detector(images_to_nn)
                    nn_ids_batch = decode_ids_classic(
                        images_to_nn, nn_corners_batch, self.opencv_decoder
                    )
                # import ipdb; ipdb.set_trace()

                nn_corners_dicts_batch = np.array(
                    [{} for _ in range(len(images_to_nn))]
                )
                nn_masks_batch = np.array(
                    [None for _ in range(len(images_to_nn))]
                )
                for n, (corners, ids) in enumerate(zip(nn_corners_batch,
                                                       nn_ids_batch)):

                    # import ipdb; ipdb.set_trace()
                    nn_mask, nn_corners_dict = filter_ids(
                        ids, self.aruco_ids_list, corners
                    )
                    nn_corners_dicts_batch[n] = nn_corners_dict
                    nn_masks_batch[n] = nn_mask

                corners_dicts_batch[~is_detection_batch
                                    ] = nn_corners_dicts_batch
                masks_batch[~is_detection_batch] = nn_masks_batch

                # corners_batch[~is_detection_batch] = nn_corners_batch
                # ids_batch[~is_detection_batch] = nn_ids_batch

                # TODO check if squeeze() is required
                nn_counter = 0
                for classic_counter, is_nn_image in enumerate(
                        ~is_detection_batch):
                    if is_nn_image:
                        corners_batch[classic_counter] = nn_corners_batch[
                            nn_counter]
                        ids_batch[classic_counter] = nn_ids_batch[nn_counter]

                        nn_counter += 1

        return corners_dicts_batch, masks_batch, corners_batch, ids_batch

    def __call__(self, *args, **kwargs):
        return self._detect_markers(*args, **kwargs)


class IdDecodeRefiner:

    def __init__(
        self,
        all_marker_poses: dict,
        marker_edge_len: float,
        matrix_coefficients: np.ndarray,
        distortion_coefficients: np.ndarray,
        guess_threshold: float = 1.
    ):
        """
        This class instances are assumed to be initialise and called
        from the PoseSingle class (or togehter with PoseSingle initialisation).
        All arguments of this __init__ are available in PoseSingle instances
        (Except for the specific arguments to adjust different methods, such as
        guess_threshold. It should be passed separately, ex. from config).

        Arguments
        ---------
        all_marker_poses : dict
            Marker coordinate matrices in base.
            This dict is returned by create_marker_mtcs() from ../utils.py
            by n_markers and their "x" coordinates.
            Looak at PoseSingle in ../estimators.py for reference.
        marker_edge_len : float
            Length of the marker side.
        matrix_coefficients : np.ndarray
            Camera intrisic calibration matrix.
        distortion_coefficients : np.ndarray
            Camera distortion vector.
        guess_threshold : float
            "X" coordindate residual threshold for setting
            the id with "guess_id" method. (in meters).

        """
        self.all_marker_poses = all_marker_poses
        self.marker_edge_len = marker_edge_len
        self.matrix_coefficients = matrix_coefficients
        self.distortion_coefficients = distortion_coefficients

        self.guess_threshold = guess_threshold

    def _guess_id(
        self,
        corners_in_image: Union[np.ndarray, List[np.ndarray]],
        prev_frame_ids: list,
        estimated_coordinates: np.ndarray
    ) -> List[Union[int, None]]:
        """
        A very simple approach to the id decoding.
        It takes all detections with non-assigned ids with their rvecs and tvecs ???
        from the frame, a list of ids seen in the previous frame, and coordinates of
        the corresponding camera in base.
        Then it tests three options for each marker and each previous id:
        1. Marker's id is -1 from the previous id
        2. Marker's id equals the previous id
        3. Marker's id is +1 to the previous id
        It calculates camera in base ccordinates for each option,
        calculates the difference between previously estimated
        camera in base coordinates and if the smallest difference is
        smaller then the threshold (1 meter), it accepts the guess.

        This method is supposed to be called from a PoseSingle instance.
        """

        assert type(estimated_coordinates) is np.ndarray, (
            f"Wrong type {type(estimated_coordinates)} "
            "of estimated_coordinates_nobias. "
            "Numpy array is required. "
            "Masked array is not supported.")

        # Do not proceed if camera coordinates are identity matrix
        # (no detections have been made after system initialised)
        # Also do not proceed if no markers are in prev_frame_ids

        if (estimated_coordinates == np.array(np.eye(4))).all() | \
           (prev_frame_ids == []) :
            return [None for _ in corners_in_image]


        # TODO test method when there're multiple detections in one image
        guess_ids = []
        for prev_frame_id in prev_frame_ids:
            guess_ids.append(prev_frame_id - 1)
            guess_ids.append(prev_frame_id)
            guess_ids.append(prev_frame_id + 1)
        guess_ids = list(set(guess_ids))

        # print("guess_ids", guess_ids)
        # /// Step 1. Calculate rvecs, tvecs
        mtcs = []
        for corners in corners_in_image:
            rvec, tvec, marker_points = \
                    custom_estimatePoseSingleMarkers_use_extrinsic_guess(
                        corners,
                        self.marker_edge_len,
                        self.matrix_coefficients,
                        self.distortion_coefficients)

            r = R.from_rotvec(rvec[0])
            mtx = np.vstack(
                [np.column_stack([r.as_matrix(), tvec[0]]), [0, 0, 0, 1]]
            )
            mtcs.append(mtx)

        # /// Step 2. Guess id for each marker ///

        choises = []

        for mtx in mtcs:
            # mtx_with_guess_ids = [{guess_id: mtx} for guess_id in guess_ids]
            mtx_with_guess_ids = dict(
                zip(
                    guess_ids,
                    [mtx for _ in guess_ids],
                )
            )
            # Example:
            # mtx_with_guess_ids = {1: same_mtx, 2: same_mtx, 3: same_mtx}
            # This strucrure is needed for further operations

            # /// Step 2.1. Estimate camera poses in markers ///
            camera_in_markers_batch = estimate_camera_pose_in_markers(
                mtx_with_guess_ids
            )

            # /// Step 2.2. Estimate camera in base for each marker ///
            camera_in_base = estimate_camera_pose_in_base(
                self.all_marker_poses, camera_in_markers_batch
            )

            # /// Step 2.3. Compare estimated poses with the last coordinate ///

            residuals_for_guessed_ids = []
            for guessed_id in camera_in_base.keys():
                residual = estimated_coordinates[0, 3] - \
                    list(camera_in_base[guessed_id])[0][3]
                residuals_for_guessed_ids.append(np.abs(residual))

            # print("residuals_for_guessed_ids")
            # print(residuals_for_guessed_ids)

            # If the minimal difference between guessed coordinate
            # is smaller than guess_threshold (meters):
            if np.min(residuals_for_guessed_ids) < self.guess_threshold:
                # Assign the id which gives the minimal difference
                choises.append(
                    list(camera_in_base.keys()
                         )[np.argmin(np.asarray(residuals_for_guessed_ids))]
                )
            else:
                # Do not assign id
                choises.append(None)
        return choises

    def refine_ids(
        self,
        method_: Callable,
        corners: np.array,
        ids: np.array,
        *args,
        **kwargs
    ):

        # Take those detections which ids were not decoded
        ids_to_refine_mask = ids == None
        # print("ids_to_refine_mask", ids_to_refine_mask)
        # print("ids before", ids)
        # Here we have a list of ids, which value was None
        refined_ids = method_(
            np.array(corners)[ids_to_refine_mask],
            *args,
            **kwargs)

        # print("refined_ids", refined_ids)

        # for i, id in enumerate(ids[ids_to_refine_mask]):
        #     if refined_ids[i] is not None:
        #         ids[ids_to_refine_mask][i] = refined_ids[i]
        ids[ids_to_refine_mask] = refined_ids
        
        # print("ids after", ids)

        mask, corners_dict = filter_ids(
                ids, list(self.all_marker_poses.keys()), corners)
        
        return corners_dict

    