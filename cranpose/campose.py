import argparse
import sys
import time
from typing import Callable

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from .utils import (ARUCO_DICT, custom_estimatePoseSingleMarkers,
                   custom_estimatePoseSingleMarkers_use_extrinsic_guess,
                   f_area, f_left_x_002, f_right_x_002, poly_area)


def estimate_marker_poses_in_camera(frame: np.ndarray,
                                    aruco_dict_type: cv2.aruco.Dictionary,
                                    edge_len: float,
                                    matrix_coefficients: np.ndarray,
                                    distortion_coefficients: np.ndarray,
                                    return_frame=False):
    """
    Function detects aruco markers from dict "aruco_dict_type"
    and estimates their poses in camera coordinate system

    args:
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients of your camera

    return:
    frame - The frame with the axis drawn on it
    mtcs - dict marker_id: marker_pose
    id - id of the largest marker
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

    # dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    # parameters = cv2.aruco.DetectorParameters()
    # parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # # frame = cv.imread(...)

    # corners, ids, rejected_img_points = detector.detectMarkers(
    #     gray,
    #     cameraMatrix=matrix_coefficients,
    #     distCoeff=distortion_coefficients)

    # If markers are detected
    # Select markers not close to the edges of the image
    height, width = frame.shape[:2]

    mtcs = []
    if len(corners) > 0:

        # Three rows below set only one largest marker for pose estimation
        sizes = [
            max(a[0, :, 0]) - min(a[0, :, 0]) + max(a[0, :, 1]) -
            min(a[0, :, 1]) for a in corners
        ]
        corners = np.array([corners[np.argmax(sizes)]])
        ids = np.array([ids[np.argmax(sizes)]])

        for i in range(0, len(ids)):

            # Estimate pose of each marker and return the values rvec and tvec
            # rvec, tvec, marker_points = cv2.aruco.estimatePoseSingleMarkers
            # (corners[i], edge_len, matrix_coefficients,
            # distortion_coefficients)

            rvec, tvec, marker_points = custom_estimatePoseSingleMarkers(
                corners[i], edge_len, matrix_coefficients,
                distortion_coefficients)

            r = R.from_rotvec(rvec[0][0])
            mtx = np.vstack(
                [np.column_stack([r.as_matrix(), tvec[0][0]]), [0, 0, 0, 1]])

            if return_frame:
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(frame, corners)

                # Draw Axis
                cv2.aruco.drawAxis(frame, matrix_coefficients,
                                   distortion_coefficients, rvec, tvec,
                                   edge_len)

            mtcs.append(mtx)

    else:
        if return_frame:
            return frame, {}, 0
        else:
            return 1, {}, 0

    if return_frame:
        return frame, dict(zip(tuple(ids.reshape(1, -1)[0]),
                               mtcs)), np.argmax(sizes)
    else:
        return 0, dict(zip(tuple(ids.reshape(1, -1)[0]),
                           mtcs)), np.argmax(sizes)


def estimate_marker_poses_in_camera_weighted(
        frame: np.ndarray,
        aruco_dict_type: str,
        edge_len: float,
        matrix_coefficients: np.ndarray,
        distortion_coefficients: np.ndarray,
        return_frame=False):
    """
    Function detects aruco markers from dict "aruco_dict_type"
    and estimates their poses in camera coordinate system

    It uses weigh

    args:
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients of your camera

    return:
    frame - The frame with the axis drawn on it
    mtcs - dict marker_id: marker_pose
    weights - dict marker_id: marker_weight
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray,
        aruco_dict,
        parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients)

    # If markers are detected
    # Select markers not close to the edges of the image
    height, width = frame.shape[:2]

    mtcs = []
    if len(corners) > 0:

        # Calculate respective of each marker
        areas = np.array([
            poly_area(corner[0][:, 0], corner[0][:, 1]) / (height * width)
            for corner in corners
        ])
        area_weights = f_area(areas)
        area_weights /= sum(area_weights)

        sizes = [
            max(a[0, :, 0]) - min(a[0, :, 0]) + max(a[0, :, 1]) -
            min(a[0, :, 1]) for a in corners
        ]

        leftmost_x_coords = np.array(
            [min(a[0, :, 0]) / width for a in corners])
        rightmost_x_coords = np.array(
            [max(a[0, :, 0]) / width for a in corners])

        left_x_weights = f_left_x_002(leftmost_x_coords)
        left_x_weights /= sum(left_x_weights)

        right_x_weights = f_right_x_002(rightmost_x_coords)
        right_x_weights /= sum(right_x_weights)

        weights = right_x_weights * right_x_weights * area_weights
        weights /= sum(weights)

        for i in range(0, len(ids)):

            # Estimate pose of each marker and return the values rvec and tvec

            rvec, tvec, marker_points = custom_estimatePoseSingleMarkers(
                corners[i], edge_len, matrix_coefficients,
                distortion_coefficients)

            r = R.from_rotvec(rvec[0][0])
            mtx = np.vstack(
                [np.column_stack([r.as_matrix(), tvec[0][0]]), [0, 0, 0, 1]])

            if return_frame:
                # Draw a square around the markers
                cv2.aruco.drawDetectedMarkers(frame, corners)

                # Draw Axis
                cv2.aruco.drawAxis(frame, matrix_coefficients,
                                   distortion_coefficients, rvec, tvec,
                                   edge_len)

            mtcs.append(mtx)

    else:
        if return_frame:
            return frame, {}, np.array([0])
        else:
            return 1, {}, np.array([0])

    if return_frame:
        return frame, dict(zip(tuple(ids.reshape(1, -1)[0]), mtcs)), dict(
            zip(tuple(ids.reshape(1, -1)[0]), weights))
    else:
        return 0, dict(zip(tuple(ids.reshape(1, -1)[0]), mtcs)), dict(
            zip(tuple(ids.reshape(1, -1)[0]), weights))


def estimate_marker_poses_in_camera_extrinsic_guess(
        ids,
        corners,
        edge_len: float,
        matrix_coefficients: np.ndarray,
        distortion_coefficients: np.ndarray,
        use_extrinsic_guess=False,
        init_rvecs={},
        init_tvecs={},
):
    """
    Function detects aruco markers from dict "aruco_dict_type"
    and estimates their poses in camera coordinate system

    args:
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:
    frame - The frame with the axis drawn on it
    ids - list of markers' ids
    mtcs - list of matrices (marker coordinate systems in camera coordinate system)
    """

    mtcs = {}
    rvecs = {}
    tvecs = {}

    # If markers are detected
    if len(corners) > 0:
        for i, id in enumerate(ids):

            rvec, tvec, marker_points = \
                custom_estimatePoseSingleMarkers_use_extrinsic_guess(
                    corners[i],
                    edge_len,
                    matrix_coefficients,
                    distortion_coefficients,
                    use_extrinsic_guess,
                    init_rvecs.get(id, None),
                    init_tvecs.get(id, None),
                )

            r = R.from_rotvec(rvec[0][0])
            mtx = np.vstack([np.column_stack([r.as_matrix(), tvec[0][0]]), [0, 0, 0, 1]])

            mtcs[id] = mtx
            rvecs[id] = rvec
            tvecs[id] = tvec

    return mtcs, rvecs, tvecs


def detect_markers(
    frame: np.ndarray,
    aruco_dict_type: str,
    matrix_coefficients: np.ndarray,
    distortion_coefficients: np.ndarray,
    invert_image: bool = False,
):
    """
    Function detects aruco markers from dict "aruco_dict_type"
    and estimates their poses in camera coordinate system

    args:
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients of your camera

    return:
    corners, ids, rejected_img_points
    """

    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if invert_image:
        frame = 255 - frame
    # aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    # parameters = cv2.aruco.DetectorParameters_create()
    # parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
    #     frame,
    #     aruco_dict,
    #     parameters=parameters,
    #     cameraMatrix=matrix_coefficients,
    #     distCoeff=distortion_coefficients)

    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    # Exps:
    # parameters.useAruco3Detection = False
    # parameters.polygonalApproxAccuracyRate = 0.25

    # parameters.minOtsuStdDev = 14.0

    # parameters.adaptiveThreshConstant = 7.0
    # parameters.adaptiveThreshWinSizeMin = 7
    # parameters.adaptiveThreshWinSizeStep = 49
    # parameters.adaptiveThreshWinSizeMax = 369

    # parameters.minMarkerDistanceRate = 0.014971725679291437
    # parameters.maxMarkerPerimeterRate = 10.075976700411534
    # parameters.minMarkerPerimeterRate = 0.2524866841549599
    # parameters.polygonalApproxAccuracyRate = 0.05562707541937206
    # parameters.cornerRefinementWinSize = 9
    # parameters.minCornerDistanceRate = 0.09167132584946237

    # parameters.minDistanceToBorder = 7
    # parameters.cornerRefinementMaxIterations = 149

    # print(parameters.polygonalApproxAccuracyRate)

    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    # print(dir(detector.getDetectorParameters()))
    corners, ids, rejected_img_points = detector.detectMarkers(
        frame,)
        # cameraMatrix=matrix_coefficients,
        # distCoeff=distortion_coefficients)

    if ids is not None:
        ids = ids.reshape(1, -1)[0]
    return corners, ids, rejected_img_points


def compute_marker_weights(
    corner_dict: dict,
    height: float,
    width: float,
    weight_func_area: Callable = f_area,
    weight_func_left_edge: Callable = f_left_x_002,
    weight_func_right_edge: Callable = f_right_x_002,
) -> dict[float]:
    """
    Computes marker weights
    Arguments:
        corner_dict: dict - corner dict from opencv method,
        height: float - frame height,
        width: float - frame width,
        weight_func_area: Callable - function to weight marker size,
        weight_func_left_edge - function to weight marker location relative to leftmost edge,
        weight_func_right_edge - function to weight marker location relative to rightmost edge)

    """
    corners = corner_dict.values()
    ids = corner_dict.keys()
    f_area = weight_func_area
    f_left_x = weight_func_left_edge
    f_right_x = weight_func_right_edge

    # If markers are detected
    if len(corners) > 0:
        # Calculate respective of each marker
        areas = np.array([
            poly_area(corner[0][:, 0], corner[0][:, 1]) / (height * width)
            for corner in corners
        ])
        area_weights = f_area(areas)
        if not sum(area_weights) == 0:
            area_weights /= sum(area_weights)

        sizes = [
            max(a[0, :, 0]) - min(a[0, :, 0]) + max(a[0, :, 1]) -
            min(a[0, :, 1]) for a in corners
        ]

        leftmost_x_coords = np.array(
            [min(a[0, :, 0]) / width for a in corners])
        rightmost_x_coords = np.array(
            [max(a[0, :, 0]) / width for a in corners])
        left_x_weights = f_left_x(leftmost_x_coords)
        if not sum(left_x_weights) == 0:
            left_x_weights /= sum(left_x_weights)

        right_x_weights = f_right_x(rightmost_x_coords)
        if not sum(right_x_weights) == 0:
            right_x_weights /= sum(right_x_weights)

        weights = left_x_weights * right_x_weights * area_weights
        if not sum(weights) == 0:
            weights /= sum(weights)
    else:
        return {}

    return dict(zip(ids, weights))


def estimate_camera_pose_in_base_old(all_markers_in_base: dict,
                                     detected_markers_in_camera: dict):
    """
    Estimates camera pose based on detected markers' poses

    :param all_markers_in_base: dict of all possible markers to be seen. key - marker id, value - marker coordinates in base
    :type all_markers_in_base: dict
    :param detected_markers_in_camera: dict of all detected. key - marker id, value - marker coordinates in camera
    :type detected_markers_in_camera: dict
    :return: camera coordinate system in base
    :rtype: np.array
    """

    camera_in_markers = dict(
        zip(detected_markers_in_camera.keys(), [
            np.linalg.inv(mtx) for mtx in detected_markers_in_camera.values()
        ]))  # camera coordinate system in markers' coordinate systems

    camera_in_base = dict(
        zip(detected_markers_in_camera.keys(), [
            np.dot(all_markers_in_base[marker], camera_in_markers[marker])
            for marker in detected_markers_in_camera
        ]))  # estimation of camera coordinate system in base

    return camera_in_base


def estimate_camera_pose_in_markers(
        detected_markers_in_camera: dict
    ):
    """
    Estimates camera pose based on detected markers' poses

    :param all_markers_in_base: dict of all possible markers to be seen. key - marker id, value - marker coordinates in base (matrix 4x4)
    :type all_markers_in_base: dict
    :param detected_markers_in_camera: dict of all detected. key - marker id, value - marker coordinates in camera
    :type detected_markers_in_camera: dict
    :return: camera coordinate system in base
    :rtype: np.array
    """

    camera_in_markers = dict(
        zip(
            detected_markers_in_camera.keys(),
            [np.linalg.inv(mtx) for mtx in detected_markers_in_camera.values()]
        )
    )  # camera coordinate system in markers' coordinate systems

    return camera_in_markers


def estimate_camera_pose_in_base(
        all_markers_in_base: dict,
        camera_in_markers: dict
    ):

    camera_in_base = dict(
        zip(
            camera_in_markers.keys(),
            [np.dot(all_markers_in_base[marker], camera_in_markers[marker]) for marker in camera_in_markers]
        )
    )  # estimation of camera coordinate system in base

    return camera_in_base


def compute_weighted_pose_estimation(poses: dict, weights: dict):

    rvecs = np.array(
        [R.from_matrix(pose[:3, :3]).as_rotvec() for pose in poses.values()])
    tvecs = np.array([pose[:3, 3] for pose in poses.values()])

    rvec = np.sum(
        [rvec * weight for rvec, weight in zip(rvecs, weights.values())],
        axis=0)
    tvec = np.sum(
        [tvec * weight for tvec, weight in zip(tvecs, weights.values())],
        axis=0)

    if type(rvec) is np.float64:
        return np.array([])
    else:
        r = R.from_rotvec(rvec)
        return np.vstack(
            [np.column_stack([r.as_matrix(), tvec]), [0, 0, 0, 1]])


def estimate_camera_pose_in_base_weighted(
        all_markers_in_base: dict,
        detected_markers_in_camera: dict,
        marker_weights: dict):
    """
    Estimates camera pose based on detected markers' poses

    :param all_markers_in_base: dict of all possible markers to be seen. key - marker id, value - marker coordinates in base
    :type all_markers_in_base: dict
    :param detected_markers_in_camera: dict of all detected. key - marker id, value - marker coordinates in camera
    :type detected_markers_in_camera: dict
    :param marker_weights: weights
    :type marker_weights: dict
    :return: camera coordinate system in base
    :rtype: np.array
    """

    markers_in_base = {key: all_markers_in_base[key] for key in detected_markers_in_camera.keys()}

    camera_in_markers = dict(
        zip(
            detected_markers_in_camera.keys(),
            [np.linalg.inv(mtx) for mtx in detected_markers_in_camera.values()]
        )
    )  # camera coordinate system in markers' coordinate systems

    # By camera_in_markers we understand whether the estimation is correct.
    pop_ids = []
    for marker_id in camera_in_markers:
        if camera_in_markers[marker_id][0][3] > 0:
            pop_ids.append(marker_id)
    for marker_id in pop_ids:
        markers_in_base.pop(marker_id)
        camera_in_markers.pop(marker_id)
        marker_weights.pop(marker_id)
    # We reject incorrect estimations

    camera_in_base = dict(
        zip(
            camera_in_markers.keys(),
            [np.dot(markers_in_base[marker], camera_in_markers[marker]) for marker in camera_in_markers]
        )
    )  # estimation of camera coordinate system in base

    rvecs = np.array([R.from_matrix(camera_in_base_[:3, :3]).as_rotvec() for camera_in_base_ in camera_in_base.values()])
    tvecs = np.array([camera_in_base_[:3, 3] for camera_in_base_ in camera_in_base.values()])

    rvec = np.sum([rvec*weight for rvec, weight in zip(rvecs, marker_weights.values())], axis=0)
    tvec = np.sum([tvec*weight for tvec, weight in zip(tvecs, marker_weights.values())], axis=0)

    if type(rvec) is np.float64:
        camera_in_base = {
            "weighted": np.array([])
        }
    else:
        r = R.from_rotvec(rvec)
        camera_in_base = {
            "weighted": np.vstack([np.column_stack([r.as_matrix(), tvec]), [0, 0, 0, 1]])
        }

    return camera_in_base


def check_wrong_estimations(camera_in_markers, correct_view_dir=-1):
    """
    correct_view_dir= 1 for front camera
    correct_view_dir=-1 for rear camera
    """
    # By camera_in_markers we understand whether the estimation is correct.
    pop_ids = []
    for marker_id in camera_in_markers:
        if camera_in_markers[marker_id][0][3]*correct_view_dir > 0:
            pop_ids.append(marker_id)

    return pop_ids


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        output = pose_esitmation(frame, aruco_dict_type, k, d)

        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
