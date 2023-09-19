import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time

from scipy.spatial.transform import Rotation as R


def pose_esitmation(frame, aruco_dict_type, edge_len, matrix_coefficients, distortion_coefficients):
    """
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    ids - list of markers' ids
    mtcs - list of matrices (marker coordinate systems in camera coordinate system)
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict, parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)

    # If markers are detected

    mtcs = []

    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], edge_len, matrix_coefficients,
                                                                           distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, edge_len)

            r = R.from_rotvec(rvec[0][0])
            mtx = np.vstack([np.column_stack([r.as_matrix(), tvec[0][0]]), [0, 0, 0, 1]])
            mtcs.append(mtx)

    return frame, dict(zip(ids, mtcs))


def estimate_camera_pose_in_base(known_markers: dict, detected_markers: dict):
    """
    Estimates camera pose based on detected markers' poses

    :param known_markers: dict of all possible markers to be seen. key - marker id, value - marker coordinates in base
    :type known_markers: dict
    :param detected_markers: dict of all detected. key - marker id, value - marker coordinates in camera
    :type detected_markers: dict
    :return: camera coordinate system in base
    :rtype: np.array
    """

    markers_in_base = {key: known_markers[key] for key in detected_markers.keys()}

    cam_in_markers = dict(
        zip(
            detected_markers.keys(),
            [np.linalg.inv(mtx) for mtx in detected_markers.values()]
        )
    )  # camera coordinate system in markers' coordinate systems

    cam_in_base = dict(
        zip(
            detected_markers.keys(),
            [np.dot(markers_in_base[marker], cam_in_markers[marker]) for marker in detected_markers]
        )
    )  # estimation of camera coordinate system in base

    return np.mean(cam_in_base.values())


def draw_pose(frame, pose):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)
    thickness = 2

    for i, row in enumerate(pose):
        for j, col in enumerate(row):
            org = (50 + 150 * j, 50 + 50 * i)
            frame = cv2.putText(frame, str("{:.2f}".format(col)), org, font,
                                font_scale, color, thickness, cv2.LINE_AA)

    return frame


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