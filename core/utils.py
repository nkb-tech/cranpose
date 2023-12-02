import os
import pickle as pk
import sys

import cv2
import numpy as np

f_left_x_002 = lambda x: np.tanh(10*(x-0.02)).clip(min=0)
f_right_x_002 = lambda x: np.tanh(10*(1-x-0.02)).clip(min=0)

f_left_x_04 = lambda x: np.tanh(10*(x-0.35)).clip(min=0)
f_right_x_04 = lambda x: np.tanh(10*(1-x-0.35)).clip(min=0)

f_area = lambda x: x**1.2


def calibrate(dirpath, square_size, width, height, visualize=False, save_file=False):
    """ Apply camera calibration operation for images in the given directory path. """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = os.listdir(dirpath)

    for fname in images:
        img = cv2.imread(os.path.join(dirpath, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)

        if visualize:
            cv2.imshow('img',img)
            cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if save_file:
        np.save("calibration_matrix", mtx)
        np.save("distortion_coefficients", dist)

    return [ret, mtx, dist, rvecs, tvecs]


ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the ArUco
            # marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # draw the ArUco marker ID on the image
            cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)
            print("[Inference] ArUco marker ID: {}".format(markerID))
            # show the output image
    return image


def display_pose(frame, pose):
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


def create_marker_imgs(savedir, dict_type, marker_size, n_markers) -> None:
    # savedir = 'demo_tags'
    # dict_type = 'DICT_7X7_100'
    # marker_size = 1000
    # n_markers = 3

    # Генерация и сохранение маркеров
    if ARUCO_DICT.get(dict_type, None) is None:
        print(f"ArUCo tag type '{dict_type}' is not supported")
        sys.exit(0)

    aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[dict_type])

    for id in range(n_markers):
        print("Generating ArUCo tag of type '{}' with ID '{}'".format(dict_type, id))
        tag = np.zeros((marker_size, marker_size, 1), dtype="uint8")
        cv2.aruco.drawMarker(aruco_dict, id, marker_size, tag, 1)

        # Save the tag generated
        tag_name = f'{savedir}/{dict_type}_id_{id}.png'
        cv2.imwrite(tag_name, tag)


def create_marker_mtcs(n_markers,
                       marker_poses,
                       save=False,
                       savedir=None) -> dict:
    # savedir = 'demo_tags'
    # n_markers = 3
    # step = 2.095

    # Вычисление систем координат маркеров в системе координат базы

    X, Y, Z = (np.array(list(marker_poses.values())),
               np.zeros(n_markers), np.zeros(n_markers))

    def M_X(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1,  0,  0],
                         [0,  c, -s],
                         [0,  s,  c]])

    def M_Y(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s,  0,  c]])

    def M_Z(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s,  0],
                         [s,  c,  0],
                         [0,  0,  1]])


    a = -90 * 3.1415/180 # угол поворота
    M = M_X(a) # только одно вращение вокруг оси X

    marker_poses = [np.vstack([np.column_stack([M, np.array([x, y, z])]), [0, 0, 0, 1]]) for x, y, z in zip(X, Y, Z)]
    marker_poses_dict = dict(zip(range(n_markers), marker_poses))

    if save:
        with open(f"{savedir}/marker_poses.pk", "wb") as f:
            pk.dump(marker_poses_dict, f)

    return marker_poses_dict


def custom_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        # print(R)
        rvecs.append(R.reshape(1, -1))
        tvecs.append(t.reshape(1, -1))
        trash.append(nada)

    rvecs = np.asarray(rvecs)
    tvecs = np.asarray(tvecs)

    return rvecs, tvecs, trash


def draw_markers_on_frame(
    frame,
    corners,
    matrix_coefficients,
    distortion_coefficients,
    rvec,
    tvec,
    edge_len,
):
    # Draw a square around the markers
    cv2.aruco.drawDetectedMarkers(frame, corners)

    # Draw Axis
    cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients,
                      rvec, tvec, edge_len)


def draw_weights_on_frame(frame, corners, weights):
    font = cv2.FONT_HERSHEY_SIMPLEX

    # print(frame.shape[0])
    # print(corners[3][0][:, 1])
    # print(corners)
    for id in corners:
        org = (int(np.mean(corners[id][0][:, 0])), frame.shape[0] - 100)
        font_scale = 3
        color = (255, 50, 50)
        thickness = 2
        frame = cv2.putText(frame, str("%.2f" % weights[id]), org, font,
                            font_scale, color, thickness, cv2.LINE_AA)

    return frame

def custom_estimatePoseSingleMarkers_use_extrinsic_guess(
        corners,
        marker_size,
        mtx,
        distortion,
        use_extrinsic_guess=False,
        rvec=None,
        tvec=None,
):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    nadas = []
    rvecs = []
    tvecs = []
    i = 0

    if (rvec is None) or (tvec is None):
        use_extrinsic_guess = False

    for c in corners:
        nada, R, t = cv2.solvePnP(
            marker_points,
            corners[i],
            mtx,
            distortion,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=use_extrinsic_guess,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        # print(R)
        rvecs.append(R.reshape(1, -1))
        tvecs.append(t.reshape(1, -1))
        nadas.append(nada)

    rvecs = np.asarray(rvecs)
    tvecs = np.asarray(tvecs)

    return rvecs, tvecs, nadas


def flip_z_axis_neg_det(mtx):
    # Flips Z-axis in case the coordinate system is left-handed

    print('before')

    print('det', np.linalg.det(mtx[:3, :3]))

    print(mtx)
    # print(np.linalg.det(mtx[:3, :3]))

    if not np.isclose(np.linalg.det(mtx[:3, :3]), 1):
        mtx[:, 2] *= -1
        # mtx[2, 3] *= -1

    print('after')

    print('det', np.linalg.det(mtx[:3, :3]))

    print(mtx)

    return mtx


def poly_area(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))
