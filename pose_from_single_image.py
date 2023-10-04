'''
Sample Usage:-
python pose_from_single_image.py.py --k_matrix calibration_matrix.npy --d_coeff distortion_coefficients.npy --type DICT_5X5_100
'''
import pickle as pk

import numpy as np
import cv2
import sys
import argparse
import time
import datetime

from core.utils import ARUCO_DICT
from core.posesingle import PoseSingle

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image for pose estimation")
    ap.add_argument("-k", "--k_matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--d_coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-m", "--m_dict", required=True, help="Path to dict of markers' coordinates in base (pickle)")
    ap.add_argument("-e", "--edge_len", required=True, help="Length of merker edge in meters")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")

    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["k_matrix"]
    distortion_coefficients_path = args["d_coeff"]
    marker_dict_path = args["m_dict"]

    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    with open(marker_dict_path,'rb') as f:
        marker_dict = pk.load(f)

    estimator = PoseSingle(aruco_dict_type,
                           args["--edge_len"],
                           k,
                           d,
                           marker_dict)

    image = cv2.imread(args["image"])
    time.sleep(.1)

    frame, pose, dt = estimator(image, "now", True)

    print(pose)
    cv2.imshow(frame)