*** 
# ArUCo positioning v0.1
***
## Basic usage:
1. Generate marker images with generate_markers.ipynb
2. Calibrate camera using calc_camera_matrix.ipynb
3. See sample usages in estimate_pose_hikvision.ipynb or estimate_pose_render.ipynb
 
!!! NOW LOOK THROUGH sample_usage.ipynb 

## Sample calibrations

This repo includes sample calibrations for hikvision DS-2CD2143G2-IS camera with 2.8mm lens (https://hikvision.ru/product/ds_2cd2143g2_is).  
One version is calculated from lens spec (calib/calibration_matrix_render.npy, calib/distortion_coefficients_render.npy) and another one is calibrated from real camera (calib/calibration_matrix_hikvision.npy, calib/distortion_coefficients_hikvision.npy).

## Sample markers

You may find sample markers in /tags directory


