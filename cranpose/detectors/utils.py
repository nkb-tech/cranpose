import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from ..utils import crop_poly_fill_bg

def detect_nn_corners_decode_nn_id(
        images,
        detector):

    all_corners_per_batch, all_ids_per_batch = [], []
    for image in images:
        corners, ids = detector.inference(image)
        if len(corners) > 0:
            ids = np.array(ids)
            # corners = np.array([marker_corners[0] for marker_corners in corners])
            corners = np.array(corners)

        else:
            ids = None
            corners = np.array([])
        all_corners_per_batch.append(corners)
        all_ids_per_batch.append(ids)
    return all_corners_per_batch, all_ids_per_batch


def detect_nn_corners_decode_classic_id(
        images,
        detector,
        decoder):
    """
    Refactored detect_markers_combined_v2 from campose.py

    It detects corners with a NN detector and decodes ids with
    an opencv decoder. 
    IMPORTANT: "detector" must implement a __call__ method
    suitable for batched inference.

    Arguments
    ---------
    detector : Union[DetectorEngine, YoloCranpose]
        A NN detector. In DetectorEngine case, its __call__ method
        must call DetectorEngine.process_stage_1_batch method.
    decoder : cv2.aruco.ArucoDetector
        A classic detector used only for ids decoding. It takes a
        cropped and color-adjusted version of image for more
        robust detection.
    """

    # /// Step 1. Deep detection ///
    rois_info = detector(images)

    all_corners_per_batch = []
    all_ids_per_batch = []
    
    # /// Step 2. In each image, iterate over rois to find valid markers ///
    # TODO in the future this can be done in parallel
    for image, rois_info_per_image in zip(images, rois_info):
        all_corners_per_image = []
        all_ids_per_image = []
        for roi_info in rois_info_per_image:
            # Crop
            # import ipdb; ipdb.set_trace()
            croped, mask, dst, dst2 = crop_poly_fill_bg(
                image, 
                roi_info.astype(int),
                boundary=15, bgfill=155)
            # TODO find best size and pass it to the func            
            resized_dst_2 = cv2.resize(dst2, [220,220])
            # TODO refine adjustment settings and pass them to the func
            alpha = 2.5 # Contrast control (1.0-3.0)
            beta = -150 # Brightness control (0-100)

            adjusted = cv2.convertScaleAbs(cv2.cvtColor(resized_dst_2, cv2.COLOR_BGR2GRAY),
                                        alpha=alpha, beta=beta)

            # Find markers
            # TODO find best blur parameter
            corners, ids, rejected_img_points = decoder.detectMarkers(
                cv2.medianBlur(adjusted,7))
            
            if ids is not None:
                all_corners_per_image.append([roi_info])
                all_ids_per_image.append(ids[0][0])

        all_corners_per_image = np.array(all_corners_per_image)
    
        if all_ids_per_image == []:
            all_ids_per_image = None
        else:
            all_ids_per_image = np.array(all_ids_per_image)

        all_corners_per_batch.append(all_corners_per_image)
        all_ids_per_batch.append(all_ids_per_image)

    return all_corners_per_batch, all_ids_per_batch


def decode_ids_classic(
        images,
        corners,
        decoder):
    """
    Refactored detect_markers_combined_v2 from campose.py

    It detects corners with a NN detector and decodes ids with
    an opencv decoder. 
    IMPORTANT: "detector" must implement a __call__ method
    suitable for batched inference.

    Arguments
    ---------
    detector : Union[DetectorEngine, YoloCranpose]
        A NN detector. In DetectorEngine case, its __call__ method
        must call DetectorEngine.process_stage_1_batch method.
    decoder : cv2.aruco.ArucoDetector
        A classic detector used only for ids decoding. It takes a
        cropped and color-adjusted version of image for more
        robust detection.
    """

    # /// Step 1. Deep detection ///
    # rois_info = detector(images)

    # import ipdb; ipdb.set_trace()
    all_ids_per_batch = []
    
    # /// Step 2. In each image, iterate over rois to find valid markers ///
    # TODO in the future this can be done in parallel
    for image, corners_per_image in zip(images, corners):
        all_ids_per_image = []
        for single_marker_corners in corners_per_image:
            # Crop
            # import ipdb; ipdb.set_trace()
            croped, mask, dst, dst2 = crop_poly_fill_bg(
                image, 
                single_marker_corners[0].astype(int),
                boundary=15, bgfill=155)
            # TODO find best size and pass it to the func            
            resized_dst_2 = cv2.resize(dst2, [220,220])
            # TODO refine adjustment settings and pass them to the func
            alpha = 2.5 # Contrast control (1.0-3.0)
            beta = -150 # Brightness control (0-100)

            adjusted_image = cv2.convertScaleAbs(cv2.cvtColor(resized_dst_2, cv2.COLOR_BGR2GRAY),
                                        alpha=alpha, beta=beta)

            # Find markers
            # TODO find best blur parameter
            corners, ids, rejected_img_points = decoder.detectMarkers(
                cv2.medianBlur(adjusted_image,7))
            
            if ids is not None:
                all_ids_per_image.append(ids[0][0])
            else: 
                all_ids_per_image.append(None)
            
        all_ids_per_image = np.array(all_ids_per_image)
        all_ids_per_batch.append(all_ids_per_image)

    return all_ids_per_batch