import cv2
import numpy as np
from typing import Any, Tuple
from collections import deque

from .utils import patch_img, draw_img_grid


class BaseComparison:
    def __init__(self, *args, **kwargs):

        # Create some random colors
        self.color = np.random.randint(0, 255, (100, 3))

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> bool:
        raise NotImplementedError

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(src=img, 
                                ksize=(self.blur_size, self.blur_size), 
                                sigmaX=0)

    def draw_diff(self, img1: np.ndarray, img2: np.ndarray
                  ) -> Tuple[bool, np.ndarray]:
        raise NotImplementedError


class AbsdiffComparison(BaseComparison):
    def __init__(self,
                 nrows: int,
                 ncols: int,
                 blur_size: int = 5,
                 threshold: int = 20,
                 movement_perc_threshold: float = 0.05,
                 **kwargs):
        super().__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.blur_size = blur_size
        self.threshold = threshold
        self.movement_perc_threshold = movement_perc_threshold

    # def __call__(self, img1: np.ndarray, img2: np.ndarray) -> bool:
    #     img1_blurred = self.preprocess(img1)
    #     img2_blurred = self.preprocess(img2)
    #     diff = cv2.absdiff(src1=img1_blurred, src2=img2_blurred)
    #     thresh_diff = cv2.threshold(src=diff, thresh=self.threshold, 
    #                                 maxval=1, type=cv2.THRESH_BINARY)[1]
    #     patches = patch_img(thresh_diff, self.nrows, self.ncols)
    #     nrows, ncols, h, w, c = patches.shape
    #     print(patches)
    #     perc = np.sum(patches, axis=(2, 3, 4)) / (h*w*c)

    #     return perc > self.movement_perc_threshold

    def __call__(self,
                 img1: np.ndarray,
                 img2: np.ndarray,
                 visualize: bool = False
                 ) -> Tuple[bool, np.ndarray]:
        img1_blurred = self.preprocess(img1)
        img2_blurred = self.preprocess(img2)
        diff = cv2.absdiff(src1=img1_blurred, src2=img2_blurred)
        kernel = np.ones((15, 15))
        diff = cv2.dilate(diff, kernel, 1)
        thresh_diff = cv2.threshold(src=diff, thresh=self.threshold, 
                                    maxval=255, type=cv2.THRESH_BINARY)[1]
        patches_diff = patch_img(thresh_diff, self.nrows, self.ncols)
        nrows, ncols, h, w, c = patches_diff.shape
        perc = np.sum(patches_diff, axis=(2, 3, 4)) / (h*w*c*255)
        if visualize:
            gray_diff = cv2.cvtColor(thresh_diff, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(image=gray_diff,
                                           mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)

            vis_img = img2.copy()
            cv2.drawContours(image=vis_img, contours=contours, 
                             contourIdx=-1, color=(0, 255, 0), 
                             thickness=2, lineType=cv2.LINE_AA)
            patches = patch_img(vis_img, self.nrows, self.ncols)
            for ij, ij_perc in np.ndenumerate(perc):
                if ij_perc > self.movement_perc_threshold:
                    patches[ij] = cv2.circle(patches[ij],
                                             (h//2, w//2), radius=(h+w)//20,
                                             color=(255, 0, 0), thickness=-1)
            vis_img = draw_img_grid(patches)
        else:
            vis_img = img2

        return perc > self.movement_perc_threshold, vis_img


class CameraMovement():
    def __init__(self,
                 compare_class: BaseComparison,
                 init_frame: np.ndarray = None,
                 history_depth: int = 5,
                 moved_patches_threshold: float = 0.6,
                 *kwargs):

        # Support frame is compared with current frame
        if init_frame:
            self.support = init_frame
        else:
            self.support = None

        # Buffer stores last $history_depth$ frames
        self.buffer = deque()
        self.compare = compare_class
        self.results_history = []
        self.history_depth = history_depth
        self.moved_patches_threshold = moved_patches_threshold

    def has_moved(self, img: np.ndarray,
                  draw: bool = False):
        """
        Logic is following:
        1. If we have NO support frame -> has_moved = True.
            support frame = current image
            add current image to buffer
            (initialization without support frame)
        2. If we have a support frame -> do comparison.
            If has_moved = False -> support frame = current image
                add current image to buffer
            If has_moved = True -> keep previous support frame
                add current image to buffer
            If has_moved = True AND all(results_history) = True
            (the camera if moving for long time)
            -> support frame = last image from buffer
                add current image to buffer
        """
        if self.support is not None:


            moved_patches, vis_img = self.compare(self.support, img,
                                                    visualize=draw)

            # calculate percentage of moved patches
            # self.moved_patches_threshold
            # print(moved_patches)

            # print(moved_patches.shape)

            # print(moved_patches.sum())
            # print(moved_patches.sum()/(moved_patches.shape[0]*moved_patches.shape[1]))

            moved_patches_perc = moved_patches.sum() / \
                (moved_patches.shape[0]*moved_patches.shape[1])

            moved = moved_patches_perc >= self.moved_patches_threshold

            self._update_buffer(img)
            self._update_history(img, moved)

            if not moved:
                self.support = img

            if (len(self.buffer) == self.history_depth) \
                    and all(self.results_history):
                self.support = self.buffer[0]

            font = cv2.FONT_HERSHEY_SIMPLEX

            org = (180, 100)
            font_scale = 3
            color = (167, 94, 13)
            thickness = 2
            vis_img = cv2.putText(vis_img, str(moved), org, font,
                                  font_scale, color, thickness, cv2.LINE_AA)

        else:
            self.results_history = [True]
            vis_img = img
            self.support = img
            self._update_buffer(img)


        return np.all(self.results_history), vis_img

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.has_moved(*args, **kwds)

    def _update_buffer(self, image: np.ndarray):
        self.buffer.append(image)
        if len(self.buffer) > self.history_depth:
            self.buffer.popleft()

    def _update_history(self, img: np.ndarray,
                        has_moved: bool):
        self.results_history.append(has_moved)
        if len(self.results_history) > self.history_depth:
            del self.results_history[0]
