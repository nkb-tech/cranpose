import os
import torch
import numpy as np
from ultralytics import YOLO

model = YOLO(os.path.join(os.path.dirname(__file__), 'models/baseline_nano.pt'))

class YoloCranpose:
    def __init__(self,
                 model = model):
        self.model = model

    def predict_keypoints(self, image):
        results = self.model(image, save=False, augment=False, verbose=False)

        corners = []
        for result in results:
            kptss = np.array(result.keypoints.data.cpu())
            for kpts in kptss:
                # print(kpts.shape)
                # print(kpts)
                if results[0].boxes.cls.shape != torch.Size([0]):
                # if kpts != np.array([]):
                    corners.append(kpts)
        
        return corners
    
    def __call__(self, *args, **kwargs):
        return self.predict_keypoints(*args, **kwargs)