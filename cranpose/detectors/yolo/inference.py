import os
import torch
import numpy as np
from ultralytics import YOLO

model = YOLO(os.path.join(os.path.dirname(__file__), 'models/baseline_nano.pt'))

class YoloCranpose:
    def __init__(self,
                 model = model,
                 device = 'cpu'):
        self.model = model
        self.device = device
        for _ in range(100):
            dummy_img = np.random.randint(
                    low=0,
                    high=255,
                    size=(720, 1280, 3),
                    dtype=np.uint8,
                )
            self.model(
                source=dummy_img,
                device=self.device,
                half=True,
                verbose=False
            )

    def predict_keypoints(self, image):
        results = self.model(image, 
                             device=self.device, 
                             verbose=False,
                             half=True)

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

    def predict_keypoints_batch(self, images):
        results = self.model(images, 
                             device=self.device, 
                             verbose=False,
                             half=True)

        corners_per_batch = []
        for result in results:
            kpts_per_image = np.array(result.keypoints.data.cpu())
            corners_per_image = []

            for kpts in kpts_per_image:
                # print(kpts.shape)
                # print(kpts)
                if result.boxes.cls.shape != torch.Size([0]):
                # if kpts != np.array([]):
                    corners_per_image.append(np.expand_dims(kpts, 0))
            corners_per_batch.append(corners_per_image)
        
        return corners_per_batch
    
    def __call__(self, *args, **kwargs):
        return self.predict_keypoints(*args, **kwargs)