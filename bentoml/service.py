import cv2
import torch
import bentoml
import numpy as np
from bentoml.io import Image
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2

def mask_overlay(image, mask, color=(0, 1, 0)):
    mask = mask.squeeze()
    mask = np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8) * 255
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.0)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img

# Initialize Runner
runner = bentoml.torchscript.get("unet34-torchscript:x3yngrb346ardhbp").to_runner()

# Initialize BentoML Service
ship_segment = bentoml.Service(
    "airbus-segmentation",
    runners=[runner],
)

def preprocess(image):
    transform = Compose([
            A.Resize(768, 768),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return transform(image=image)['image']

# API Creation
@ship_segment.api(input=Image(), output=Image())
def segment(input_image):
    input_image = np.array(input_image)
    transformed_image = preprocess(input_image).unsqueeze(0)
    mask = runner.run(transformed_image)
    mask = (torch.sigmoid(mask.squeeze(0)) >= 0.5).cpu().numpy().transpose((1,2,0)).astype(np.uint8)
    mask = cv2.resize(mask, (input_image.shape[1], input_image.shape[0]))
    return mask_overlay(input_image, mask)


