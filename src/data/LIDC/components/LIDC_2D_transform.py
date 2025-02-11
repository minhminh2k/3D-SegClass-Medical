import numpy as np 
import torch
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms 
from typing import Any, Dict, Optional, Tuple
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from src.data.LIDC.components.LIDC_2D_dataset import LIDC_IDRI_2D_Dataset

class LIDC_IDRI_2D_Transform(Dataset):
    mean = None
    std = None

    def __init__(self, dataset: LIDC_IDRI_2D_Dataset, transform: Optional[Compose] = None) -> None:
        super().__init__()

        self.dataset = dataset

        if transform is not None:
            self.transform = transform
        else:
            self.transform = Compose(
                [
                    A.Resize(256, 256),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        image, mask = self.dataset[index] 

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            
            image = transformed["image"]  # (1, img_size, img_size)
            mask = transformed["mask"]  # (img_size, img_size)
            image = image.to(torch.float)  # (1, img_size, img_size)
            mask = mask.unsqueeze(0).to(torch.float)  # (1, img_size, img_size)

        return image, mask