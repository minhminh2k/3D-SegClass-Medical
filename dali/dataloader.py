import os

import time
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm


def loss_func(pred, y):
    pass


def model(x):
    pass


def backward(loss, model):
    pass

class AirbusDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "/home/lenovo/Documents/workspace/3D-SegClass-Medical/ship",
    ) -> None:
        super().__init__()

        self.data_dir = data_dir

        self.filenames = self._get_file_list()
        
        self.transform = Compose(
                [
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Resize(height=768, width=768),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image = self.filenames[index]
        image = np.array(Image.open(image).convert("RGB"), dtype=np.uint8)

        transformed = self.transform(image=image)
        image = transformed["image"]

        return image, 0

    def _get_file_list(self, exts={'.jpg', '.jpeg', '.png'}):
        image_paths = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in exts:
                    image_paths.append(os.path.join(root, file))
        return image_paths

airbus_dataset = AirbusDataset()

airbus_dataloader = DataLoader(
    dataset=airbus_dataset,
    batch_size=16,
    num_workers=4,
)

start_time = time.time()

for batch in tqdm(airbus_dataloader):
    # x, y = batch[0].to("cuda"), batch[1] # torch.Size([16, 3, 768, 768])    
    x, y = batch[0], batch[1]
    # For fun
    pred = model(x)
    loss = loss_func(pred, y)
    backward(loss, model)
    pass

end_time = time.time()

total_time = end_time - start_time
print(f"Total time: {total_time:.4f} seconds") 
# Total time: 143.2727 seconds 6.81it/s
# Total time: 134.3824 seconds 7.26it/s