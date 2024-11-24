from typing import Any, Optional

# import albumentations as A
# from albumentations import Compose
# from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset
import monai.transforms as transforms

from .brats2017_dataset import Brats2017Task1Dataset

class TransformBrats2017(Dataset):
    mean = None
    std = None

    def __init__(
        self, 
        dataset: Brats2017Task1Dataset, 
        transform: Optional[transforms.Compose] = None
    ) -> None:
        
        super().__init__()

        self.dataset = dataset

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose(
                [
                    transforms.EnsureTyped(keys=["image", "label"], track_meta=False),
                ]
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index) -> Any:
        data = self.dataset[index]
        
        if self.transform:
            data = self.transform(data)
        
        return data