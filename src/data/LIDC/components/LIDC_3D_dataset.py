import os
import torch
import logging
import collections.abc
import numpy as np

from typing import Sequence, Callable, Any, Literal
from torch.utils.data import Dataset, Subset
from monai.transforms import Compose

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class LIDC_IDRI_3D_Dataset(Dataset):
    def __init__(
        self, 
        data_nodule_dir: list[str],
        data_clean_dir: list[str] = [],
        transform: Callable | None = None,
        data_type: Literal["image", "lung"] = "image"
    ) -> None:
        """
        Args:
            data_dir: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data_nodule_dir = data_nodule_dir
        self.data_clean_dir = data_clean_dir
        self.data_type = data_type
        self.data_list = self._get_file_list()
        
        try:
            self.transform = Compose(transform) if not isinstance(transform, Compose) else transform
        except Exception as e:
            raise ValueError("`transform` must be a callable or a list of callables that is Composable") from e
    
    def _get_file_list(self) -> list:
        file_list = []
        
        if self.data_type == "image":
            for dicom_path in self.data_nodule_dir:
                # Get mask path of nodule image
                mask_path = dicom_path.replace("Image", "Mask")
                mask_path = mask_path.replace("NI", "MA")

                # Check whether mask path exist
                if os.path.exists(mask_path):
                    data = {
                        'image': dicom_path,
                        'label': mask_path
                    }
                    file_list.append(data)
                else:
                    logging.error(f"Path {mask_path} does not exist.")
            
            for dicom_path in self.data_clean_dir:
                # Get mask path of nodule image
                mask_path = dicom_path.replace("Image", "Mask")
                mask_path = mask_path.replace("CN", "CM")

                # Check whether mask path exist
                if os.path.exists(mask_path):
                    data = {
                        'image': dicom_path,
                        'label': mask_path
                    }
                    file_list.append(data)
                else:
                    logging.error(f"Path {mask_path} does not exist.")
        else:
            for dicom_path in self.data_nodule_dir:
                # Get mask path of nodule image
                mask_path = dicom_path.replace("Lung_Segmentation_v2", "Mask")
                mask_path = mask_path.replace("NI", "MA")

                # Nodule Mask
                if os.path.exists(mask_path):
                    data = {
                        'image': dicom_path,
                        'label': mask_path
                    }
                    file_list.append(data)
                else:
                    # Clean Mask
                    mask_path = mask_path.replace("Mask", "Clean/Mask")
                    mask_path = mask_path.replace("CN", "CM")
                    
                    data = {
                        'image': dicom_path,
                        'label': mask_path
                    }
                    file_list.append(data)
            
        # Seed
        # np.random.seed(42)
        # file_list = np.random.permutation(file_list)

        return file_list
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data_list[index]
        # return apply_transform(self.transform, data_i) if self.transform is not None else data_i
        return self.transform(data_i)
    

    def __getitem__(self, index: int | slice | Sequence[int]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)
    