from typing import Any, Dict, Optional, Tuple

import os
import logging
import torch
import numpy as np
from typing import Literal
from lightning import LightningDataModule
from torch.utils.data import Dataset
from monai.transforms import Compose
from monai.data import DataLoader
from src.data.LIDC.components.LIDC_3D_vista import LIDC_IDRI_3D_Vista
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class LIDC_IDRI_3D_Vista_Datamodule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_nodule_dir: str = "/data/hpc/dqm/lidc-preprocessing/data/Image",
        data_lung_dir: str = "/data/hpc/dqm/lidc-preprocessing/data/Lung_Segmentation_v2",
        data_clean_dir: str = "/data/hpc/dqm/lidc-preprocessing/data/Clean/Image",
        train_val_test_split: Tuple[int, int, int] = (8, 1, 1),
        data_type: Literal["image", "lung"] = "image", 
        transform_train:  Optional[Compose] = None,
        transform_val:  Optional[Compose] = None,
        image_size: tuple = (128, 128, 128),
        image_size_before_resized: tuple = (128, 128, 128),
        batch_size: int = 2,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_nodule_samples: int = 2000,
        num_lung_samples: int = 2000,
        num_clean_samples: int = 1000,
        num_classes: int = 2,
        samples: int = 1,
        nodule_clean_divide: bool = True,
        shuffle_seed: bool = True
    ) -> None:
        """Initialize a `MonAIBraTsDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        
        self.shuffle_seed = shuffle_seed

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of BraTs classes (4).
        """
        return self.hparams.num_classes


    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            self.data_train, self.data_val, self.data_test = self._read_data_dir()
            
            logging.info(f"Train Dataset: {len(self.data_train)}")
            logging.info(f"Val Dataset: {len(self.data_val)}")
            logging.info(f"Test Dataset: {len(self.data_test)}")
    
    def _read_data_dir(self) -> list:
        # Get all the data files

        file_nodule_list = []
        file_lung_list = []
        file_clean_list = []
        
        if self.hparams.nodule_clean_divide:
            # Get nodule files
            for root, _, files in os.walk(self.hparams.data_nodule_dir):
                for file in files:
                    if file.endswith(".npy"):
                        dicom_path = os.path.join(root, file)
                        file_nodule_list.append(dicom_path)
            
            # Get clean files
            for root, _, files in os.walk(self.hparams.data_clean_dir):
                for file in files:
                    if file.endswith(".npy"):
                        dicom_path = os.path.join(root, file)
                        file_clean_list.append(dicom_path)
                        
            file_nodule_list = file_nodule_list[:self.hparams.num_nodule_samples] \
                if len(file_nodule_list) > self.hparams.num_nodule_samples else file_nodule_list
            file_clean_list = file_clean_list[:self.hparams.num_clean_samples] \
                if len(file_clean_list) > self.hparams.num_clean_samples else file_clean_list
            
            train_nodule_dir, val_nodule_dir, test_nodule_dir = self._split_data(
                file_paths=file_nodule_list, train_val_test_split=self.hparams.train_val_test_split)
            
            train_clean_dir, val_clean_dir, test_clean_dir = self._split_data(
                file_paths=file_clean_list, train_val_test_split=self.hparams.train_val_test_split)
            
            # Transform Train
            data_train = LIDC_IDRI_3D_Vista(
                data_nodule_dir=train_nodule_dir, 
                data_clean_dir=train_clean_dir, 
                transform=self.hparams.transform_train,
                data_type=self.hparams.data_type,
                nodule_clean_divide=self.hparams.nodule_clean_divide,
                # image_size_before_resized=self.hparams.image_size_before_resized,
                # image_size=self.hparams.image_size,
                # samples=self.hparams.samples,
            )
            # Transform Val
            data_val = LIDC_IDRI_3D_Vista(
                data_nodule_dir=val_nodule_dir, 
                data_clean_dir=val_clean_dir, 
                transform=self.hparams.transform_val,
                data_type=self.hparams.data_type,
                nodule_clean_divide=self.hparams.nodule_clean_divide,
                # image_size_before_resized=self.hparams.image_size_before_resized,
                # image_size=self.hparams.image_size,
                # samples=self.hparams.samples,
            )
            data_test = LIDC_IDRI_3D_Vista(
                data_nodule_dir=test_nodule_dir, 
                data_clean_dir=test_clean_dir, 
                transform=self.hparams.transform_val,
                data_type=self.hparams.data_type,
                nodule_clean_divide=self.hparams.nodule_clean_divide,
                # image_size_before_resized=self.hparams.image_size_before_resized,
                # image_size=self.hparams.image_size,
                # samples=self.hparams.samples,
            )
            
            return data_train, data_val, data_test
        else:
            
            # Get lung files
            for root, _, files in os.walk(self.hparams.data_lung_dir):
                for file in files:
                    if file.endswith(".npy"):
                        dicom_path = os.path.join(root, file)
                        file_lung_list.append(dicom_path)
            file_lung_list = file_lung_list[:self.hparams.num_lung_samples] \
                if len(file_lung_list) > self.hparams.num_lung_samples else file_lung_list
            train_lung_dir, val_lung_dir, test_lung_dir = self._split_data(
                file_paths=file_lung_list, train_val_test_split=self.hparams.train_val_test_split)
            
            # Transform train
            data_train = LIDC_IDRI_3D_Vista(
                data_nodule_dir=train_lung_dir, 
                data_clean_dir=[], 
                transform=self.hparams.transform_train,
                data_type=self.hparams.data_type,
                nodule_clean_divide=self.hparams.nodule_clean_divide,
                # image_size_before_resized=self.hparams.image_size_before_resized,
                # image_size=self.hparams.image_size,
                # samples=self.hparams.samples,
            )
            # Transform Val
            data_val = LIDC_IDRI_3D_Vista(
                data_nodule_dir=val_lung_dir, 
                data_clean_dir=[], 
                transform=self.hparams.transform_val,
                data_type=self.hparams.data_type,
                nodule_clean_divide=self.hparams.nodule_clean_divide,
                # image_size_before_resized=self.hparams.image_size_before_resized,
                # image_size=self.hparams.image_size,
                # samples=self.hparams.samples,
            )
            data_test = LIDC_IDRI_3D_Vista(
                data_nodule_dir=test_lung_dir, 
                data_clean_dir=[], 
                transform=self.hparams.transform_val,
                data_type=self.hparams.data_type,
                nodule_clean_divide=self.hparams.nodule_clean_divide,
                # image_size_before_resized=self.hparams.image_size_before_resized,
                # image_size=self.hparams.image_size,
                # samples=self.hparams.samples,
            )
            
            return data_train, data_val, data_test
            
    
    def _split_data(self, file_paths, train_val_test_split) -> Tuple[list, list, list]:
        if self.shuffle_seed:
            np.random.seed(42) # If needed
        
        # get len files
        num_files = len(file_paths)
        
        # ratio
        train_ratio, val_ratio, test_ratio = train_val_test_split
        
        # get num train, val, test
        num_train = int(num_files * train_ratio / (train_ratio + val_ratio + test_ratio))
        num_val = int(num_files * val_ratio / (train_ratio + val_ratio + test_ratio))
        
        # get random index
        train_paths = list(np.random.choice(file_paths, num_train, replace=False))
        val_paths = list(np.random.choice(list(set(file_paths) - set(train_paths)), num_val, replace=False))
        test_paths = list(set(file_paths) - set(train_paths) - set(val_paths))
        return train_paths, val_paths, test_paths
        

    def train_dataloader(self) -> DataLoader: # DataLoader[Any]
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader: # DataLoader[Any]
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device, # 1
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader: # DataLoader[Any]
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device, # 1
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass

