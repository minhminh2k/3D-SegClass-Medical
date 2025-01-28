import os
from typing import Any

import torch
import random
import math
import logging
import numpy as np
import pandas as pd
import lightning as pl
import albumentations as A
import torch.nn.functional as F

from io import BytesIO
import nibabel as nib
from monai import transforms
from monai.transforms import Compose
from PIL import Image
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from functools import partial
from src.utils.visualization.custom_visualize import custom_visualization, mask_overlay, scale_image
from monai.inferers import sliding_window_inference

# References: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class SwinUNETRCallback(Callback):
    def __init__(
        self,
        data_dir: str= '/data/hpc/dqm/data/brats-2021/TrainingData',
        case_num: str = "00001",
        n_images_to_log: int = 5,
        slice_num: int = 67,
        roi_size: tuple = [128, 128, 128],
        num_classes: int = 4,
        classes_name: list[str] = ["Whole Tumor", "Enhancing Tumore", "Tumor Core"],
        colors: list[str] = ["#FFFF1E", "#007339", "#FF0000", "#FF0000"] 
    ):
        self.num_classes = num_classes
        self.slice_num = slice_num
        self.n_images_to_log = n_images_to_log  # number of logged images when eval
        self.roi_size = roi_size
        self.classes_name = classes_name
        self.colors = colors

        self.four_first_preds = []
        self.four_first_targets = []
        self.four_first_batch = []
        self.four_first_image = []
        self.show_pred = []
        self.show_target = []

        self.batch_size = 1
        self.num_samples = 8
        self.num_batch = 2

        self.data_dir = data_dir
        self.case_num = case_num

        self.image_dict = {
            'image': [
                f'{self.data_dir}/BraTS2021_{self.case_num}/BraTS2021_{self.case_num}_flair.nii.gz',
                f'{self.data_dir}/BraTS2021_{self.case_num}/BraTS2021_{self.case_num}_t1ce.nii.gz',
                f'{self.data_dir}/BraTS2021_{self.case_num}/BraTS2021_{self.case_num}_t1.nii.gz',
                f'{self.data_dir}/BraTS2021_{self.case_num}/BraTS2021_{self.case_num}_t2.nii.gz'
            ],
            'label': f'{self.data_dir}/BraTS2021_{self.case_num}/BraTS2021_{self.case_num}_seg.nii.gz'
        }

        self.transform = Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        wandb_logger = trainer.logger
        
        sample = nib.load(self.image_dict["image"][0]).get_fdata()[:, :, self.slice_num]
        label = nib.load(self.image_dict["label"]).get_fdata()[:, :, self.slice_num]
        
        viz_sample = mask_overlay(sample, label)
                
        wandb_logger.log_image(
            key="Ground truth sample",
            images=[viz_sample],
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # transformed = self.transform(self.image_dict)
        # model_inferer = partial(
        #     roi_size=self.roi_size,
        #     sw_batch_size=1,
        #     predictor=trainer.model,
        #     overlap=0.6,
        # )
        # image = transformed["image"]
        # label = transformed["label"]
        # image = image.unsqueeze(0).to(trainer.model.device)

        # prob = torch.sigmoid(model_inferer(image))
        # seg = prob[0].detach().cpu().numpy()
        # seg = (seg > 0.5).astype(np.int8)
        # seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        # seg_out[seg[1] == 1] = 2
        # seg_out[seg[0] == 1] = 1
        # seg_out[seg[2] == 1] = 4
        
        # wandb_logger = trainer.logger
        
        # custom_visualization(
        #     image=image,
        #     label=label,
        #     pred=seg_out,
        #     num_classes=self.num_classes,
        #     class_names=self.classes_name,
        #     colors=self.colors,
        #     logger=wandb_logger,
        #     fig_size=[1, 2],
        #     key="predicted mask (training)"
        # )
        pass

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        # preds = outputs["preds"]
        # targets = outputs["targets"]
        # logging.info(f"Callback Batch: Len Batch: {len(batch)}, Batch: {batch}")
        # logging.info(f"Callback: Batch size: {preds.shape[0]}, Pred {preds.shape}, Label {targets.shape}")
        # self.batch_size = preds.shape[0]

        # if len(self.four_first_batch) < 4:
        #     self.four_first_batch.extend(batch)
        #     self.four_first_preds.extend(preds[:n])
        #     self.four_first_targets.extend(targets[:n])
        pass
            

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        self.four_first_preds.clear()
        self.four_first_targets.clear()
        self.four_first_batch.clear()
        self.four_first_image.clear()
        self.show_pred.clear()
        self.show_target.clear()