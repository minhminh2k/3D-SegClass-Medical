import os
from typing import Any

import torch

import logging
import numpy as np
import pandas as pd
import lightning as pl
import albumentations as A
import torch.nn.functional as F

from io import BytesIO
import nibabel as nib
from monai import transforms
from monai.transforms import Compose, ToTensord
from PIL import Image
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from functools import partial
from src.utils.visualization.custom_visualize import custom_visualization, mask_overlay, scale_image
from monai.inferers import sliding_window_inference
import matplotlib.colors as mcolors
import cv2
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
        n_images_to_log: int = 5,
        roi_size: tuple = [128, 128, 128],
        num_classes: int = 4,
        classes_name: tuple[str] = ["Tumor Core", "Whole Tumore", "Enhancing Tumor"],
        colors: tuple[str] = ["#007339", "#FFFF1E", "#FF0000", "#FFFFFF"],
        case_names: tuple[str] = ["00058", "00059", "00076", "00077", "00099", "00113", "00114", "00124", "00139"],
        frame_nums: tuple[int] = [100, 67, 45, 48, 89, 85, 10, 44, 84],
        infer_overlap: float = 0.6,
        number_of_logged_samples: int = 2,
    ):
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.classes_name = classes_name
        
        self.colors = colors
        
        self.data_dir = data_dir
        self.case_names = case_names
        self.frame_nums = frame_nums
        self.infer_overlap = infer_overlap
        
        self.n_images_to_log = n_images_to_log  # number of logged images in grid
        self.number_of_logged_samples = number_of_logged_samples
        
        self.n_samples_validation = []
        self.n_samples_test = []

        self.image_dict = [{
            "case": i,
            'image': [
                f'{self.data_dir}/BraTS2021_{i}/BraTS2021_{i}_flair.nii.gz',
                f'{self.data_dir}/BraTS2021_{i}/BraTS2021_{i}_t1ce.nii.gz',
                f'{self.data_dir}/BraTS2021_{i}/BraTS2021_{i}_t1.nii.gz',
                f'{self.data_dir}/BraTS2021_{i}/BraTS2021_{i}_t2.nii.gz'
            ],
            'label': f'{self.data_dir}/BraTS2021_{i}/BraTS2021_{i}_seg.nii.gz'
        } for i in self.case_names]

        self.transform = Compose(
            [
                transforms.LoadImaged(keys=["image", "label"]),
                transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=self.roi_size),
                transforms.RandSpatialCropd(keys=["image", "label"], roi_size=self.roi_size, random_size=False),
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )
        
        # model_inferer_test = partial(
        #     sliding_window_inference,
        #     roi_size=self.roi,
        #     sw_batch_size=1,
        #     predictor=model,
        #     overlap=infer_overlap,
        # )

        
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        wandb_logger = trainer.logger
        
        # First sample in case names
        sample = nib.load(self.image_dict[0]["image"][0]).get_fdata()[:, :, self.frame_nums[0]]
        label = nib.load(self.image_dict[0]["label"]).get_fdata()[:, :, self.frame_nums[0]]
        
        viz_sample = mask_overlay(sample, label)
                
        wandb_logger.log_image(
            key="Ground truth sample",
            images=[viz_sample],
        )

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        for i in range(self.n_images_to_log):
            transformed = self.transform(self.image_dict[i])
            image = transformed["image"] # [4, 128, 128, 128]
            label = transformed["label"] # [3, 128, 128, 128]
            
            image = image.unsqueeze(0).to(trainer.model.device)
            
            prob = torch.sigmoid(trainer.model(image)) # [1, 3, 128, 128, 128]
            seg = prob[0].detach().cpu().numpy() # [3, 128, 128, 128]

            seg = (seg > 0.5).astype(np.int8)
                        
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3])) # [128, 128, 128]
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 3
                        
            label_out = np.zeros((label.shape[1], label.shape[2], label.shape[3])) # [128, 128, 128]
            label_out[label[1] == 1] = 2
            label_out[label[0] == 1] = 1
            label_out[label[2] == 1] = 3
                        
            image = image.squeeze(0)
            buffer = custom_visualization(
                image=image[1],
                label=label_out,
                pred=seg_out,
                num_classes=self.num_classes,
                frame_nums=self.frame_nums,
                class_names=self.classes_name,
                colors=self.colors,
                fig_size=[4, 2],
            )
            
            case = self.image_dict[i]["case"]
            trainer.logger.log_image(
                key=f"predicted mask (training) - Case {case}",
                images=[Image.open(buffer)],
            )
            buffer.close()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        
        if len(self.n_samples_validation) < self.number_of_logged_samples:
            for i, l, o in zip(batch["image"], batch["label"], outputs["preds"]):
                self.n_samples_validation.append({
                    "image": i.squeeze(0).cpu(),
                    "label": l.squeeze(0).cpu(),
                    "preds": o.squeeze(0)
                })
            
        self.n_samples_validation = self.n_samples_validation[:self.number_of_logged_samples]

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        with torch.no_grad():
            for batch in self.n_samples_validation:
                image = batch["image"] # [4, 128, 128, 128]
                label = batch["label"] # [3, 128, 128, 128]
                
                label_out = np.zeros((label.shape[1], label.shape[2], label.shape[3])) # [128, 128, 128]
                label_out[label[1] == 1] = 2
                label_out[label[0] == 1] = 1
                label_out[label[2] == 1] = 3
                
                # Affine Transform
                # affine = batch["image_meta_dict"]["original_affine"][0].numpy()
                # num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
                # img_name = "BraTS2021_" + num + ".nii.gz"
                
                prob = torch.sigmoid(batch["preds"].unsqueeze(0))
                seg = prob[0].cpu().numpy()
                seg = (seg > 0.5).astype(np.int8)
                logging.info(f"ajnidauahfiuehfiuheuifeuif: {seg.shape}")
                seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
                seg_out[seg[1] == 1] = 2
                seg_out[seg[0] == 1] = 1
                seg_out[seg[2] == 1] = 3
                
                # Save to the directory
                # nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))
                
                buffer = custom_visualization(
                    image=image[1],
                    label=label_out,
                    pred=seg_out,
                    num_classes=self.num_classes,
                    frame_nums=self.frame_nums,
                    class_names=self.classes_name,
                    colors=self.colors,
                    fig_size=[4, 2],
                )
                
                trainer.logger.log_image(
                    key=f"predicted mask (validation)",
                    images=[Image.open(buffer)],
                )
                buffer.close()
            
        self.n_samples_validation.clear()

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        
        if len(self.n_samples_test) < self.number_of_logged_samples:
            for i, l, o in zip(batch["image"], batch["label"], outputs["preds"]):
                self.n_samples_test.append({
                    "image": i.squeeze(0).cpu(),
                    "label": l.squeeze(0).cpu(),  
                    "preds": o.squeeze(0)
                })
            
        self.n_samples_test = self.n_samples_test[:self.number_of_logged_samples]

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        with torch.no_grad():
            for batch in self.n_samples_test:
                image = batch["image"] # [4, 128, 128, 128]
                label = batch["label"] # [3, 128, 128, 128]
                
                label_out = np.zeros((label.shape[1], label.shape[2], label.shape[3])) # [128, 128, 128]
                label_out[label[1] == 1] = 2
                label_out[label[0] == 1] = 1
                label_out[label[2] == 1] = 3
                
                # Affine Transform
                # affine = batch["image_meta_dict"]["original_affine"][0].numpy()
                # num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
                # img_name = "BraTS2021_" + num + ".nii.gz"
                
                prob = torch.sigmoid(batch["preds"].unsqueeze(0))
                seg = prob[0].cpu().numpy()
                seg = (seg > 0.5).astype(np.int8)
                seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
                seg_out[seg[1] == 1] = 2
                seg_out[seg[0] == 1] = 1
                seg_out[seg[2] == 1] = 3
                
                # Save to the directory
                # nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))
                
                buffer = custom_visualization(
                    image=image[1],
                    label=label_out,
                    pred=seg_out,
                    num_classes=self.num_classes,
                    frame_nums=self.frame_nums,
                    class_names=self.classes_name,
                    colors=self.colors,
                    fig_size=[4, 2],
                )
                
                trainer.logger.log_image(
                    key=f"predicted mask (testing)",
                    images=[Image.open(buffer)],
                )
                buffer.close()
            
        self.n_samples_test.clear()