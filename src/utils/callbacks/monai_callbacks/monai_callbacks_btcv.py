import os
from typing import Any

import torch

import logging
import numpy as np
import pandas as pd
import lightning as pl

import nibabel as nib
from monai import transforms
from monai.transforms import Compose, RandCropByPosNegLabeld
from PIL import Image
from lightning.pytorch.callbacks import Callback
from src.utils.visualization.custom_visualize import custom_visualization, mask_overlay, resample_3d

# References: https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/unetr_btcv_segmentation_3d_lightning.ipynb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class MonAICallbackBTCV(Callback):
    def __init__(
        self,
        data_dir: str= '/data/hpc/dqm/data/BTCV',
        n_images_to_log: int = 5,
        roi_size: tuple = [96, 96, 96],
        num_classes: int = 14,
        classes_name: tuple[str] = [],
        colors: tuple[str] = [],
        case_names: tuple[str] = [],
        frame_nums: tuple[int] = [],
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
        self.transform = Compose(
            [
                transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                transforms.ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
                transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
                transforms.Spacingd(keys=["image", "label"], pixdim=[1.5, 1.5, 2.0], mode=["bilinear", "nearest"]),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"], label_key="label", 
                    spatial_size=roi_size, pos=1, neg=1, 
                    num_samples=1, image_key="image", image_threshold=0),
                transforms.ToTensord(keys=["image", "label"]),
            ]
        )

        self.image_dict = [{
            "case": i,
            'image': f'{self.data_dir}/imagesTr/img{i}.nii.gz',
            'label': f'{self.data_dir}/labelsTr/label{i}.nii.gz'
        } for i in self.case_names]
        
        self.transformed_sample = [self.transform(i) for i in self.image_dict]

        
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # First sample in case names
        image = nib.load(self.image_dict[0]["image"]).get_fdata()
        label = nib.load(self.image_dict[0]["label"]).get_fdata()
                        
        buffer = custom_visualization(
            image=image,
            label=label,
            pred=label,
            num_classes=self.num_classes,
            frame_nums=self.frame_nums,
            class_names=self.classes_name,
            colors=self.colors,
            fig_size=[4, 1],
            vis_pred=False
        )
        
        trainer.logger.log_image(
            key="Ground truth sample",
            images=[Image.open(buffer)],
        )
        buffer.close()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        for i in range(self.n_images_to_log):
            transformed = self.transformed_sample[i]
            image = transformed[0]["image"] # [1, 96, 96, 96]
            label = transformed[0]["label"] # [1, 96, 96, 96]
            image = torch.unsqueeze(image, 1).to(trainer.model.device) # [1, 1, 96, 96, 96]
            
            preds = trainer.model(image) # [1, 14, 96, 96, 96]
            prob = torch.softmax(preds, 1).detach().cpu().numpy()
            prob = np.argmax(prob, axis=1).astype(np.uint8)          
            label = label.cpu().numpy()[0, :, :, :]
            
            buffer = custom_visualization(
                image=image[0][0],
                label=label,
                pred=prob[0],
                num_classes=self.num_classes,
                frame_nums=self.frame_nums,
                class_names=self.classes_name,
                colors=self.colors,
                fig_size=[4, 2],
                vis_pred=True
            )
            
            case = self.transformed_sample[i][0]["case"]
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
                    "image": i.cpu(),
                    "label": l.cpu(),
                    "preds": o
                })
            
        self.n_samples_validation = self.n_samples_validation[:self.number_of_logged_samples]

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        with torch.no_grad():
            for batch in self.n_samples_validation:
                image = batch["image"] # [1, 96, 96, 96]
                label = batch["label"] # [1, 96, 96, 96]
                # original_affine = batch["label_meta_dict"]["affine"][0].numpy()                
                _, h, w, d = label.shape
                target_shape = (h, w, d)

                preds = batch["preds"] # [14, 96, 96, 96]
                prob = torch.softmax(preds, 0).cpu().numpy()
                prob = np.argmax(prob, axis=0).astype(np.uint8)                
                prob = resample_3d(prob, target_shape) # [96, 96, 96]
                label = label.cpu().numpy()[0, :, :, :]
                
                # Save to the directory
                # nib.save(
                #     nib.Nifti1Image(prob.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
                # )   
                
                buffer = custom_visualization(
                    image=image[0],
                    label=label,
                    pred=prob,
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
                    "image": i.cpu(),
                    "label": l.cpu(),  
                    "preds": o
                })
            
        self.n_samples_test = self.n_samples_test[:self.number_of_logged_samples]

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        with torch.no_grad():
            for batch in self.n_samples_test:
                image = batch["image"] # [1, 96, 96, 96]
                label = batch["label"] # [1, 96, 96, 96]
                # original_affine = batch["label_meta_dict"]["affine"][0].numpy()                
                _, h, w, d = label.shape
                target_shape = (h, w, d)

                preds = batch["preds"] # [14, 96, 96, 96]
                prob = torch.softmax(preds, 0).cpu().numpy()
                prob = np.argmax(prob, axis=0).astype(np.uint8)                
                prob = resample_3d(prob, target_shape)
                logging.info(f"Prob shape {prob.shape},")

                label = label.cpu().numpy()[0, :, :, :]
                
                # Save to the directory
                # nib.save(nib.Nifti1Image(prob.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)   
                
                buffer = custom_visualization(
                    image=image[0],
                    label=label,
                    pred=prob,
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