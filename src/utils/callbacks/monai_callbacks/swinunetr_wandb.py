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
        classes_name: list[str] = ["Whole Tumor", "Enhancing Tumore", "Tumor Core"],
        colors: list[str] = ["#FFFF1E", "#007339", "#FF0000", "#FF0000"],
        case_names: list[str] = ["00058", "00059", "00076", "00077", "00099", "00113", "00114", "00124", "00139"],
        frame_nums: list[str] = [67, 60, 45, 48, 89, 85, 10, 44, 84],
    ):
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.classes_name = classes_name
        self.colors = colors
        self.data_dir = data_dir
        self.case_names = case_names
        self.frame_nums = frame_nums
        
        self.n_images_to_log = n_images_to_log  # number of logged images when eval
        self.n_first_preds = []
        self.n_first_targets = []
        self.n_first_batch = []
        self.n_first_image = []
        self.show_pred = []
        self.show_target = []

        self.image_dict = [{
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
                transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ]
        )

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
            model_inferer = partial(
                roi_size=self.roi_size,
                sw_batch_size=1,
                predictor=trainer.model,
                overlap=0.6,
            )
            image = transformed["image"]
            label = transformed["label"]
            image = image.unsqueeze(0).to(trainer.model.device)

            prob = torch.sigmoid(model_inferer(image))
            seg = prob[0].detach().cpu().numpy()
            seg = (seg > 0.5).astype(np.int8)
            seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            seg_out[seg[1] == 1] = 2
            seg_out[seg[0] == 1] = 1
            seg_out[seg[2] == 1] = 4
            
            wandb_logger = trainer.logger
            
            legend_ax = custom_visualization(
                image=image,
                label=label,
                pred=seg_out,
                num_classes=self.num_classes,
                class_names=self.classes_name,
                colors=self.colors,
                logger=wandb_logger,
                fig_size=[1, 2],
            )
            
            trainer.logger.log_image(
                key="predicted mask (training)",
                images=[legend_ax],
                caption="Visualization",
            )

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

        # with torch.no_grad():
        #     for i, batch in enumerate(test_loader):
        #         image = batch_data["image"].cuda()
        #         # affine = batch["image_meta_dict"]["original_affine"][0].numpy()
        #         # num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
        #         # img_name = "BraTS2021_" + num + ".nii.gz"
        #         prob = torch.sigmoid(model_inferer_test(image))
        #         seg = prob[0].detach().cpu().numpy()
        #         seg = (seg > 0.5).astype(np.int8)
        #         seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
        #         seg_out[seg[1] == 1] = 2
        #         seg_out[seg[0] == 1] = 1
        #         seg_out[seg[2] == 1] = 4
        #         # nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), affine), os.path.join(output_directory, img_name))
            
        # self.four_first_preds.clear()
        # self.four_first_targets.clear()
        # self.four_first_batch.clear()
        # self.four_first_image.clear()
        self.show_pred.clear()
        self.show_target.clear()
        



# class WandbCallback(Callback):
#     def __init__(
#         self,
#         image_id: str = "003b48a9e.jpg",
#         data_path: str = "data/airbus",
#         n_images_to_log: int = 5,
#         img_size: int = 384,
#     ):
#         self.img_size = img_size
#         self.n_images_to_log = n_images_to_log  # number of logged images when eval

#         self.four_first_preds = []
#         self.four_first_targets = []
#         self.four_first_batch = []
#         self.four_first_image = []
#         self.show_pred = []
#         self.show_target = []

#         self.batch_size = 1
#         self.num_samples = 8
#         self.num_batch = 0

#         image_path = os.path.join(data_path, "train_v2")
#         image_path = os.path.join(image_path, image_id)

#         self.sample_image = np.array(Image.open(image_path).convert("RGB"))
#         self.sample_image_height, self.sample_image_width = (
#             self.sample_image.shape[0],
#             self.sample_image.shape[1],
#         )
#         dataframe = pd.read_csv(os.path.join(data_path, "train_ship_segmentations_v2.csv"))
#         self.sample_mask = dataframe[dataframe["ImageId"] == image_id]["EncodedPixels"]
#         self.sample_mask = masks_as_image(self.sample_mask)

#         self.transform = Compose(
#             [
#                 A.Resize(self.img_size, self.img_size),
#                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ToTensorV2(),
#             ]
#         )

#     def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
#         wandb_logger = trainer.logger
#         wandb_logger.log_image(
#             key="real mask",
#             images=[Image.fromarray(mask_overlay(self.sample_image, self.sample_mask))],
#         )

#     def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
#         transformed = self.transform(image=self.sample_image)
#         image = transformed["image"]  # (3, img_size, img_size)
#         image = image.unsqueeze(0).to(trainer.model.device)  # (1, 3, img_size, img_size)

#         pred_mask = trainer.model(image)
#         pred_mask = pred_mask.detach()  # (1, 1, img_size, img_size)
#         pred_mask = torch.sigmoid(pred_mask)
#         pred_mask = pred_mask >= 0.5
#         pred_mask = pred_mask.squeeze(0)
#         pred_mask = pred_mask.permute(1, 2, 0)
#         pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
#         pred_mask = cv2.resize(
#             pred_mask,
#             (self.sample_image_width, self.sample_image_height),
#             interpolation=cv2.INTER_CUBIC,
#         )

#         wandb_logger = trainer.logger
#         wandb_logger.log_image(
#             key="predicted mask",
#             images=[Image.fromarray(mask_overlay(self.sample_image, pred_mask))],
#         )

#     def on_validation_batch_end(
#         self,
#         trainer: "pl.Trainer",
#         pl_module: "pl.LightningModule",
#         outputs,
#         batch: Any,
#         batch_idx: int,
#         dataloader_idx: int = 0,
#     ) -> None:
#         preds = outputs["preds"]
#         targets = outputs["targets"]
#         self.batch_size = preds.shape[0]
#         self.num_batch = self.num_samples / self.batch_size

#         if len(self.four_first_batch) < self.num_batch:
#             self.four_first_batch.append(batch)

#         n = int(self.num_batch * self.batch_size)
#         self.four_first_preds.extend(preds[:n])
#         self.four_first_targets.extend(targets[:n])

#     def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
#         IMG_MEAN = [0.485, 0.456, 0.406]
#         IMG_STD = [0.229, 0.224, 0.225]

#         def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
#             # 3, H, W, B
#             ten = x.clone().permute(1, 2, 3, 0)
#             for t, m, s in zip(ten, mean, std):
#                 t.mul_(s).add_(m)
#             # B, 3, H, W
#             return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

#         # chinh image ve (768, 768, 3)
#         for i, batch in enumerate(self.four_first_batch):
#             (
#                 image_batch,
#                 mask,
#                 label,
#                 file_id,
#             ) = batch

#             # image.shape = (b, 3, h, w)
#             images = torch.split(image_batch, 1, dim=0)

#             for j in range(self.batch_size):
#                 image = images[j]
#                 image = denormalize(image)
#                 image = image.squeeze()  # (3, 768, 768)
#                 image = image.cpu().numpy()
#                 image = (image * 255).astype(np.uint8)
#                 image = np.transpose(image, (1, 2, 0))

#                 pred = self.four_first_preds[i * self.batch_size + j]
#                 pred = pred.unsqueeze(0)
#                 pred = pred.cpu().numpy().astype(np.uint8)
#                 log_pred = mask_overlay(image, pred)
#                 log_pred = np.transpose(log_pred, (2, 0, 1))
#                 log_pred = torch.from_numpy(log_pred)
#                 self.show_pred.append(log_pred)

#                 target = self.four_first_targets[i * self.batch_size + j]
#                 target = target.unsqueeze(0)
#                 target = target.cpu().numpy().astype(np.uint8)
#                 log_target = mask_overlay(image, target)
#                 log_target = np.transpose(log_target, (2, 0, 1))
#                 log_target = torch.from_numpy(log_target)
#                 self.show_target.append(log_target)

#         stack_pred = torch.stack(self.show_pred)
#         stack_target = torch.stack(self.show_target)

#         grid_pred = make_grid(stack_pred, nrow=4)
#         grid_target = make_grid(stack_target, nrow=4)

#         grid_pred_np = grid_pred.numpy().transpose(1, 2, 0)
#         grid_target_np = grid_target.numpy().transpose(1, 2, 0)

#         grid_pred_np = Image.fromarray(grid_pred_np)
#         grid_target_np = Image.fromarray(grid_target_np)

#         wandb_logger = trainer.logger
#         wandb_logger.log_image(key="predicted mask", images=[grid_pred_np, grid_target_np])

#         self.four_first_preds.clear()
#         self.four_first_targets.clear()
#         self.four_first_batch.clear()
#         self.four_first_image.clear()
#         self.show_pred.clear()
#         self.show_target.clear()

#     def on_test_batch_end(
#         self,
#         trainer: pl.Trainer,
#         pl_module: pl.LightningModule,
#         outputs,
#         batch: Any,
#         batch_idx: int,
#         dataloader_idx: int,
#     ) -> None:
#         if self.n_images_to_log <= 0:
#             return

#         IMG_MEAN = [0.485, 0.456, 0.406]
#         IMG_STD = [0.229, 0.224, 0.225]
#         logger = trainer.logger

#         def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
#             # 3, H, W, B
#             ten = x.clone().permute(1, 2, 3, 0)
#             for t, m, s in zip(ten, mean, std):
#                 t.mul_(s).add_(m)
#             # B, 3, H, W
#             return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)

#         preds = outputs["preds"]
#         targets = outputs["targets"]
#         images, ys, labels, ids = batch

#         images = denormalize(images)
#         for img, pred, target, id in zip(images, preds, targets, ids):
#             if self.n_images_to_log <= 0:
#                 break

#             img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#             pred = torch.sigmoid(pred)
#             pred = pred >= 0.5
#             pred = pred.cpu().numpy().astype(np.uint8)
#             target = target.cpu().numpy().astype(np.uint8)

#             log_pred = mask_overlay(img, pred)
#             log_target = mask_overlay(img, target)

#             log_img = Image.fromarray(img)
#             log_pred = Image.fromarray(log_pred)
#             log_target = Image.fromarray(log_target)

#             logger.log_image(
#                 key="Sample",
#                 images=[log_img, log_pred, log_target],
#                 caption=[id + "-Real", id + "-Predict", id + "-GroundTruth"],
#             )

#             self.n_images_to_log -= 1