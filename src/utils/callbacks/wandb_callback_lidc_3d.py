import os
from typing import Any

import random
import torch
import wandb
import logging
import numpy as np
import lightning as pl
import torch.nn.functional as F
from io import BytesIO
import nibabel as nib
from monai import transforms
from monai.transforms import Compose
from PIL import Image
from lightning.pytorch.callbacks import Callback
from torchvision.utils import make_grid
from src.utils.visualization.custom_visualize import gif_visualization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class LIDC_3D_Callback(Callback):
    def __init__(
        self,
        data_nodule_dir: str = "/data/hpc/dqm/lidc-preprocessing/data/Image",
        data_clean_dir: str = "/data/hpc/dqm/lidc-preprocessing/data/Clean/Image",
        n_images_to_log: int = 5,
        roi_size: tuple = [128, 128, 128],
        num_classes: int = 2,
        classes_name: tuple[str] = ["Lung Nodule"],
        colors: tuple[str] = ["#00FF00", "#FFFF1E"],
        case_names: tuple[str] = [ "0068", "0027", "0050", "0061", "0074", "0101", "0117"],
        number_of_logged_samples: int = 2,
    ):
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.classes_name = classes_name
        self.colors = colors
        
        self.data_nodule_dir = data_nodule_dir
        self.data_clean_dir = data_clean_dir
        self.case_names = case_names
        
        self.n_images_to_log = n_images_to_log
        self.number_of_logged_samples = number_of_logged_samples
        
        self.n_samples_validation = []
        self.n_samples_test = []

        self.image_dict = [{
            "case": i,
            'image': f'{self.data_nodule_dir}/LIDC-IDRI-{i}/{i}_NI001.npy',
            'label': f'{self.data_nodule_dir}/LIDC-IDRI-{i}/{i}_MA001.npy'.replace('Image', 'Mask')
        } for i in self.case_names]
        
        self.image_dict = random.sample(self.image_dict, len(self.image_dict))

        self.transform = Compose(
            [
                transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                # transforms.NormalizeIntensityd(keys="image"),
                transforms.Resized(
                    keys=["image", "label"],
                    spatial_size=self.roi_size,
                    mode=("trilinear", "nearest"),  # Image: trilinear, Label: nearest-neighbor
                ),
                transforms.ScaleIntensityRanged(keys=["image"], a_min=-1200, a_max=800, b_min=0.0, b_max=1.0, clip=True),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
        
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # First sample in case names
        image = np.load(self.image_dict[0]["image"])
        label = np.load(self.image_dict[0]["label"])
        
        volume = gif_visualization(
            image=image,
            label=label,
            num_classes=self.num_classes,
            colors=self.colors,
            transparency=0.4
        )
        
        frames = [Image.fromarray(volume[i]) for i in range(len(volume))]
        gif_buffer = BytesIO()
        frames[0].save(gif_buffer, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
        gif_buffer.seek(0)
        
        # trainer.logger.log_image(
        #     key=f"Ground truth sample",
        #     images=[wandb.Video(gif_buffer, format="gif")],
        # )
        wandb.log({"Ground truth sample": wandb.Video(gif_buffer, format="gif")})
        
        gif_buffer.close()

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):

        for i in range(self.n_images_to_log):
            transformed = self.transform(self.image_dict[i])
            image = transformed["image"] # [1, 128, 128, 128]
            label = transformed["label"] # [1, 128, 128, 128]
            
            image = image.unsqueeze(0).to(trainer.model.device)
             
            prob = torch.sigmoid(trainer.model(image)) # [1, 1, 128, 128, 128]
            
            seg = prob[0].detach().cpu().numpy() # [1, 128, 128, 128]

            seg = (seg > 0.5).astype(np.int8)
                        
            image = image.squeeze(0)            
            gif_buffer = self.visualization_process(
                    image=image.squeeze(0), # [128, 128, 128]
                    label=label.squeeze(0),
                    pred=seg.squeeze(0)
                )
            
            case = self.image_dict[i]["case"]            
            # trainer.logger.log_image(
            #     key=f"predicted mask (training) - Case {case}",
            #     images=[wandb.Video(gif_buffer, format="gif")],
            # )
            wandb.log({f"predicted mask (training) - Case {case}": wandb.Video(gif_buffer, format="gif")})
            
            gif_buffer.close()

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
                image = batch["image"] # [128, 128, 128]
                label = batch["label"] # [128, 128, 128]                
                prob = torch.sigmoid(batch["preds"].unsqueeze(0))
                seg = prob[0].cpu().numpy()
                seg = (seg > 0.5).astype(np.int8)
                
                gif_buffer = self.visualization_process(
                    image=image,
                    label=label.numpy().astype(np.int8),
                    pred=seg
                )
                                    
                # trainer.logger.log_image(
                #     key=f"predicted mask (validation)",
                #     images=[wandb.Video(gif_buffer, format="gif")],
                # )
                wandb.log({f"predicted mask (validation)": wandb.Video(gif_buffer, format="gif")})
                gif_buffer.close()
            
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
                image = batch["image"] # [128, 128, 128]
                label = batch["label"] # [128, 128, 128]
                                
                prob = torch.sigmoid(batch["preds"].unsqueeze(0))
                seg = prob[0].cpu().numpy()
                seg = (seg > 0.5).astype(np.int8)
                
                gif_buffer = self.visualization_process(
                    image=image,
                    label=label.numpy().astype(np.int8),
                    pred=seg
                )
                                    
                # trainer.logger.log_image(
                #     key=f"predicted mask (testing)",
                #     images=[wandb.Video(gif_buffer, format="gif")],
                # )
                
                wandb.log({f"predicted mask (testing)": wandb.Video(gif_buffer, format="gif")})
                gif_buffer.close()
            
        self.n_samples_test.clear()
        
    def visualization_process(self, image, label, pred):
        volume_label = gif_visualization(
            image=image,
            label=label,
            num_classes=self.num_classes,
            colors=self.colors,
            transparency=0.4
        )
        
        volume_pred = gif_visualization(
            image=image,
            label=pred,
            num_classes=self.num_classes,
            colors=["#FFFF1E", "#00FF00", "#FF0000"],
            transparency=0.4,
        )
    
        # Stack frames
        frames = []
        for i in range(len(volume_label)): # first axis
            slice_label = volume_label[i] # , :, :]
            slice_pred = volume_pred[i] # , :, :]
            
            combined_slice = np.hstack([slice_label, slice_pred])
            
            img = Image.fromarray(combined_slice)
            
            frames.append(img)

        gif_buffer = BytesIO()
        frames[0].save(gif_buffer, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
        gif_buffer.seek(0)
        
        return gif_buffer