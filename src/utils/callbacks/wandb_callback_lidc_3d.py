import os
from typing import Any

import random
import torch
import wandb
import logging
import numpy as np
import lightning as pl
from io import BytesIO
from typing import Literal, Union
from monai import transforms
from monai.transforms import Compose
from PIL import Image
from lightning.pytorch.callbacks import Callback
from src.utils.visualization.custom_visualize import gif_visualization
from torchvision.utils import make_grid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class LIDC_3D_Callback(Callback):
    def __init__(
        self,
        data_nodule_dir: str = "/data/hpc/dqm/3D-SegClass-Medical/data/Image",
        data_lung_dir: str = "/data/hpc/dqm/3D-SegClass-Medical/data/Lung_Segmentation_v2",
        data_clean_dir: str = "/data/hpc/dqm/3D-SegClass-Medical/data/Clean/Image",
        data_type: Literal["image", 'lung'] = "image",
        n_images_to_log: int = 5,
        roi_size: tuple = [128, 128, 128],
        image_size_before_resized: tuple = [128, 256, 256],
        num_classes: int = 2,
        classes_name: tuple[str] = ["Lung Nodule"],
        colors: tuple[str] = ["#00FF00", "#FFFF1E"],
        case_names: tuple[str] = [ "0068", "0027", "0050", "0061", "0074", "0101", "0117"],
        number_of_logged_samples: int = 2,
        do_ds: bool = False,
        coefficient: int = 5
    ):
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.image_size_before_resized = image_size_before_resized
        self.classes_name = classes_name
        self.colors = colors
        self.do_ds = do_ds
        
        self.data_nodule_dir = data_nodule_dir
        self.data_clean_dir = data_clean_dir
        self.data_lung_dir = data_lung_dir
        self.case_names = case_names
        self.data_type = data_type
        
        self.n_images_to_log = n_images_to_log
        self.number_of_logged_samples = number_of_logged_samples
        self.coefficient = coefficient
        
        self.n_samples_validation = []
        self.n_samples_test = []

        self.image_dict = []
        
        for i in self.case_names:
            if self.data_type == "image":
                self.image_dict.append(
                    {
                        "case": i,
                        'image': f'{self.data_nodule_dir}/LIDC-IDRI-{i}/{i}_NI001.npy',
                        'label': f'{self.data_nodule_dir}/LIDC-IDRI-{i}/{i}_MA001.npy'.replace('Image', 'Mask')
                    }
                )
            else:
                self.image_dict.append(
                    {
                        "case": i,
                        'image': f'{self.data_lung_dir}/LIDC-IDRI-{i}/{i}_NI001.npy',
                        'label': f'{self.data_lung_dir}/LIDC-IDRI-{i}/{i}_MA001.npy'.replace('Lung_Segmentation_v2', 'Mask')
                    }
                )
        
        self.image_dict = random.sample(self.image_dict, len(self.image_dict))

        self.transform = Compose(
            [
                transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
                transforms.RandCropByPosNegLabeld(
                    keys=["image", "label"], 
                    label_key="label",
                    spatial_size=self.image_size_before_resized,
                    pos=1,
                    neg=0,
                    num_samples=1, # number of cropped samples
                    image_key="image",
                ),
                # transforms.Resized(
                #     keys=["image", "label"],
                #     spatial_size=self.roi_size,
                #     mode=("trilinear", "nearest"),  # Image: trilinear, Label: nearest-neighbor
                # ),
                transforms.ToTensord(keys=["image", "label"])
            ]
        )
        
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # # First sample in case names
        # image = np.load(self.image_dict[0]["image"])
        # label = np.load(self.image_dict[0]["label"])
        
        # # Clipped
        # image[image > self.max_px] = self.max_px
        # image[image < self.min_px] = self.min_px
        
        # volume = gif_visualization(
        #     image=image,
        #     label=label,
        #     num_classes=self.num_classes,
        #     colors=self.colors,
        #     transparency=0.4
        # )
        
        # frames = [Image.fromarray(volume[i]) for i in range(len(volume))]
        # gif_buffer = BytesIO()
        # frames[0].save(gif_buffer, format="GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
        # gif_buffer.seek(0)
        
        # # trainer.logger.log_image(
        # #     key=f"Ground truth sample",
        # #     images=[wandb.Video(gif_buffer, format="gif")],
        # # )
        # wandb.log({"Ground truth sample": wandb.Video(gif_buffer, format="gif")})
        
        # gif_buffer.close()
        pass

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # with torch.no_grad():
        for i in range(self.n_images_to_log):
            transformed = self.transform(self.image_dict[i])
            if isinstance(transformed, list):
                transformed = transformed[0]
            image = transformed["image"] # [1, 128, 128, 128]
            label = transformed["label"] # [1, 128, 128, 128]
            
            image = image.unsqueeze(0).to(trainer.model.device)
            
            if not self.do_ds:
                prob = torch.sigmoid(trainer.model(image)) # [1, 1, 128, 128, 128]
            else:
                prob = torch.sigmoid(trainer.model(image)[0])
                
            seg = prob[0].detach().cpu().numpy() # [1, 128, 128, 128]

            seg = (seg > 0.5).astype(np.int8)
                        
            image = image.squeeze(0)         
            case = self.image_dict[i]["case"]            

            self.visualization_process(
                trainer=trainer,
                image=image.squeeze(0), # [128, 128, 128]
                label=label.squeeze(0),
                pred=seg.squeeze(0),
                case=case,
                visualize_type="grid",
                visualization_key="predicted mask (training)"
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
        
        if len(self.n_samples_validation) < self.number_of_logged_samples * 30:
            for i, l, o in zip(batch["image"], batch["label"], outputs["preds"]):
                self.n_samples_validation.append({
                    "image": i.squeeze(0).cpu(),
                    "label": l.squeeze(0).cpu(),
                    "preds": o.detach().squeeze(0)
                })
            

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Random shuffle
        random.shuffle(self.n_samples_validation)
        self.n_samples_validation = self.n_samples_validation[:self.number_of_logged_samples * self.coefficient]

        with torch.no_grad():
            for batch in self.n_samples_validation:
                image = batch["image"] # [128, 128, 128]
                label = batch["label"] # [128, 128, 128]                
                prob = torch.sigmoid(batch["preds"].unsqueeze(0))
                seg = prob[0].cpu().numpy()
                seg = (seg > 0.5).astype(np.int8)
                
                self.visualization_process(
                    trainer=trainer,
                    image=image,
                    label=label.numpy().astype(np.int8),
                    pred=seg,
                    visualize_type="grid",
                    visualization_key="predicted mask (validation)"
                )     
            
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
        
        if len(self.n_samples_test) < self.number_of_logged_samples * 30:
            for i, l, o in zip(batch["image"], batch["label"], outputs["preds"]):
                self.n_samples_test.append({
                    "image": i.squeeze(0).cpu(),
                    "label": l.squeeze(0).cpu(),  
                    "preds": o.detach().squeeze(0)
                })
            

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # Random shuffle
        random.shuffle(self.n_samples_test)
        
        self.n_samples_test = self.n_samples_test[:self.number_of_logged_samples * self.coefficient]

        with torch.no_grad():
            for batch in self.n_samples_test:
                image = batch["image"] # [128, 128, 128]
                label = batch["label"] # [128, 128, 128]
                                
                prob = torch.sigmoid(batch["preds"].unsqueeze(0))
                seg = prob[0].cpu().numpy()
                seg = (seg > 0.5).astype(np.int8)
                
                self.visualization_process(
                    trainer=trainer,
                    image=image,
                    label=label.numpy().astype(np.int8),
                    pred=seg,
                    visualize_type="grid",
                    visualization_key="predicted mask (testing)"
                )
            
        self.n_samples_test.clear()
        
    def visualization_process(
        self, 
        trainer,
        image, 
        label, 
        pred, 
        case: str = '', 
        visualize_type: Literal["gif", "grid"] = "grid",
        visualization_key: str = "Predict"
    ):
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
        ) # 128 x [128, 128, 3]
        
        # Visualize with GIF
        if visualize_type == "gif":
            self._visualize_with_gif(volume_label=volume_label, volume_pred=volume_pred, case=case)
        else:
            first_idx, last_idx = self._find_first_last(label)
            indices_to_show = self._choosing_slice(first_idx=first_idx, last_idx=last_idx)

            label_tensor = self._slices_to_tensor(slice_list=volume_label, indices=indices_to_show)
            pred_tensor = self._slices_to_tensor(slice_list=volume_pred, indices=indices_to_show)
            
            label_grid = make_grid(label_tensor, nrow=3, padding=2) # 9, 3, 128, 128
            pred_grid = make_grid(pred_tensor, nrow=3, padding=2)

            grid_label_np = label_grid.numpy().transpose(1, 2, 0)
            grid_predict_np = pred_grid.numpy().transpose(1, 2, 0)

            grid_label_np = Image.fromarray(grid_label_np)
            grid_predict_np = Image.fromarray(grid_predict_np)

            if case:
                logger_key = visualization_key + " - " + f"Case {str(case)}"
                label_caption = f"Label - {case}"
                pred_caption = f"Predict - {case}"
            else:
                logger_key = visualization_key
                label_caption = f"Label"
                pred_caption = f"Predict"

            wandb_logger = trainer.logger
            wandb_logger.log_image(
                key=logger_key, 
                images=[grid_label_np, grid_predict_np],
                caption=[label_caption, pred_caption]
            )
    
    def _visualize_with_gif(
        self,
        volume_label: list[np.ndarray],
        volume_pred: list[np.ndarray],
        case: str,
        visualization_key: str
    ):
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

        if case:
            logger_key = visualization_key + " - " + f"Case {case}"
        else:
            logger_key = visualization_key
        
        wandb.log({logger_key: wandb.Video(gif_buffer, format="gif")})
            
        gif_buffer.close()
    
    def _find_first_last(self, slices: list[np.ndarray]) -> Union[int, None]:
        first = None
        last = None
        for idx, arr in enumerate(slices):
            if np.any(arr == 1):
                if first is None:
                    first = idx
                last = idx
        if first is None or last is None:
            first, last = 0, len(slices) - 1
        return first, last
    
    def _choosing_slice(
        self, 
        num_grid: int = 9, 
        first_idx: int = 0, 
        last_idx: int = 127
    ) -> list[int]:
        if last_idx - first_idx < num_grid - 1:
            indices_to_show = list(range(first_idx, last_idx + 1))
            while len(indices_to_show) < num_grid:
                indices_to_show.append(last_idx)
        else:
            indices_to_show = np.linspace(first_idx, last_idx, num_grid)
            indices_to_show = np.around(indices_to_show).astype(int).tolist()

        return indices_to_show
    
    def _slices_to_tensor(
        self,
        slice_list: list[int], 
        indices: list[int]
    ) -> torch.Tensor:
        tensor_list = []
        for i in indices:
            t = torch.from_numpy(slice_list[i])
            t = t.permute(2, 0, 1)
            tensor_list.append(t)
        return torch.stack(tensor_list)