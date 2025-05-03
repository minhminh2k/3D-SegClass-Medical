import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import os

from src.models.LIDC_module import LIDC_Module
from src.models.STU_Net.stunet_ft_model import STUNET_FT_Model
from src.models.swin_unetr.SwinUNETR import SwinUNETR
from src.models.unetr.UNETR import UNETR
from src.models.unet3d.unet3d import UNet3D

from src.models.losses.combine_loss import CombineLoss, UnifiedFocalLoss
from src.models.losses.monai_losses import CELoss, DiceLoss, FocalLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LIDC_Module.load_from_checkpoint(
        "checkpoints/unetr.ckpt",
        net=UNETR(
            feature_size=32,
            hidden_size=768,
            mlp_dim=3072,
            proj_type="perceptron",
        ),
        criterion=UnifiedFocalLoss(
            sigmoid=True,
            to_onehot_y=False,
            num_classes=2,
            gamma=0.75,
            delta=0.7,
            softmax=False,
            include_background=False,
            epsilon=1e-7,
            lambda_dice=1.0,
            lambda_focal=1.0
        ),
        ce_loss=CELoss(),
        dice_loss=DiceLoss(
            to_onehot_y=False,
            sigmoid=True,
            softmax=False,
            include_background=False,
            squared_pred=True
        ),
        focal_loss=FocalLoss(
            to_onehot_y=False,
            include_background=False
        ),
        map_location=torch.device(device),
    )

    # os.makedirs("triton/model_repository/unet/1/", exist_ok=True)
    
    model = model.to_torchscript("unetr.pt")

if __name__ == "__main__":
    main()