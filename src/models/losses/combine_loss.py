from typing import Literal

import torch
import torch.nn as nn
from monai.utils import LossReduction
from monai.networks import one_hot
import logging
from .focal_tversky_3d import AsymmetricUnifiedFocalLoss
from .lovasz_loss import LovaszSoftmax
from .monai_losses import DiceCELoss, DiceLoss
from monai.losses import GeneralizedDiceFocalLoss, HausdorffDTLoss, FocalLoss
from .focal_tversky_3d import FocalTverskyLoss

class CombineLoss(nn.Module):
    def __init__(
        self,
        to_onehot_y: bool = True,
        num_classes: int = 2,
        gamma: float = 0.75,
        delta: float = 0.7,
        softmax: bool = True,
        sigmoid: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        include_background: bool = False,
        epsilon: float = 1e-7,
        loss_type: Literal["gdl_focal", "tversky", "dicece"] = "tversky",
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
        lambda_lovasz: float = 1.0,
        lambda_hd: float = 0.5,
    ):
        """
        Args:
            to_onehot_y : whether to convert `y` into the one-hot format. Defaults to False.
            num_classes : number of classes, it only supports 2 now. Defaults to 2.
            delta : weight of the background. Defaults to 0.7.
            gamma : value of the exponent gamma in the definition of the Focal loss. Defaults to 0.75.
            epsilon : it defines a very small number each time. simmily smooth value. Defaults to 1e-7.
            weight : weight for each loss function, if it's none it's 0.5. Defaults to None.

        Example:
            >>> import torch
            >>> from monai.losses import AsymmetricUnifiedFocalLoss
            >>> pred = torch.ones((1,1,32,32), dtype=torch.float32)
            >>> grnd = torch.ones((1,1,32,32), dtype=torch.int64)
            >>> fl = AsymmetricUnifiedFocalLoss(to_onehot_y=True)
            >>> fl(pred, grnd)
        """
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.num_classes = num_classes
        self.gamma = gamma
        self.delta = delta
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.lambda_lovasz = lambda_lovasz
        self.lambda_hd = lambda_hd
        self.reduction = reduction
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.epsilon = epsilon
        
        self.focal = None
        
        # Auto sigmoid
        if loss_type == "tversky":
            self.dice_focal = FocalTverskyLoss(
                delta=delta,
                gamma=gamma,
                epsilon=epsilon,
                reduction=reduction
            )
            
            self.focal = FocalLoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                reduction=reduction,
                use_softmax=False
            )
            
        elif loss_type == "gdl_focal":
            self.dice_focal = GeneralizedDiceFocalLoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                reduction=reduction,
                gamma=gamma,
                lambda_focal=lambda_focal,
                lambda_gdl=lambda_dice
            )
        else:
            self.dice_focal = DiceCELoss(
                include_background=include_background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                softmax=softmax,
                reduction=reduction,
                lambda_ce=lambda_focal,
                lambda_dice=lambda_dice
            )

        # self.lovasz = LovaszSoftmax(
        #     include_background=include_background,
        #     to_onehot_y=to_onehot_y,
        #     softmax=softmax,
        #     num_classes=num_classes,
        #     sigmoid=sigmoid,
        #     reduction=reduction,
        #     epsilon=epsilon
        # )
        
        self.hd = HausdorffDTLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            reduction=reduction
        )
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if not self.focal:
            asymm_focal_loss = self.dice_focal(y_pred, y_true)
            # lovasz_loss = self.lovasz(y_pred, y_true)
            hd_loss = self.hd(y_pred, y_true)

            total_loss: torch.Tensor = asymm_focal_loss + self.lambda_hd * hd_loss \
                # + self.lambda_lovasz * lovasz_loss

            return total_loss
        else:
            tversky_loss = self.dice_focal(y_pred, y_true)
            focal_loss = self.focal(y_pred, y_true)
            # lovasz_loss = self.lovasz(y_pred, y_true)
            hd_loss = self.hd(y_pred, y_true)

            total_loss: torch.Tensor = self.lambda_dice * tversky_loss + self.lambda_focal * focal_loss + self.lambda_hd * hd_loss \
                # + self.lambda_lovasz * lovasz_loss

            return total_loss
        
class UnifiedFocalLoss(nn.Module):
    def __init__(
        self,
        to_onehot_y: bool = False,
        num_classes: int = 2,
        gamma: float = 0.75,
        delta: float = 0.7,
        softmax: bool = False,
        sigmoid: bool = True,
        reduction: LossReduction | str = LossReduction.MEAN,
        include_background: bool = False,
        epsilon: float = 1e-7,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
    ):

        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.num_classes = num_classes
        self.gamma = gamma
        self.delta = delta
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

        self.reduction = reduction
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.epsilon = epsilon

        self.focal_tversky = FocalTverskyLoss(
            delta=delta,
            gamma=gamma,
            epsilon=epsilon,
            reduction=reduction
        )
        
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            reduction=reduction,
            use_softmax=False
        )
            
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        focal_tversky_loss = self.focal_tversky(y_pred, y_true)
        focal_loss = self.focal(y_pred, y_true)
        total_loss: torch.Tensor = self.lambda_dice * focal_tversky_loss + self.lambda_focal * focal_loss

        return total_loss