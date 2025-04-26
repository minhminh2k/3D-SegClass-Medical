from typing import Literal

import torch
import torch.nn as nn
from monai.utils import LossReduction
from monai.networks import one_hot
import logging
from .focal_tversky_3d import AsymmetricUnifiedFocalLoss
from .lovasz_loss import LovaszSoftmax
from .monai_losses import DiceCELoss, DiceLoss
from monai.losses import GeneralizedDiceFocalLoss, HausdorffDTLoss

class CombineLoss(nn.Module):
    def __init__(
        self,
        to_onehot_y: bool = True,
        num_classes: int = 2,
        gamma: float = 1.25,
        delta: float = 0.8,
        softmax: bool = True,
        sigmoid: bool = False,
        reduction: LossReduction | str = LossReduction.MEAN,
        include_background: bool = False,
        epsilon: float = 1e-7,
        loss_type: Literal["gdl_focal", "asymm_focal", "dicece"] = "dicece",
        lambda_dice: float = 0.7,
        lambda_focal: float = 0.3,
        lambda_lovasz: float = 0.5,
        lambda_hd: float = 5.0,
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
        
        if loss_type == "asymm_focal":
            self.dice_focal = AsymmetricUnifiedFocalLoss(
                to_onehot_y=to_onehot_y,
                num_classes=num_classes,
                lambda_focal=lambda_focal,
                lamdba_dice=lambda_dice,
                gamma=gamma,
                delta=delta,
                softmax=softmax,
                reduction=reduction,
                include_background=include_background,
                epsilon=epsilon,
                sigmoid=sigmoid
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

        self.lovasz = LovaszSoftmax(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax,
            num_classes=num_classes,
            sigmoid=sigmoid,
            reduction=reduction,
            epsilon=epsilon
        )
        
        self.hd = HausdorffDTLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            reduction=reduction
        )
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        asymm_focal_loss = self.dice_focal(y_pred, y_true)
        lovasz_loss = self.lovasz(y_pred, y_true)
        hd_loss = self.hd(y_pred, y_true)

        total_loss: torch.Tensor = asymm_focal_loss + self.lambda_lovasz * lovasz_loss + self.lambda_hd * hd_loss

        return total_loss