import torch
import torch.nn as nn
from monai.utils import LossReduction
from monai.networks import one_hot
import logging

# https://github.com/IvanVassi/FocalTversky3D_pytorch/blob/main/README.md
# https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py

class FocalTverskyLoss(nn.Module):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842

    For binary segmentation -> sigmoid
    """
    def __init__(
        self, 
        alpha: float = 0.7, 
        beta: float = 0.3, 
        gamma: float = 0.75, 
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.sigmoid(y_pred)
        
        # clip the prediction to avoid NaN
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        
        axis = list(range(2, len(y_pred.shape)))
        
        # Calculate the Tversky loss
        tp = (y_true * y_pred).sum(dim=axis)
        fn = (y_true * (1 - y_pred)).sum(dim=axis)
        fp = ((1 - y_true) * y_pred).sum(dim=axis)
        
        tversky_index = (tp + self.epsilon) / (tp + self.alpha * fn + self.beta * fp + self.epsilon)
        
        # Calculate the Focal Tversky loss
        loss: torch.Tensor = (1 - tversky_index).pow(self.gamma)
        
        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE.value:
            return loss  # returns [N, num_classes] losses
        if self.reduction == LossReduction.MEAN.value:
            return loss.mean()
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
    
class TverskyLoss(nn.Module):
    def __init__(
        self, 
        delta: float = 0.7, 
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        super().__init__()
        self.delta = delta
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.sigmoid(y_pred)
                
        axis = list(range(2, len(y_pred.shape)))
        
        # Calculate the Tversky loss
        tp = (y_true * y_pred).sum(dim=axis)
        fn = (y_true * (1 - y_pred)).sum(dim=axis)
        fp = ((1 - y_true) * y_pred).sum(dim=axis)
        
        tversky_index = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)
        
        loss = 1 - tversky_index
        
        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE.value:
            return loss  # returns [N, num_classes] losses
        if self.reduction == LossReduction.MEAN.value:  
            return loss.mean()
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

################################
#       Symmetric Focal loss      #
################################
class SymmetricFocalLoss(nn.Module):
    """
    SymmetricFocalLoss is a variant of FocalTverskyLoss, which attentions to the foreground class.

    Actually, it's only supported for binary image segmentation now.

    Reimplementation of the Asymmetric Focal Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self, 
        delta: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        
        cross_entropy = -y_true * torch.log(y_pred)
        
        back_ce = torch.pow(1 - y_pred[:, 0], self.gamma) * cross_entropy[:, 0]
        back_ce = (1 - self.delta) * back_ce
        
        fore_ce = torch.pow(1 - y_pred[:, 1], self.gamma) * cross_entropy[:, 1]
        fore_ce = self.delta * fore_ce
        
        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], dim=1), dim=1))    
        
        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE.value:
            return loss  # returns [N, num_classes] losses
        if self.reduction == LossReduction.MEAN.value:  
            return loss.mean()
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
    
#################################
# Symmetric Focal Tversky loss  #
#################################

class SymmetricFocalTverskyLoss(nn.Module):
    """
    SymmetricFocalTverskyLoss is a variant of FocalTverskyLoss, which attentions to the foreground class.

    Actually, it's only supported for binary image segmentation now.

    Reimplementation of the Asymmetric Focal Tversky Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self, 
        delta: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)

        axis = list(range(2, len(y_pred.shape)))
        
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1 - y_pred), dim=axis)
        fp = torch.sum((1 - y_true) * y_pred, dim=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)

        
        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1 - dice_class[:, 0]) * torch.pow(1 - dice_class[:, 0], -self.gamma)
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice, fore_dice], dim=-1))
        return loss
    
################################
#     Asymmetric Focal loss    #
################################
class AsymmetricFocalLoss(nn.Module):
    """
    AsymmetricFocalLoss is a variant of FocalTverskyLoss, which attentions to the foreground class.

    Actually, it's only supported for binary image segmentation now.

    Reimplementation of the Asymmetric Focal Tversky Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self, 
        delta: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
        include_background: bool = False,
        sigmoid: bool = False
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction
        self.include_background = include_background
        self.sigmoid = sigmoid
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        axis = list(range(2, len(y_pred.shape)))
        
        cross_entropy = -y_true * torch.log(y_pred)
        
        if self.sigmoid:
            fore_ce = torch.pow(1 - y_pred[:, 0], -self.gamma) * cross_entropy[:, 0]
            return torch.mean(fore_ce)
        else:
            if self.include_background:      
                back_ce = torch.pow(1 - y_pred[:, 0], self.gamma) * cross_entropy[:, 0]
                back_ce = (1 - self.delta) * back_ce
                
                fore_ce = cross_entropy[:, 1]
                fore_ce = self.delta * fore_ce

                loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], dim=1), dim=1))
                return loss
            else:
                fore_ce = torch.pow(1 - y_pred[:, 1], -self.gamma) * cross_entropy[:, 1]
                return torch.mean(fore_ce)
    
#################################
# Asymmetric Focal Tversky loss #
#################################
class AsymmetricFocalTverskyLoss(nn.Module):
    """
    AsymmetricFocalLoss is a variant of FocalTverskyLoss, which attentions to the foreground class.

    Actually, it's only supported for binary image segmentation now.

    Reimplementation of the Asymmetric Focal Tversky Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self, 
        delta: float = 0.7,
        gamma: float = 0.75,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
        include_background: bool = False,
        sigmoid: bool = False,
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction
        self.sigmoid = sigmoid
        self.include_background = include_background

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        axis = list(range(2, len(y_pred.shape)))
        
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1 - y_pred), dim=axis)
        fp = torch.sum((1 - y_true) * y_pred, dim=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)
        
        if self.sigmoid:
            fore_dice = (1 - dice_class[:, 0]) * torch.pow(1 - dice_class[:, 0], -self.gamma)
            return torch.mean(fore_dice)
        
        # Calculate losses separately for each class, enhancing both classes
        back_dice = 1 - dice_class[:, 0]
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)
        
        if self.include_background:
            # Average class scores
            loss = torch.mean(torch.stack([back_dice, fore_dice], dim=-1))
            return loss
        else:
            loss = torch.mean(fore_dice)
            return loss
    
###########################################
#      Symmetric Unified Focal loss       #
###########################################

class SymmetricUnifiedFocalLoss(nn.Module):
    """
    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """
    def __init__(
        self,
        to_onehot_y: bool = True,
        num_classes: int = 2,
        weight: float = 0.5,
        delta: float = 0.7,
        gamma: float = 0.5,
        epsilon: float = 1e-7,
        softmax: bool = True,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.weight = weight
        self.num_classes = num_classes
        self.reduction = reduction
        self.sy_focal_loss = SymmetricFocalLoss(gamma=self.gamma, delta=self.delta)
        self.sy_focal_tversky_loss = SymmetricFocalTverskyLoss(gamma=self.gamma, delta=self.delta)
        

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred : the shape should be BNH[WD], where N is the number of classes.
                It only supports binary segmentation.
                The input should be the original logits since it will be transformed by
                    a sigmoid in the forward function.
            y_true : the shape should be BNH[WD], where N is the number of classes.
                It only supports binary segmentation.

        Raises:
            ValueError: When input and target are different shape
            ValueError: When len(y_pred.shape) != 4 and len(y_pred.shape) != 5
            ValueError: When num_classes
            ValueError: When the number of classes entered does not match the expected number
        """
        if len(y_pred.shape) != 4 and len(y_pred.shape) != 5:
            raise ValueError(f"input shape must be 4 or 5, but got {y_pred.shape}")
        
        n_pred_ch = y_pred.shape[1]
        
        if self.softmax:
            if n_pred_ch == 1:
                raise ValueError("single channel prediction, `softmax=True` ignored.")
            else:
                y_pred = torch.softmax(y_pred, 1)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                raise ValueError("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)
        
        sy_focal_loss = self.sy_focal_loss(y_pred, y_true)
        sy_focal_tversky_loss = self.sy_focal_tversky_loss(y_pred, y_true)
        
        loss: torch.Tensor = self.weight * sy_focal_loss + (1 - self.weight) * sy_focal_tversky_loss
        
        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE.value:
            return loss  # returns [N, num_classes] losses
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(loss)
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
    
###########################################
#      Asymmetric Unified Focal loss      #
###########################################
class AsymmetricUnifiedFocalLoss(nn.Module):
    """
    AsymmetricUnifiedFocalLoss is a variant of Focal Loss.

    Actually, it's only supported for binary image segmentation now

    Reimplementation of the Asymmetric Unified Focal Tversky Loss described in:

    - "Unified Focal Loss: Generalising Dice and Cross Entropy-based Losses to Handle Class Imbalanced Medical Image Segmentation",
    Michael Yeung, Computerized Medical Imaging and Graphics
    """

    def __init__(
        self,
        to_onehot_y: bool = True,
        num_classes: int = 2,
        lambda_focal: float = 0.3,
        lambda_dice: float = 0.7,
        gamma: float = 0.5,
        delta: float = 0.7,
        softmax: bool = True,
        reduction: LossReduction | str = LossReduction.MEAN,
        include_background: bool = False,
        epsilon: float = 1e-7,
        sigmoid: bool = True
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
        self.reduction = reduction
        self.softmax = softmax
        self.epsilon = epsilon
        self.include_background = include_background
        self.sigmoid = sigmoid
        self.asy_focal_loss = AsymmetricFocalLoss(gamma=self.gamma, delta=self.delta, epsilon=epsilon, reduction=reduction, include_background=include_background, sigmoid=sigmoid)
        self.asy_focal_tversky_loss = AsymmetricFocalTverskyLoss(gamma=self.gamma, delta=self.delta, epsilon=epsilon, reduction=reduction, include_background=include_background, sigmoid=sigmoid)

    # TODO: Implement this  function to support multiple classes segmentation
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred : the shape should be BNH[WD], where N is the number of classes.
                It only supports binary segmentation.
                The input should be the original logits since it will be transformed by
                    a sigmoid in the forward function.
            y_true : the shape should be BNH[WD], where N is the number of classes.
                It only supports binary segmentation.

        Raises:
            ValueError: When input and target are different shape
            ValueError: When len(y_pred.shape) != 4 and len(y_pred.shape) != 5
            ValueError: When num_classes
            ValueError: When the number of classes entered does not match the expected number
        """

        if len(y_pred.shape) != 4 and len(y_pred.shape) != 5:
            raise ValueError(f"input shape must be 4 or 5, but got {y_pred.shape}")
        
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
        
        n_pred_ch = y_pred.shape[1]
        
        if self.softmax:
            if n_pred_ch == 1:
                raise ValueError("single channel prediction, `softmax=True` ignored.")
            else:
                y_pred = torch.softmax(y_pred, 1)

        if self.to_onehot_y:
            if n_pred_ch == 1:
                raise ValueError("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                y_true = one_hot(y_true, num_classes=n_pred_ch)

        asy_focal_loss = self.asy_focal_loss(y_pred, y_true)
        asy_focal_tversky_loss = self.asy_focal_tversky_loss(y_pred, y_true)

        loss: torch.Tensor = self.lambda_focal * asy_focal_loss + self.lambda_dice * asy_focal_tversky_loss

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE.value:
            return loss  # returns [N, num_classes] losses
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(loss)
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
    
if __name__ == "__main__":
    loss = AsymmetricUnifiedFocalLoss()
    
    