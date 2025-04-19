import torch
import torch.nn as nn
from monai.losses import AsymmetricUnifiedFocalLoss
from monai.utils import LossReduction
from monai.networks import one_hot

# https://github.com/IvanVassi/FocalTversky3D_pytorch/blob/main/README.md
# https://github.com/mlyg/unified-focal-loss/blob/main/loss_functions.py

class FocalTverskyLoss(nn.Module):
    """A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation
    Link: https://arxiv.org/abs/1810.07842
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
        
        fore_ce = torch.pow(1 - cross_entropy[:, 1], self.gamma) * cross_entropy[:, 1]
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
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        axis = list(range(2, len(y_pred.shape)))
        
        cross_entropy = -y_true * torch.log(y_pred)
        
        back_ce = torch.pow(1 - y_pred[:, 0], self.gamma) * cross_entropy[:, 0]
        back_ce = (1 - self.delta) * back_ce
        
        fore_ce = cross_entropy[:, 1]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], dim=1), dim=1))
        return loss
    
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
    ):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        axis = list(range(2, len(y_pred.shape)))
        
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)
        tp = torch.sum(y_true * y_pred, dim=axis)
        fn = torch.sum(y_true * (1 - y_pred), dim=axis)
        fp = torch.sum((1 - y_true) * y_pred, dim=axis)
        dice_class = (tp + self.epsilon) / (tp + self.delta * fn + (1 - self.delta) * fp + self.epsilon)
        
        # Calculate losses separately for each class, enhancing both classes
        back_dice = 1 - dice_class[:, 0]
        fore_dice = (1 - dice_class[:, 1]) * torch.pow(1 - dice_class[:, 1], -self.gamma)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice, fore_dice], dim=-1))
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
        num_classes: int = 2,
        weight: float = 0.5,
        delta: float = 0.7,
        gamma: float = 0.5,
        epsilon: float = 1e-7,
        reduction: LossReduction | str = LossReduction.MEAN,
    ):
        super().__init__()
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
        if y_pred.shape != y_true.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

        if len(y_pred.shape) != 4 and len(y_pred.shape) != 5:
            raise ValueError(f"input shape must be 4 or 5, but got {y_pred.shape}")
        
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.sigmoid(y_pred)
        
        if y_pred.shape[1] == 1:
            y_pred = one_hot(y_pred, num_classes=self.num_classes)
            y_true = one_hot(y_true, num_classes=self.num_classes)
            
        if torch.max(y_true) != self.num_classes - 1:
            raise ValueError(f"Please make sure the number of classes is {self.num_classes-1}")
        
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
        to_onehot_y: bool = False,
        num_classes: int = 2,
        weight: float = 0.5,
        gamma: float = 0.5,
        delta: float = 0.7,
        reduction: LossReduction | str = LossReduction.MEAN,
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
        self.weight: float = weight
        self.reduction = reduction
        self.asy_focal_loss = AsymmetricFocalLoss(gamma=self.gamma, delta=self.delta)
        self.asy_focal_tversky_loss = AsymmetricFocalTverskyLoss(gamma=self.gamma, delta=self.delta)

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
        
        if y_pred.shape != y_true.shape:
            raise ValueError(f"ground truth has different shape ({y_true.shape}) from input ({y_pred.shape})")

        if len(y_pred.shape) != 4 and len(y_pred.shape) != 5:
            raise ValueError(f"input shape must be 4 or 5, but got {y_pred.shape}")

        y_pred = torch.sigmoid(y_pred)
        
        if y_pred.shape[1] == 1:
            y_pred = one_hot(y_pred, num_classes=self.num_classes)
            y_true = one_hot(y_true, num_classes=self.num_classes)
        
        if torch.max(y_true) != self.num_classes - 1:
            raise ValueError(f"Please make sure the number of classes is {self.num_classes-1}")

        asy_focal_loss = self.asy_focal_loss(y_pred, y_true)
        asy_focal_tversky_loss = self.asy_focal_tversky_loss(y_pred, y_true)

        loss: torch.Tensor = self.weight * asy_focal_loss + (1 - self.weight) * asy_focal_tversky_loss

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss)  # sum over the batch and channel dims
        if self.reduction == LossReduction.NONE.value:
            return loss  # returns [N, num_classes] losses
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(loss)
        raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')