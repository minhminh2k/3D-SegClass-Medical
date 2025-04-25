"""Lovasz-Softmax and Jaccard hinge loss in PyTorch Maxim Berman 2018 ESAT-PSI KU Leuven (MIT
License)"""

"""
Implementation from https://github.com/bermanmaxim/LovaszSoftmax/
# copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/LovaszSoftmax/lovasz_loss.py

"""

import warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss
from monai.data import MetaTensor
from collections.abc import Callable, Sequence
from monai.utils import DiceCEReduction, LossReduction, Weight, look_up_option, pytorch_after
from monai.networks import one_hot

# from torch.autograd import Function

# --------------------------- HELPER FUNCTIONS ---------------------------


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels):
    r"""
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    """
    loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels))
    return loss


def lovasz_hinge_flat(logits, labels):
    r"""
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
    """

    signs = 2.0 * labels.float() - 1.0  # labels = 0, signs < 0; labels = 1, signs > 0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    #     perm = perm.data -> fixed
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.elu(errors_sorted) + 1, grad)
    return loss


def flatten_binary_scores(scores, labels):
    """Flattens predictions in the batch (binary case)."""
    scores = scores.view(-1)
    labels = labels.view(-1)
    return scores, labels


def binary_xloss(logits, labels):
    r"""
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
    """
    logits, labels = flatten_binary_scores(logits, labels)
    loss = StableBCELoss()(logits, labels.float())
    return loss


# --------------------------- MODULES ---------------------------


class LovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit, labels):
        if isinstance(logit, MetaTensor):
            logit = logit.as_tensor()
        if isinstance(labels, MetaTensor):
            labels = labels.as_tensor()
        return lovasz_hinge(logit, labels)


class BCE_Lovasz(nn.Module):
    def __init__(self, pos_weight: torch.FloatTensor = None):
        super().__init__()
        self.nll_loss = BCEWithLogitsLoss(pos_weight=pos_weight)

    def update_pos_weight(self, pos_weight: torch.FloatTensor = None):
        if pos_weight is not None:
            self.nll_loss.pos_weight = pos_weight

    def forward(self, logit, labels):
        return lovasz_hinge(logit, labels) + self.nll_loss(logit, labels)


class SBCE_Lovasz(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit, truth):
        bce = binary_xloss(logit, truth)
        lovasz = lovasz_hinge(logit, truth)
        return bce + lovasz


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()



class LovaszSoftmax(nn.Module):
    def __init__(
        self, 
        reduction: LossReduction | str = LossReduction.MEAN,
        include_background: bool = False,
        to_onehot_y: bool = True,
        softmax: bool = True,
        num_classes: int = 2,
        sigmoid: bool = False,
        epsilon: float = 1e-7
    ):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.num_classes = num_classes
        self.include_background = include_background
        self.sigmoid = sigmoid
        self.epsilon = epsilon

    def prob_flatten(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        assert y_pred.dim() in [4, 5]
        num_class = y_pred.size(1)
        
        assert y_pred.size(1) == y_true.size(1)
        
        if y_pred.dim() == 4:
            y_pred = y_pred.permute(0, 2, 3, 1).contiguous()
            y_pred_flatten = y_pred.view(-1, num_class)
            
            y_true = y_true.permute(0, 2, 3, 1).contiguous()
            y_true_flatten = y_true.view(-1, num_class)
        elif y_pred.dim() == 5:
            y_pred = y_pred.permute(0, 2, 3, 4, 1).contiguous()
            y_pred_flatten = y_pred.view(-1, num_class)
            
            y_true = y_true.permute(0, 2, 3, 4, 1).contiguous()
            y_true_flatten = y_true.view(-1, num_class)
            
        # y_true_flatten = y_true.view(-1)
        return y_pred_flatten, y_true_flatten

    def lovasz_softmax_flat(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        num_classes = y_pred.size(1)
        assert y_pred.size(1) == y_true.size(1)

        losses = []
        
        for c in range(num_classes):
            target_c = y_true[:, c].float()
            input_c = y_pred[:, c]
            
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)
                
        if self.reduction == LossReduction.MEAN.value:
            loss = torch.mean(losses)
        elif self.reduction == LossReduction.SUM.value:
            loss = torch.sum(losses)
        else:
            loss = torch.mean(losses)
        return loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        if len(y_pred.shape) != 4 and len(y_pred.shape) != 5:
            raise ValueError(f"input shape must be 4 or 5, but got {y_pred.shape}")
        
        if self.sigmoid:
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.clamp(y_pred, self.epsilon, 1.0 - self.epsilon)
        
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
                
        if not self.include_background:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `include_background=False` ignored.")
            else:
                # if skipping background, removing first channel
                y_true = y_true[:, 1:]
                y_pred = y_pred[:, 1:]
                
        if isinstance(y_pred, MetaTensor):
            y_pred = y_pred.as_tensor()
        if isinstance(y_true, MetaTensor):
            y_true = y_true.as_tensor()
        
        # print(y_pred.shape, y_true.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        y_pred, y_true = self.prob_flatten(y_pred, y_true)
        # print(y_pred.shape, y_true.shape)
        losses = self.lovasz_softmax_flat(y_pred, y_true)
        return losses


if __name__ == "__main__":
    ch0 = torch.zeros((1, 1, 128, 128, 128))

    ch1 = torch.ones((1, 1, 128, 128, 128))

    x1 = torch.cat([ch0, ch1], dim=1)
    y1 = torch.randint(0, 1, (1, 1, 128, 128, 128))

    lovasz_v2 = LovaszSoftmax(
        to_onehot_y=True,
        softmax=True,
        num_classes=2,
        include_background=False
    )
    
    # print(lovasz(x1, y1))
    print(lovasz_v2(x1, y1))
    