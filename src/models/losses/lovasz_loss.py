"""Lovasz-Softmax and Jaccard hinge loss in PyTorch Maxim Berman 2018 ESAT-PSI KU Leuven (MIT
License)"""

"""
Implementation from https://github.com/bermanmaxim/LovaszSoftmax/
# copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/LovaszSoftmax/lovasz_loss.py

"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss
from monai.data import MetaTensor

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
            logit_torch = logit.as_tensor()
        if isinstance(labels, MetaTensor):
            labels_torch = labels.as_tensor()
        return lovasz_hinge(logit_torch, labels_torch)


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
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        inputs, targets = self.prob_flatten(inputs, targets)
        # print(inputs.shape, targets.shape)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


if __name__ == "__main__":
    x1 = torch.rand((2, 1, 128, 128, 128))
    y1 =  torch.rand((2, 1, 128, 128, 128)) 
    lovasz = LovaszLoss()
    lovasz_v2 = LovaszSoftmax()
    
    print(lovasz(x1, y1))
    print(lovasz_v2(x1, y1))
    