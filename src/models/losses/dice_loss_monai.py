import torch
from torch import nn
import torch.nn.functional as F
from monai import losses
"""
Implementation of Dice Loss from https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
"""

class DiceLossMonAI(nn.Module):
    def __init__(self, to_onehot: bool, sigmoid: bool, softmax: bool, include_background: bool):
        super().__init__()
        self._loss = losses.DiceLoss(to_onehot_y=to_onehot, sigmoid=sigmoid, softmax=softmax, include_background=include_background)

    def forward(self, predicted, target):
        loss = self._loss(predicted, target)
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)
        inputs = torch.clamp(inputs, min=0, max=1) # If needed

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice