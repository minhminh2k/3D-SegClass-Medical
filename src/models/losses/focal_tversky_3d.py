import torch
import torch.nn as nn

# https://github.com/IvanVassi/FocalTversky3D_pytorch/blob/main/README.md

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0., epsilon: float = 1e-7):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Ensure the predictions are in the same dimension as y_true
        y_pred = torch.sigmoid(y_pred)
        
        # Calculate the Tversky loss
        tp = (y_true * y_pred).sum(dim=(2, 3, 4))
        fn = (y_true * (1 - y_pred)).sum(dim=(2, 3, 4))
        fp = ((1 - y_true) * y_pred).sum(dim=(2, 3, 4))
        
        tversky_index = (tp + self.epsilon) / (tp + self.alpha * fn + self.beta * fp + self.epsilon)
        
        # Calculate the Focal Tversky loss
        loss = (1 - tversky_index).pow(self.gamma)
        
        return loss.mean()