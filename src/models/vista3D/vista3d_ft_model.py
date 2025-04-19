from typing import Union, Type, List, Tuple, Sequence

import os
import torch
from torch import nn
from .vista3d import vista3d132

class Vista3D_FT_Model(nn.Module):
    def __init__(
        self,
        in_channels: int = 1, 
        encoder_embed_dim: int = 48,
        label_mappings = {"default": [[1, 23]]},
        label_set: list = [0, 23],
        label_dict_path: str = "/home/duong.quang.minh/project/3D-SegClass-Medical/preprocess_data/vista3d/label_dict.json",
        pretrained_weights: str = "/home/duong.quang.minh/project/3D-SegClass-Medical/checkpoints/vista3d/model_vista3d.pt",
    ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        
        self.pretrained_weights = pretrained_weights
        
        self.model = vista3d132(
            encoder_embed_dim=encoder_embed_dim,
            in_channels=in_channels
        )

        self.model.load_state_dict(torch.load(self.pretrained_weights, weights_only=True))

        self.label_mappings = label_mappings
        self.label_dict_path = label_dict_path
        self.label_set = label_set

    def forward(
        self, 
        input_images: torch.Tensor, 
        point_coords: list[Sequence[slice]] | None = None, 
        point_labels: torch.Tensor | None = None, 
        class_vector: torch.Tensor | None = None,
        transpose: bool = False
    ):
        return self.model(
            input_images=input_images,
            point_coords=point_coords,
            point_labels=point_labels,
            class_vector=class_vector,
            transpose=transpose
        )

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = torch.rand((20, 1, 1, 128, 128, 128)).to(device)
    
    model = Vista3D_FT_Model().to(device)
    
    for i in data:
        print("Input shape: ", i.shape) # torch.Size([1, 1, 128, 128, 128])
        output = model(i)
        print("Output shape: ", output.shape) # torch.Size([1, 1, 128, 128, 128])