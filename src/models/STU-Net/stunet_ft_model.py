from typing import Union, Type, List, Tuple, Literal

import os
import torch
from torch import nn
from .network_architecture import STUNet
from monai.losses import AsymmetricUnifiedFocalLoss

class STUNET_FT_Model(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1, 
        enable_deep_supervision: bool = False,
        model_type: Literal["base", "large"] = "base",
        pretrained_weights_base: str = "/home/duong.quang.minh/project/3D-SegClass-Medical/checkpoints/stunet/base_ep4k.model",
        pretrained_weights_large: str = "/home/duong.quang.minh/project/3D-SegClass-Medical/checkpoints/stunet/large_ep4k.model",
    ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()

        if model_type == "base":
            depth = [1,1,1,1,1,1]
            dims = [32 * x for x in [1, 2, 4, 8, 16, 16]]
            self.pretrained_weights = pretrained_weights_base

        if model_type == "large":
            depth = [2] * 6
            dims = [64 * x for x in [1, 2, 4, 8, 16, 16]]
            self.pretrained_weights = pretrained_weights_large


        pool_op_kernel_sizes = [[2, 2, 2],[2, 2, 2],[2, 2, 2],[2, 2, 2],[1, 1, 2]]
        conv_kernel_sizes = [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
                
        self.model = STUNet(
            input_channels=input_channels, 
            num_classes=num_classes, 
            depth=depth, 
            dims=dims,
            pool_op_kernel_sizes=pool_op_kernel_sizes, 
            conv_kernel_sizes=conv_kernel_sizes, 
            enable_deep_supervision=enable_deep_supervision
        )
        
        self._load_stunet_pretrained_weights()
        
        del self.model

    def forward(self, x):
        output = self.mod(x)
        return output

    def _load_stunet_pretrained_weights(self, verbose=False):

        saved_model = torch.load(self.pretrained_weights, weights_only=False)

        if self.pretrained_weights.endswith('pth'):
            pretrained_dict = saved_model['network_weights']
        elif self.pretrained_weights.endswith('model'):
            pretrained_dict = saved_model['state_dict']

        skip_strings_in_pretrained = [
            'seg_outputs',
        ]

        self.mod = self.model

        model_dict = self.mod.state_dict()

        # Adjust for multimodal inputs
        num_inputs = model_dict['conv_blocks_context.0.0.conv1.weight'].shape[1]
        if num_inputs > 1:
            pretrained_conv1_weight = pretrained_dict['conv_blocks_context.0.0.conv1.weight']
            pretrained_conv3_weight = pretrained_dict['conv_blocks_context.0.0.conv3.weight']
            pretrained_dict['conv_blocks_context.0.0.conv1.weight'] = pretrained_conv1_weight.repeat(1, num_inputs, 1, 1, 1)
            pretrained_dict['conv_blocks_context.0.0.conv3.weight'] = pretrained_conv3_weight.repeat(1, num_inputs, 1, 1, 1)

        # Verify that all but the segmentation layers have the same shape
        for key, _ in model_dict.items():
            if all([i not in key for i in skip_strings_in_pretrained]):
                assert key in pretrained_dict, \
                    f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                    f"compatible with your network."
                assert model_dict[key].shape == pretrained_dict[key].shape, \
                    f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                    f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
                    f"does not seem to be compatible with your network."

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

        model_dict.update(pretrained_dict)

        print("################### Loading pretrained weights from file ", self.pretrained_weights, '###################')
        if verbose:
            print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
            for key, value in pretrained_dict.items():
                print(key, 'shape', value.shape)
            print("################### Done ###################")
            
        self.mod.load_state_dict(model_dict)
        return self.mod

if __name__ == '__main__':

    from monai.inferers import sliding_window_inference

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    data = torch.rand((1, 1, 256, 128, 128)).to(device)
    
    model = STUNET_FT_Model(
        num_classes=2,
        enable_deep_supervision=True
    ).to(device)
    
    # init lr: 1e-3
    # print(model.mod.state_dict()['conv_blocks_context.0.0.conv1.weight'])
    
    outputs = sliding_window_inference(data, (128, 128, 128), 1, model)

    if isinstance(outputs, tuple):
        for i in outputs:
            print(i.shape) 

    else:
        print(outputs.shape) # 1, 1, 256, 128, 128

