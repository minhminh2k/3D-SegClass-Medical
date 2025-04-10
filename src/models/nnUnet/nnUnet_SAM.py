from typing import Union, Type, List, Tuple

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from .building_blocks.plain_conv_encoder import PlainConvEncoder
from .building_blocks.unet_decoder import UNetDecoder
from .building_blocks.sam_decoder import SAMDecoder
from .building_blocks.helper import convert_conv_op_to_dim
from .building_blocks.residual_encoders import ResidualEncoder
from .building_blocks.residual import BasicBlockD, BottleneckD
from src.models.SAMs.SAMMed3D.modeling.image_encoder3D import ImageEncoderViT3D

# from mobile_sam import sam_model_registry
# from mobile_sam import SamAutomaticMaskGenerator, SamPredictor
# from mobile_sam.modeling.tiny_vit_sam import TinyViT

class SAMMed3D_nnUnet(nn.Module):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]] = 3,
        strides: Union[int, List[int], Tuple[int, ...]] = (1, 2, 2, 2, 2, 2),
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]] = [2, 2, 2, 2, 2, 2],
        num_classes: int = 1,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]] = (2, 2, 2, 2, 2),
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = nn.BatchNorm3d,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = nn.ReLU,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        # model_weight_path: str = "/home/duong.quang.minh/project/3D-SegClass-Medical/checkpoints/mobilesam/mobile_sam.pt",
        # sam_model_type: str = "vit_t",
        sam_image_encoder_ckpt: str = "/home/duong.quang.minh/project/3D-SegClass-Medical/checkpoints/sammed3d/sam_med3d_turbo_image_encoder.pth",
        pretrained: bool = True,
        trainable_encoder: bool = False,
    ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        
        self.n_stages  = n_stages
        
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = SAMDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)

        # 3D ViT
        self.sam_image_encoder = ImageEncoderViT3D(
            depth=12,
            embed_dim=768,
            img_size=128,
            mlp_ratio=4,
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            out_chans=384,
            in_chans=1
        )
        
        if pretrained:
            image_encoder_model_state=torch.load(sam_image_encoder_ckpt, weights_only = False)
            self.sam_image_encoder.load_state_dict(state_dict=image_encoder_model_state)
            
            if not trainable_encoder:
                for param in self.sam_image_encoder.parameters():
                    param.requires_grad = False        
        
        ## MobileSAM for 2D
        # mobile_sam = sam_model_registry[sam_model_type](checkpoint=model_weight_path)
        # self.sam_image_encoder = mobile_sam.image_encoder
        # for param in self.sam_image_encoder.parameters():
            # param.requires_grad = False

    def forward(self, x):

        # sam_input = x.detach()
        # if sam_input.shape[1] == 1:
        #     sam_input = sam_input.repeat(1, 3, 1, 1)
        # sam_input = F.interpolate(sam_input, size=(1024, 1024), mode='bilinear', align_corners=True)
        # sam_embed = F.interpolate(sam_embed, size=(skips[3].shape[2], skips[3].shape[3]), mode='bilinear', align_corners=True) # 1, 256, 64, 64
        
        sam_embed, _ = self.sam_image_encoder(x) # 1, 384, 8, 8, 8
        skips = self.encoder(x)
        skips[self.n_stages - 2] = torch.cat((skips[self.n_stages - 2], sam_embed), dim=1)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = torch.rand((10, 2, 1, 128, 128, 128)).to(device)
    
    # 18 Gb VRAM
    model = SAMMed3D_nnUnet(1, 6, (32, 64, 128, 256, 384, 512), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 1,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=False).to(device)
    
    for i in data:
        print(i.shape)
        output = model(i)
        print(output.shape)
    