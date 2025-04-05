from typing import Sequence, Tuple, Union

import torch.nn as nn
from functools import partial
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep
import torch.nn.functional as F
import torch
from functools import partial
from src.models.SAMs.SAMMed3D.modeling.image_encoder3D import ImageEncoderViT3D

class SAMMed3D_UNETR(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        image_encoder_ckpt: str = '/data/hpc/dqm/3D-SegClass-Medical/checkpoints/sammed3d/sam_med3d_turbo_image_encoder.pth',
        in_channels: int = 1,
        out_channels: int = 1,
        feature_size: int = 16,
        norm_name: Union[Tuple, str] = "instance",
        spatial_dims: int = 3,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        encoder_global_attn_indexes: tuple =[2, 5, 8, 11],
        vit_patch_size: int = 16,
        encoder_out_channels: int = 384,
        pretrained: bool = True,
        trainable_encoder: bool = True,
        res_block: bool = True,
        conv_block: bool = True
    ) -> None:

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(vit_patch_size, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")
        
        if embed_dim % feature_size != 0:
            raise ValueError("embed_dim should be divisible by feature_size.")
        
        # Image Encoder using Vision Transformer (ViT)
        self.image_encoder_vit = ImageEncoderViT3D(
                    depth=encoder_depth,
                    embed_dim=embed_dim,
                    img_size=img_size[0],
                    mlp_ratio=4,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                    num_heads=encoder_num_heads,
                    patch_size=vit_patch_size,
                    qkv_bias=True,
                    use_rel_pos=True,
                    global_attn_indexes=encoder_global_attn_indexes,
                    window_size=14,
                    out_chans=encoder_out_channels,
                    in_chans=in_channels
        )
        
        if pretrained:
            image_encoder_model_state=torch.load(image_encoder_ckpt, weights_only = False)
            self.image_encoder_vit.load_state_dict(state_dict=image_encoder_model_state)
            
            if not trainable_encoder:
                for param in self.image_encoder_vit.parameters():
                    param.requires_grad = False        
        
        # Encoder blocks        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        
        
        # Decoder blocks
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=embed_dim,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [embed_dim]


    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x
    
    def forward(self, x_in):
        x, hidden_states_out = self.image_encoder_vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[2]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[5]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[8]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(hidden_states_out[11])
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)        
        out = self.decoder2(dec1, enc1)        
        return self.out(out)