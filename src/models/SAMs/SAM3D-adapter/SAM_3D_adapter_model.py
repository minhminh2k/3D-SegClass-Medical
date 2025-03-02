import logging

import os
import shutil
import torch
import numpy as np
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from typing import Optional, Tuple, Type, Literal

from segment_anything import sam_model_registry
from segment_anything import SamPredictor
from segment_anything import SamAutomaticMaskGenerator

from .components.image_encoder import ImageEncoderViT_3d_v2 as ImageEncoderViT_3d
from .components.mask_decoder import VIT_MLAHead_h as VIT_MLAHead
from .components.prompt_encoder import PromptEncoder, TwoWayTransformer

from monai.losses import DiceCELoss, DiceLoss
from monai.inferers import sliding_window_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# References: https://github.com/med-air/3DSAM-adapter

class SAM_3D_Adapter(nn.Module):

    def __init__(
        self, 
        model_type: str = "vit_b", 
        mode_checkpoint: str = "/data/hpc/dqm/3D-SegClass-Medical/checkpoints/sam_vit_b_01ec64.pth",
        freeze_image_encoder: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = False,
        auto_finetuning: bool = False, # If False, Using Prompt Encoder
        rand_crop_size: tuple = (128, 128, 128),
        num_classes: int = 2,
        img_size: int = 1024,
        patch_size: int = 16,
        patch_depth: int=32,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        cubic_window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        num_slice = 1,
        device: str = 'cuda'
    ):
        super().__init__()
        self.model_type = model_type
        self.model_checkpoint = mode_checkpoint
        self.freeze_image_encoder = freeze_image_encoder
        self.freeze_prompt_encoder = freeze_prompt_encoder
        self.freeze_mask_decoder = freeze_mask_decoder
        self.num_classess = num_classes
        self.rand_crop_size = rand_crop_size
        self.path_size = self.rand_crop_size[0]
        self.auto_finetuning = auto_finetuning
        
        self.depth = depth
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.qkv_bias = qkv_bias
        self.use_rel_pos = use_rel_pos
        self.global_attn_indexes = global_attn_indexes
        self.window_size = window_size
        self.cubic_window_size = cubic_window_size
        self.out_chans = out_chans
        self.num_slice = num_slice
        
        # Cuda
        self.device = device

        # Load SAM checkpoint
        self.sam_model = sam_model_registry[self.model_type](checkpoint=self.model_checkpoint)
        
        # Load MaskGenerator
        self.mask_generator = SamAutomaticMaskGenerator(self.sam_model)
        
        self.initialize_image_encoder()
        
        # Delete sam model
        del self.sam_model
        # self.image_encoder.to(self.device)
        
        # Initialize prompt encoder 
        self.prompt_encoder_list, self.parameter_list = self.initialize_prompt_encoder()
        
        # Initialize mask decoder
        self.mask_decoder = VIT_MLAHead(img_size=96, num_classes=self.num_classess)
        # self.mask_decoder.to(self.device) 
        
        # Optimizer parameter
        self.optimizer_parameters = self.get_optimizer_parameters()
        
        
        if self.freeze_image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.freeze_prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.freeze_mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False
                
        # dice_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True, reduction="none")
        # loss_cal = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
        
    def forward(self, img, seg, spacing):
        out = F.interpolate(img.float(), scale_factor=512 / self.path_size, mode='trilinear')
        # input_batch = (out.cuda() - pixel_mean) / pixel_std
        input_batch = out.to(self.device)
        input_batch = input_batch[0].transpose(0, 1)
        batch_features, feature_list = self.image_encoder(input_batch)
        feature_list.append(batch_features)
        
        if not self.auto_finetuning:
            # feature_list = feature_list[::-1]
            l = len(torch.where(seg == 1)[0])
            points_torch = None
            if l > 0:
                sample = np.random.choice(np.arange(l), 10, replace=True)
                x = torch.where(seg == 1)[1][sample].unsqueeze(1)
                y = torch.where(seg == 1)[3][sample].unsqueeze(1)
                z = torch.where(seg == 1)[2][sample].unsqueeze(1)
                points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
                points_torch = points.to(self.device)
                points_torch = points_torch.transpose(0,1)
            l = len(torch.where(seg < 10)[0])
            sample = np.random.choice(np.arange(l), 20, replace=True)
            x = torch.where(seg < 10)[1][sample].unsqueeze(1)
            y = torch.where(seg < 10)[3][sample].unsqueeze(1)
            z = torch.where(seg < 10)[2][sample].unsqueeze(1)
            points = torch.cat([x, y, z], dim=1).unsqueeze(1).float()
            points_torch_negative = points.to(self.device)
            points_torch_negative = points_torch_negative.transpose(0, 1)
            if points_torch is not None:
                points_torch = torch.cat([points_torch, points_torch_negative], dim=1)
            else:
                points_torch = points_torch_negative
            new_feature = []
            for i, (feature, prompt_encoder) in enumerate(zip(feature_list, self.prompt_encoder_list)):
                if i == 3:
                    new_feature.append(
                        prompt_encoder(feature, points_torch.clone(), [self.path_size, self.path_size, self.path_size])
                    )
                else:
                    new_feature.append(feature)
        else:
            new_feature = feature_list

        img_resize = F.interpolate(img[:, 0].permute(0, 2, 3, 1).unsqueeze(1).to(self.device), scale_factor=64/self.path_size, # .to(device)
            mode='trilinear')
        new_feature.append(img_resize)
        masks = self.mask_decoder(new_feature, 2, self.path_size//64)
        masks = masks.permute(0, 1, 4, 2, 3)
        seg = seg.to(self.device)
        seg = seg.unsqueeze(1)
        # loss = loss_function(masks, seg)
        
        
        return img, seg, masks
        

    def get_predictor(self):
        return SamPredictor(self.sam_model)
    
    def initialize_image_encoder(self):
        # Initialize Image Encoder
        self.image_encoder = ImageEncoderViT_3d(
            depth=self.depth,
            embed_dim=self.embed_dim,
            img_size=self.img_size,
            mlp_ratio=self.mlp_ratio,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=self.num_heads,
            patch_size=self.patch_size,
            qkv_bias=self.qkv_bias,
            use_rel_pos=self.use_rel_pos,
            global_attn_indexes=self.global_attn_indexes,
            window_size=self.window_size,
            cubic_window_size=self.cubic_window_size,
            out_chans=self.out_chans,
            num_slice=self.num_slice
        )
        
        self.image_encoder.load_state_dict(self.mask_generator.predictor.model.image_encoder.state_dict(), strict=False)
        
        for p in self.image_encoder.parameters():
            p.requires_grad = False
            
        self.image_encoder.depth_embed.requires_grad = True
        
        for p in self.image_encoder.slice_embed.parameters():
            p.requires_grad = True
            
        for i in self.image_encoder.blocks:
            for p in i.norm1.parameters():
                p.requires_grad = True
            for p in i.adapter.parameters():
                p.requires_grad = True
            for p in i.norm2.parameters():
                p.requires_grad = True
            i.attn.rel_pos_d = nn.parameter.Parameter(0.5 * (i.attn.rel_pos_h + i.attn.rel_pos_w), requires_grad=True)

        for i in self.image_encoder.neck_3d:
            for p in i.parameters():
                p.requires_grad = True
        
    def initialize_prompt_encoder(self):
        prompt_encoder_list = []
        parameter_list = []
        if not self.auto_finetuning:
            for i in range(4):
                prompt_encoder = PromptEncoder(
                    transformer=TwoWayTransformer(
                        depth=2,
                        embedding_dim=256,
                        mlp_dim=2048,
                        num_heads=8
                    )
                )
                prompt_encoder.to(self.device)
                prompt_encoder_list.append(prompt_encoder)
                parameter_list.extend([j for j in prompt_encoder.parameters() if j.requires_grad == True])
            
            return prompt_encoder_list, parameter_list
        return [], []
        
    
    def get_optimizer_parameters(self):
        
        optimizer_parameters = []
        
        encoder_opt = [i for i in self.image_encoder.parameters() if i.requires_grad==True]
        feature_opt = self.parameter_list
        decoder_opt = [i for i in self.mask_decoder.parameters() if i.requires_grad == True]
        
        optimizer_parameters.extend(encoder_opt)
        optimizer_parameters.extend(feature_opt)
        optimizer_parameters.extend(decoder_opt)
        
        return optimizer_parameters
    
    def _save_checkpoint(self):
        # save_checkpoint({"epoch": epoch_num + 1,
        #                 "best_val_loss": best_loss,
        #                  "encoder_dict": img_encoder.state_dict(),
        #                  "decoder_dict": mask_decoder.state_dict(),
        #                  "feature_dict": [i.state_dict() for i in prompt_encoder_list],
        #                  "encoder_opt": encoder_opt.state_dict(),
        #                  "feature_opt": feature_opt.state_dict(),
        #                  "decoder_opt": decoder_opt.state_dict()
        #                  },
        #                 is_best=is_best,
        #                 checkpoint=args.snapshot_path)
        pass
    
    def _resume_checkpoint(self):
        # ckpt = torch.load(os.path.join(args.snapshot_path, "best.pth.tar"))

        # start_epoch = ckpt["epoch"]
        # best_loss = ckpt["best_val_loss"]
        # seg_net.load_state_dict(ckpt["network_dict"])
        # seg_net_opt.load_state_dict(ckpt["opt_dict"])
        # lr_scheduler.load_state_dict(ckpt["lr_scheduler_dict"])
        # logger.info(f"Resume training from epoch {start_epoch}!")
        # del ckpt
        # torch.cuda.empty_cache()
        pass

    def _load_checkpoint(self, snapshot_path: str, file: str):
        # Image Encoder
        self.image_encoder.load_state_dict(torch.load(os.path.join(snapshot_path, file), map_location='cpu')["encoder_dict"], strict=True)
        self.image_encoder.to(self.device)
        
        # Prompt Encoder
        if self.auto_finetuning:
            for prompt_encoder in self.prompt_encoder_list:
                prompt_encoder.load_state_dict(
                    torch.load(os.path.join(snapshot_path, file), map_location='cpu')["feature_dict"][i], strict=True)
                prompt_encoder.to(self.device)
                
        # Mask Decoder
        self.mask_decoder.load_state_dict(torch.load(os.path.join(snapshot_path, file), map_location='cpu')["decoder_dict"],
                          strict=True)
        self.mask_decoder.to(self.device)
        
        # # Testing
        # self.image_encoder.eval()
        # for i in self.prompt_encoder_list:
        #     i.eval()
        # self.mask_decoder.eval()
    
    def model_predict_auto(self, img, img_encoder, mask_decoder):
        out = F.interpolate(img.float(), scale_factor=512 / self.path_size, mode='trilinear')
        input_batch = out[0].transpose(0, 1)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)

        new_feature = feature_list
        img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(self.device), scale_factor=64/self.path_size,
                                   mode="trilinear")
        new_feature.append(img_resize)
        masks = mask_decoder(new_feature, 2, self.path_size//64)
        masks = masks.permute(0, 1, 4, 2, 3)
        masks = torch.softmax(masks, dim=1)
        masks = masks[:, 1:]
        return masks
    
    def _inference_model_predict_auto(self, test_data):
        list_masks = []
        with torch.no_grad():
            loss_summary = []
            loss_nsd = []
            for idx, (img, seg, spacing) in enumerate(test_data):
                seg = seg.float()
                seg = seg.to(self.device)
                img = img.to(self.device)
                pred = sliding_window_inference(img, [256, 256, 256], overlap=0.5, sw_batch_size=1,
                                                mode="gaussian",
                                                predictor=partial(self.model_predict_auto,
                                                                img_encoder=self.image_encoder,
                                                                mask_decoder=self.mask_decoder))
                pred = F.interpolate(pred, size=seg.shape[1:], mode="trilinear")
                seg = seg.unsqueeze(0)
                masks = pred > 0.5
                list_masks.append(masks)
            return list_masks
            # logging.info("- Test metrics Dice: " + str(np.mean(loss_summary)))
        
        
    def model_predict_prompt(self, img, prompt, img_encoder, prompt_encoder, mask_decoder):
        out = F.interpolate(img.float(), scale_factor=512 / self.patch_size, mode='trilinear')
        input_batch = out[0].transpose(0, 1)
        batch_features, feature_list = img_encoder(input_batch)
        feature_list.append(batch_features)
        #feature_list = feature_list[::-1]
        points_torch = prompt.transpose(0, 1)
        new_feature = []
        if not self.auto_finetuning:
            for i, (feature, feature_decoder) in enumerate(zip(feature_list, prompt_encoder)):
                if i == 3:
                    new_feature.append(
                        feature_decoder(feature.to(self.device), points_torch.clone(), [self.patch_size, self.patch_size, self.patch_size])
                    )
                else:
                    new_feature.append(feature.to(self.device))
        img_resize = F.interpolate(img[0, 0].permute(1, 2, 0).unsqueeze(0).unsqueeze(0).to(self.device), scale_factor=64/self.patch_size,
                                   mode="trilinear")
        new_feature.append(img_resize)
        masks = mask_decoder(new_feature, 2, self.patch_size//64)
        masks = masks.permute(0, 1, 4, 2, 3)
        return masks
    
    def _inference_model_predict_prompt(self, test_data, num_prompts: int = 1):
        # dice_loss = DiceLoss(include_background=False, softmax=False, to_onehot_y=True, reduction="none")
        with torch.no_grad():
            for idx, (img, seg, spacing) in enumerate(test_data):
                seg = seg.float()
                prompt = F.interpolate(seg[None, :, :, :, :], img.shape[2:], mode="nearest")[0]
                seg = seg.to(self.device).unsqueeze(0)
                img = img.to(self.device)
                seg_pred = torch.zeros_like(prompt).to(self.device)
                l = len(torch.where(prompt == 1)[0])
                #np.random.seed(0)
                sample = np.random.choice(np.arange(l), num_prompts, replace=True)
                #sample = sample[:3]
                x = torch.where(prompt == 1)[1][sample].unsqueeze(1)
                y = torch.where(prompt == 1)[3][sample].unsqueeze(1)
                z = torch.where(prompt == 1)[2][sample].unsqueeze(1)

                x_m = (torch.max(x) + torch.min(x)) // 2
                y_m = (torch.max(y) + torch.min(y)) // 2
                z_m = (torch.max(z) + torch.min(z)) // 2

                d_min = x_m - self.patch_size//2
                d_max = x_m + self.patch_size//2
                h_min = z_m - self.patch_size//2
                h_max = z_m + self.patch_size//2
                w_min = y_m - self.patch_size//2
                w_max = y_m + self.patch_size//2
                d_l = max(0, -d_min)
                d_r = max(0, d_max - prompt.shape[1])
                h_l = max(0, -h_min)
                h_r = max(0, h_max - prompt.shape[2])
                w_l = max(0, -w_min)
                w_r = max(0, w_max - prompt.shape[3])

                points = torch.cat([x-d_min, y-w_min, z-h_min], dim=1).unsqueeze(1).float()
                points_torch = points.to(self.device)
                d_min = max(0, d_min)
                h_min = max(0, h_min)
                w_min = max(0, w_min)
                img_patch = img[:, :,  d_min:d_max, h_min:h_max, w_min:w_max].clone()
                img_patch = F.pad(img_patch, (w_l, w_r, h_l, h_r, d_l, d_r))
                pred = self.model_predict_prompt(img_patch,
                                    points_torch,
                                    self.image_encoder,
                                    self.prompt_encoder_list,
                                    self.mask_decoder)
                pred = pred[:,:, d_l:self.patch_size-d_r, h_l:self.patch_size-h_r, w_l:self.patch_size-w_r]
                pred = F.softmax(pred, dim=1)[:,1]
                seg_pred[:, d_min:d_max, h_min:h_max, w_min:w_max] += pred

                final_pred = F.interpolate(seg_pred.unsqueeze(1), size = seg.shape[2:],  mode="trilinear")
                masks = final_pred > 0.5
                # loss = 1 - dice_loss(masks, seg)
                # logging.info(
                #     " Case {} - Dice {:.6f}".format(
                #         test_data.dataset.img_dict[idx], loss.item()
                #     ))
            # logging.info("- Test metrics Dice: " + str(np.mean(loss_summary)))

    
def save_checkpoint(state, is_best, checkpoint):
    filepath_last = os.path.join(checkpoint, "last.pth.tar")
    filepath_best = os.path.join(checkpoint, "best.pth.tar")
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Masking directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint DIrectory exists!")
    torch.save(state, filepath_last)
    if is_best:
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)
