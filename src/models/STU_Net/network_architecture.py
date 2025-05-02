import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deep_supervision = True

class STUNet(nn.Module):
    def __init__(
        self, 
        input_channels: int = 1, 
        num_classes: int = 1, 
        depth: list = [1,1,1,1,1,1], 
        dims: list = [32, 64, 128, 256, 512, 512], # [32 * x for x in [1, 2, 4, 8, 16, 16]]
        pool_op_kernel_sizes: list = [[2, 2, 2],[2, 2, 2],[2, 2, 2],[2, 2, 2],[1, 1, 2]], 
        conv_kernel_sizes=[[3,3,3]] * 6, 
        enable_deep_supervision: bool = False
    ):
        super().__init__()
        self.conv_op = nn.Conv3d
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # self.final_nonlin = lambda x:x 
        self.decoder = Decoder()
        self.decoder.deep_supervision = enable_deep_supervision
        self.upscale_logits = False

        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([i // 2 for i in krnl])

        
        num_pool  = len(pool_op_kernel_sizes)
        
        assert num_pool == len(dims) - 1
        
        # encoder
        self.conv_blocks_context = nn.ModuleList()
        stage = nn.Sequential(BasicResBlock(input_channels, dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0], use_1x1conv=True), 
                              *[BasicResBlock(dims[0], dims[0], self.conv_kernel_sizes[0], self.conv_pad_sizes[0]) for _ in range(depth[0]-1)])
        self.conv_blocks_context.append(stage)
        for d in range(1, num_pool+1):
            stage = nn.Sequential(BasicResBlock(dims[d-1], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d], stride=self.pool_op_kernel_sizes[d-1], use_1x1conv=True),
                *[BasicResBlock(dims[d], dims[d], self.conv_kernel_sizes[d], self.conv_pad_sizes[d]) for _ in range(depth[d]-1)])
            self.conv_blocks_context.append(stage)

        # upsample_layers
        self.upsample_layers = nn.ModuleList()
        for u in range(num_pool):
            upsample_layer = Upsample_Layer_nearest(dims[-1-u], dims[-2-u], pool_op_kernel_sizes[-1-u])
            self.upsample_layers.append(upsample_layer)

        # decoder
        self.conv_blocks_localization = nn.ModuleList()
        for u in range(num_pool):
            stage = nn.Sequential(BasicResBlock(dims[-2-u] * 2, dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u], use_1x1conv=True),
                *[BasicResBlock(dims[-2-u], dims[-2-u], self.conv_kernel_sizes[-2-u], self.conv_pad_sizes[-2-u]) for _ in range(depth[-2-u]-1)])
            self.conv_blocks_localization.append(stage)
            
        # outputs    
        self.seg_outputs = nn.ModuleList()
        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(nn.Conv3d(dims[-2-ds], num_classes, kernel_size=1))

        # self.upscale_logits_ops = []
        # for usl in range(num_pool - 1):
        #     self.upscale_logits_ops.append(lambda x: x)
        self.upscale_logits_ops = [self.upscale_op] * (num_pool - 1)
    
    def upscale_op(self, x: torch.Tensor) -> torch.Tensor:
        return x
    
    def final_nonlin(self, x):
        return x

    def forward(self, x):
        skips = []
        seg_outputs = []
        
        # for d in range(len(self.conv_blocks_context) - 1):
        #     x = self.conv_blocks_context[d](x)
        #     skips.append(x)
        
        for idx, layer in enumerate(self.conv_blocks_context):
            if idx < len(self.conv_blocks_context) - 1:
                x = layer(x)
                skips.append(x)

        x = self.conv_blocks_context[-1](x)

        # for u in range(len(self.conv_blocks_localization)):
        #     x = self.upsample_layers[u](x)
        #     x = torch.cat((x, skips[-(u + 1)]), dim=1) 
        #     x = self.conv_blocks_localization[u](x)
        #     seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))
            
        for u, (upsample, conv, seg_output) in enumerate(zip(self.upsample_layers, self.conv_blocks_localization, self.seg_outputs)):
            x = upsample(x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = conv(x)
            seg_outputs.append(self.final_nonlin(seg_output(x)))

        if self.decoder.deep_supervision:
            # return tuple([seg_outputs[-1]] + [i(j) for i, j in
            #                                   zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(self.upscale_logits_ops[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]


class BasicResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, padding=1, stride=1, use_1x1conv=False):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1 = nn.LeakyReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        
        if use_1x1conv:
            self.conv3 = nn.Conv3d(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        y += x
        return self.act2(y)

class Upsample_Layer_nearest(nn.Module):
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        super().__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        self.mode = mode
        
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=[float(s) for s in self.pool_op_kernel_size], mode=self.mode)
        x = self.conv(x)
        return x

def load_stunet_pretrained_weights(network: nn.Module, fname: str, verbose=False):

    saved_model = torch.load(fname, weights_only=False)

    if fname.endswith('pth'):
        pretrained_dict = saved_model['network_weights']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']

    skip_strings_in_pretrained = [
        'seg_outputs',
    ]

    mod = network

    model_dict = mod.state_dict()

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

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)
    return mod

    
    
    