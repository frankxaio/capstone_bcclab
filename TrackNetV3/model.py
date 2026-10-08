import torch
import torch.nn as nn

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        # Adaptive pooling shrinks the map and keeps the number of channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # ============================== TODO [Ex 1.4a] ==============================
        # Channel attention (CBAM): squeeze the spatial dims with BOTH average and
        # max pooling, pass each through the shared MLP, add them, apply sigmoid.
        # Input (B, C, H, W) -> output weights (B, C, 1, 1).
        # ============================================================================
        raise NotImplementedError("TODO [Ex 1.4a]")
 
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # Keeps the map size and reduces the number of channels
        # ============================== TODO [Ex 1.4b] ==============================
        # Spatial attention (CBAM): squeeze the CHANNEL dim with mean and max,
        # concatenate the two maps -> (B, 2, H, W), run self.conv2d, apply sigmoid.
        # Output weights (B, 1, H, W).
        # ============================================================================
        raise NotImplementedError("TODO [Ex 1.4b]")
 
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
 
    def forward(self, x):
        out = self.channel_attention(x) * x
        # out = self.spatial_attention(out) * out
        return out

class Conv2DBlock(nn.Module):
    """ Conv + ReLU + BN"""
    def __init__(self, in_dim, out_dim, kernel_size, padding='same', bias=True, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double2DConv(nn.Module):
    """ Conv2DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class Double2DConv2(nn.Module):
    """ Conv2DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double2DConv2, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (1, 1))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))

        self.conv_3 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_4 = Conv2DBlock(out_dim, out_dim, (3, 3))

        self.conv_5 = Conv2DBlock(in_dim, out_dim, (5, 5))
        self.conv_6 = Conv2DBlock(out_dim, out_dim, (3, 3))

        self.conv_7 = Conv2DBlock(out_dim*3, out_dim, (3, 3))

    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.conv_2(x1)

        x2 = self.conv_3(x)
        x2 = self.conv_4(x2)

        x3 = self.conv_5(x)
        x3 = self.conv_6(x3)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.conv_7(x)
        x = x + x2

        return x
    
class Triple2DConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim, (3, 3))
        self.conv_2 = Conv2DBlock(out_dim, out_dim, (3, 3))
        self.conv_3 = Conv2DBlock(out_dim, out_dim, (3, 3))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

class TrackNetV2(nn.Module):
    """ Original structure but less two layers 
        Total params: 10,161,411
        Trainable params: 10,153,859
        Non-trainable params: 7,552
    """
    def __init__(self, in_dim=9, out_dim=3):
        super(TrackNetV2, self).__init__()
        self.down_block_1 = Double2DConv2(in_dim=in_dim, out_dim=64)
        self.down_block_2 = Double2DConv2(in_dim=64, out_dim=128)
        self.down_block_3 = Double2DConv2(in_dim=128, out_dim=256)
        self.bottleneck = Triple2DConv(in_dim=256, out_dim=512)
        self.up_block_1 = Double2DConv(in_dim=768, out_dim=256)
        self.up_block_2 = Double2DConv(in_dim=384, out_dim=128)
        self.up_block_3 = Double2DConv(in_dim=192, out_dim=64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()
        self.cbam1 = CBAM(channel=256) #only channel attention
        self.cbam2 = CBAM(channel=128)
        self.cbam3 = CBAM(channel=64)

        self.cbam0_2 = CBAM(channel=256)
        self.cbam1_2 = CBAM(channel=128)
        self.cbam2_2 = CBAM(channel=64)

    def forward(self, x):
        """ model input shape: (F*3, 288, 512), output shape: (F, 288, 512) """
        # ============================== TODO [Ex 1.4c] ==============================
        # U-Net forward pass. Input (B, F*3, H, W) -> output heatmaps (B, F, H, W)
        # (H = 288, W = 512 in the pipeline; any H, W divisible by 8 works).
        #
        # Encoder: down_block_1 -> x1 (64ch),  nn.MaxPool2d((2, 2), stride=(2, 2)),
        #          down_block_2 -> x2 (128ch), MaxPool,
        #          down_block_3 -> x3 (256ch), MaxPool,
        #          bottleneck   -> (512ch, H/8, W/8)
        # Decoder, level k = 1, 2, 3 (skip = x3, x2, x1):
        #     1. re-weight the skip with its CBAM: cbam0_2(x3), cbam1_2(x2), cbam2_2(x1)
        #     2. upsample the current x with nn.Upsample(scale_factor=2) (nearest)
        #     3. torch.cat([upsampled_x, skip], dim=1)   <- upsampled x FIRST
        #        (channels after concat: 768, 384, 192)
        #     4. up_block_k, then cbam{k} (cbam1, cbam2, cbam3)
        # Head: predictor (1x1 conv) followed by sigmoid.
        # ============================================================================
        raise NotImplementedError("TODO [Ex 1.4c]")


# from torchsummary import summary
# Tr = TrackNetV2().cuda()
# summary(Tr, (9, 288, 512))