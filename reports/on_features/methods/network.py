from urllib.request import urlopen
from PIL import Image
import timm

from timm.models.efficientvit_mit import (
    MBConv,
    ResidualBlock,
    ConvNormAct,
)
import torch


class DummyUpsampling(torch.nn.Module):
    def __init__(self, in_channel, pixelshuffle_scale, output_channels):
        super().__init__()
        self.pixelshuffle_scale = pixelshuffle_scale
        self.in_channel = in_channel
        self.input_conv = torch.nn.Conv2d(
            in_channels=in_channel,
            out_channels=pixelshuffle_scale**2 * output_channels,
            kernel_size=1,
        )
        self.pixelshuffle = torch.nn.PixelShuffle(pixelshuffle_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.pixelshuffle(x)  # when reg loss is not enabled
        return x


class MaskFeatureDecoder(torch.nn.Module):
    def __init__(
        self,
        in_channel: int = 256,
        head_width: int = 256,
        head_depth: int = 4,
        expand_ratio: float = 4,
        pixelshuffle_scale: int = 8,
        norm=torch.nn.BatchNorm2d,
        act_func=torch.nn.ReLU6,
    ):
        super().__init__()
        self.input_conv = ConvNormAct(
            in_channel, head_width, 1, norm_layer=norm, act_layer=None
        )

        middle = []
        for _ in range(head_depth):
            block = MBConv(
                head_width,
                head_width,
                expand_ratio=expand_ratio,
                norm_layer=norm,
                act_layer=(act_func, act_func, None),
            )
            middle.append(ResidualBlock(block, torch.nn.Identity()))
        self.middle = torch.nn.ModuleList(middle)
        self.pixelshuffle = torch.nn.PixelShuffle(pixelshuffle_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        for ind, block in enumerate(self.middle):
            x = block(x)
        x = self.pixelshuffle(x)  # when reg loss is not enabled
        return x


class ViT_small_with_registers_forward(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        B, C, H, W = x.shape
        HoverP, WoverP = H // 14, W // 14
        x = self.model.forward_features(x)
        x = x[:, 5:, :]  # B, H//P x W//P, C
        x = x.permute(0, 2, 1)  # B, C, H//P x W//P
        x = x.reshape(x.shape[0], x.shape[1], HoverP, WoverP)  # B, C, H//P, W//P
        return x

class EffifientViT_b3_forward(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_features(x)

def get_network(output_channels=3, dummy=False, model='effvitb3'):
    assert model in ['vitregs', 'effvitb3']

    if model == 'vitregs':
        bbone = timm.create_model(
            "vit_small_patch14_reg4_dinov2.lvd142m", pretrained=False, num_classes=0, pretrained_cfg_overlay=dict(input_size=(3,224,224))
        )
        encoder = ViT_small_with_registers_forward(bbone)
        pixelshuffle_scale = 14
        in_channel = 348
    elif model == 'effvitb3':
        bbone = timm.create_model(
            "efficientvit_b3.r224_in1k", pretrained=False, num_classes=0
        )
        encoder = EffifientViT_b3_forward(bbone)
        pixelshuffle_scale = 32
        in_channel = 512

    if dummy:
        decoder = DummyUpsampling(
            in_channel=in_channel,
            pixelshuffle_scale=pixelshuffle_scale,
            output_channels=output_channels,
        )
    else:
        decoder = MaskFeatureDecoder(
            in_channel=in_channel,
            head_width=pixelshuffle_scale**2 * output_channels,
            pixelshuffle_scale=pixelshuffle_scale,
        )
    network = torch.nn.Sequential(encoder, decoder)
    return network


