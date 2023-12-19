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
        self.input_conv = torch.nn.Conv2d(in_channels=in_channel, out_channels=pixelshuffle_scale**2 * output_channels, kernel_size=1)
        self.pixelshuffle = torch.nn.PixelShuffle(pixelshuffle_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        x = torch.sigmoid(self.pixelshuffle(x))  # when reg loss is not enabled
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
        x = torch.sigmoid(self.pixelshuffle(x))  # when reg loss is not enabled
        return x


class SegFeatures(torch.nn.Module):
    def __init__(self, model, decoder):
        super().__init__()
        self.model = model
        self.decoder = decoder

    def forward(self, x):
        x = self.model.forward_features(x)
        x = x[:, 5:, :]  # B, H//P x W//P, C
        x = x.permute(0, 2, 1)  # B, C, H//P x W//P
        x = x.reshape(x.shape[0], x.shape[1], 37, 37)  # B, C, H//P, W//P
        x = self.decoder(x)
        return x


def get_network(output_channels=3, dummy=False):
    # model = EfficientVit(
    #     widths=(initial_dim, initial_dim*2), depths=(1, 8), head_dim=16, num_classes=0, global_pool=None
    # )
    encoder = timm.create_model('vit_small_patch14_reg4_dinov2.lvd142m', pretrained=True, num_classes=0)
    pixelshuffle_scale = 14
    if dummy:
        decoder = DummyUpsampling(in_channel=348, pixelshuffle_scale=pixelshuffle_scale, output_channels=output_channels)
    else:
        decoder = MaskFeatureDecoder(
            in_channel=384, 
            head_width=pixelshuffle_scale**2 * output_channels,
            pixelshuffle_scale=pixelshuffle_scale,
        )
    network = SegFeatures(encoder, decoder)
    return network


def _get_processing():
    model = timm.create_model(
        "efficientvit_b0.r224_in1k",
        pretrained=False,
    )  # same layhers are the same
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    return transforms


def _test():
    transforms = _get_processing()
    network = get_network()

    img = Image.open(
        urlopen(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
        )
    )
    input = transforms(img).unsqueeze(0)  # unsqueeze single image into batch of 1
    network.eval()
    with torch.no_grad():
        output = network(input)

    print(output.shape)
    print("Success!")

def _test2():
    net = get_network()
    print(net(torch.rand(1,3,518,518)).shape)
    print('done!')