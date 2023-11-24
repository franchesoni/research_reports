import sys
sys.path.append("efficientvit")
import numpy as np
import torch
from PIL import Image
import cv2
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.nn import (
    IdentityLayer,
    MBConv,
    ResidualBlock,
    ConvLayer,
)


def create_synthetic_image(height, width, num_rectangles):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    masks = np.zeros((num_rectangles, height, width), dtype=np.uint8)

    for i in range(num_rectangles):
        x1, x2 = sorted(np.random.randint(width, size=2))
        y1, y2 = sorted(np.random.randint(height, size=2))
        
        random_color = tuple(int(c) for c in np.random.randint(0, 255, size=(3), dtype=np.uint8))
        image = cv2.rectangle(image, (x1, y1), (x2, y2), random_color, -1)
        masks[:, y1:y2, x1:x2] = 0
        masks[i, y1:y2, x1:x2] = 1

    return image, 0 < masks 


def custom_loss(features, masks, threshold_in=0.05, threshold_out=0.15):
    loss = 0
    means = []
    for i, mask in enumerate(masks):
        mask = torch.from_numpy(mask).to(features.device)
        features_inside_mask = features[:, :, mask.bool()]  # 1, C, M
        features_outside_mask = features[:, :, ~mask.bool()]  

        if features_inside_mask.numel() > 0:
            mean_feature = torch.mean(features_inside_mask, dim=2, keepdim=True)  # 1, C, 1
            dist_in = torch.norm(features_inside_mask - mean_feature, dim=1)
            pull_loss = torch.mean(torch.clamp(dist_in - threshold_in, min=0))
            means.append(mean_feature.squeeze())
    
        if features_outside_mask.numel() > 0:
            dist_out = torch.norm(features_outside_mask - mean_feature, dim=1)
            push_loss1 = torch.mean(torch.clamp(threshold_out - dist_out, min=0))

        loss += pull_loss + push_loss1

    if len(means) > 0:  # diff of means
        means_diff = torch.cdist(torch.stack(means)[None], torch.stack(means)[None])
        push_loss2 = torch.mean(torch.clamp(threshold_out - means_diff, min=0))
    loss += push_loss2
        
    # print(pull_loss.item(), push_loss1.item(), push_loss2.item())
    print(#pull_loss.item(), push_loss1.item(), 
        push_loss2.item())
    return loss




class MaskFeatureDecoder(torch.nn.Module):
    def __init__(
        self,
        in_channel: int=256,
        head_width: int = 256,
        head_depth: int = 4,
        expand_ratio: float = 4,
        norm="bn2d",
        act_func="hswish",
    ):
        super().__init__()
        self.input_conv = ConvLayer(in_channel, head_width, 1, norm=norm, act_func=None)

        middle = []
        for _ in range(head_depth):
            block = MBConv(
                head_width,
                head_width,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=(act_func, act_func, None),
            )
            middle.append(ResidualBlock(block, IdentityLayer()))
        self.middle = torch.nn.ModuleList(middle)
        self.pixelshuffle = torch.nn.PixelShuffle(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)
        for ind, block in enumerate(self.middle):
            x = block(x)
        x = torch.sigmoid(self.pixelshuffle(x)[:,:3])
        return x


if __name__ == '__main__':
    # Example usage
    np.random.seed(0)
    synthetic_image, masks = create_synthetic_image(512, 512, 2)
    Image.fromarray(synthetic_image).save('img.png')

    evitsam = create_sam_model(name='l0', weight_url='efficientvit/assets/checkpoints/sam/l0.pt').eval()
    decoder = MaskFeatureDecoder()

    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    epochs = 100  # Number of epochs

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Assuming synthetic_image and masks are already defined
        tinput = evitsam.transform(synthetic_image)[None]
        embedding = evitsam.image_encoder(tinput)
        decoded = decoder(embedding)
        
        # Compute loss
        loss = custom_loss(decoded, masks)

        # Backpropagation
        loss.backward()
        for name, param in decoder.named_parameters():
            if param.requires_grad:
                print(f"Gradient of {name}: {param.grad.sum()}")
        optimizer.step()

        # print(f"Epoch {epoch+1}, Loss: {loss.item()}")
        Image.fromarray((decoded[0].detach().permute(1,2,0).numpy()*255).astype(np.uint8)).save(f'output_{epoch+1}.png')