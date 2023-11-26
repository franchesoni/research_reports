import json
import shutil
import tqdm
from pathlib import Path
from pycocotools import mask as mask_utils
from PIL import Image, ImageOps
import numpy as np
import torchvision.transforms as transforms
import torch
import sys
sys.path.append("efficientvit")
from efficientvit.sam_model_zoo import create_sam_model
from efficientvit.models.nn import (
    IdentityLayer,
    MBConv,
    ResidualBlock,
    ConvLayer,
)
import pytorch_lightning as pl



def minmaxnorm(x):
    return (x - x.min()) / (x.max() - x.min())

        

def pad_resized_img(pil_img, size=512):
    h, w = pil_img.height, pil_img.width
    longest_side = max(h, w)
    scale_ratio = size / longest_side
    new_h, new_w = int(h * scale_ratio), int(w * scale_ratio)
    pil_img = pil_img.resize(
size=(new_w, new_h))  # bicubic, default antialiasing (idk which)
    pil_img = ImageOps.expand(pil_img, border=(0, 0, size - new_w, size - new_h), fill=0)  # 0 pad
    return pil_img

def prepare_images():
    samdata_dir = Path('/home/franchesoni/adisk/samdata')
    dstdir = Path('/home/franchesoni/adisk/samdata_512')
    if dstdir.exists():
        shutil.rmtree(dstdir)
    dstdir.mkdir()

    image_names = sorted([f for f in samdata_dir.glob('sa_*.jpg')])
    label_names = sorted([f for f in samdata_dir.glob('sa_*.json')])
    assert all([img.stem == label.stem for img, label in zip(image_names, label_names)])
    for idx in tqdm.tqdm(range(len(image_names))):
        name = image_names[idx].stem
        img = Image.open(image_names[idx])
        anns = json.load(open(label_names[idx]))['annotations']
        # sort by area (decreasing)
        anns = sorted(anns, key=lambda ann: ann['area'], reverse=True)
        masks = [mask_utils.decode(ann['segmentation'])>0 for ann in anns[:10]]
        outimg = pad_resized_img(img)
        outimg.save(dstdir / f'{name}_img.png')
        for mind, mask in enumerate(masks):
            outm = pad_resized_img(Image.fromarray(mask))
            outm.save(dstdir / f'{name}_{mind}_mask.png')


def load_one_sample(dirpath, stem):
    img = Image.open(dirpath / f'{stem}_img.png')
    masks = [Image.open(dirpath / f'{stem}_{mind}_mask.png') for mind in range(10)]
    return img, masks

class SAM512Dataset(torch.utils.data.Dataset):
    def __init__(self, dirpath, split='train'):
        self.dirpath = Path(dirpath)
        self.split = split
        assert self.split in ['train', 'val', 'trainval']
        self.image_names = sorted([f for f in self.dirpath.glob('sa_*_img.png') if (f.parent / f"sa_{f.stem.split('_')[1]}_9_mask.png").exists()])
        self.label_names = sorted([f for f in self.dirpath.glob('sa_*_9_mask.png')])
        assert all([img.stem.split('_')[1] == label.stem.split('_')[1] for img, label in zip(self.image_names, self.label_names)])
        if self.split == 'train':
            self.image_names = self.image_names[:-min(len(self.image_names)//10, 1000)]
            self.label_names = self.label_names[:-min(len(self.label_names)//10, 1000)]
        elif self.split == 'val':
            self.image_names = self.image_names[-min(len(self.image_names)//10, 1000):]
            self.label_names = self.label_names[-min(len(self.label_names)//10, 1000):]  # only 10% of the masks or max 1000 
        elif self.split == 'trainval':
            pass  # use all
        self.num_samples = len(self.image_names)
        self.totensor = transforms.ToTensor()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        img, masks = load_one_sample(self.dirpath, 'sa_' + self.image_names[idx].stem.split('_')[1])
        img = self.totensor(img)
        img = img / img.max()
        masks = torch.from_numpy(np.array([np.array(mask) for mask in masks]))
        return img, masks


def custom_loss(features, masks, ball_radius=0.02):
    """Pull features of one mask to its center if they're outside pull ball. Push features external to the mask away from the center if they're inside push ball. Push centers apart if they're contained in the same push centers ball."""
    # define ball radii
    pull_ball_radius = ball_radius
    push_ball_radius = 2 * ball_radius
    push_centers_ball_radius = 4 * ball_radius
    # get shapes right
    B, M, H, W = masks.shape
    Bf, F, Hf, Wf = features.shape
    assert H == Hf and W == Wf and B == Bf
    masks = masks.view(B, M, 1, H*W)
    features = features.view(B, 1, F, H*W)

    mask_features = features * masks  # B, M, F, H*W
    # now we have to sum and divide to get the mean correctly
    mean_features = mask_features.sum(dim=3, keepdim=True) / masks.sum(dim=3, keepdim=True)  # B, M, F, 1

    # now we can mask and compute
    distances = torch.norm(features - mean_features, dim=2, p=1)  # B, M, H*W
    # pull loss: pull features to the center if they're outside the pull ball
    pull_loss = torch.clamp(distances * masks.view(B, M, H*W) - pull_ball_radius, min=0).sum(dim=2) / masks.view(B, M, H*W).sum(dim=2)
    pull_loss = pull_loss.sum(dim=1) / M  # mean over masks
    # push loss: push features away from the center if they're inside the push ball
    push_loss = (torch.clamp(push_ball_radius - distances, min=0) * (~masks.view(B, M, H*W))).sum(dim=2) / (~masks.view(B, M, H*W)).sum(dim=2)
    push_loss = push_loss.sum(dim=1) / M  # mean over masks
    # push centers loss: push centers away from each other if they're inside the push centers ball
    push_centers_loss = (torch.clamp(push_centers_ball_radius - torch.cdist(mean_features.view(B, M, F), mean_features.view(B, M, F)), min=0) * (torch.eye(M, device=masks.device)[None] == 0)).sum(dim=(1,2)) / (M*(M-1))

    loss = pull_loss + push_loss + push_centers_loss
    loss = loss.mean()  # mean over batch
    return loss





class MaskFeatureDecoder(torch.nn.Module):
    def __init__(
        self,
        in_channel: int = 256,
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
        x = torch.sigmoid(self.pixelshuffle(x))
        return x

class MySAFeats(torch.nn.Module):
    def __init__(self, weight_url='efficientvit/assets/checkpoints/sam/l0.pt'):
        super().__init__()
        self.encoder = create_sam_model(name='l0', weight_url=weight_url).train().image_encoder
        self.decoder = MaskFeatureDecoder()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class PLModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = custom_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = custom_loss(y_hat, y)
        self.log('val_loss', loss)
        # save image
        if batch_idx == 0:
            self.logger.experiment.add_image('val_img', x[0], self.current_epoch)
            self.logger.experiment.add_image('val_pred', y_hat[0, :3], self.current_epoch)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == '__main__':
    dirpath = Path('/home/franchesoni/adisk/samdata_512')
    batch_size = 8
    train_ds = SAM512Dataset(dirpath, split='train')
    val_ds = SAM512Dataset(dirpath, split='val')
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    model = MySAFeats()
    plmodel = PLModule(model)
    trainer = pl.Trainer(gpus=1, max_epochs=1000000,)# fast_dev_run=True, overfit_batches=4, log_every_n_steps=1)
    trainer.fit(plmodel, train_dataloaders=train_dl, val_dataloaders=val_dl)


# i've got half an hour
# create the dataset
    # resize longest to 512
    # pad to 512x512
    # do the same with the masks
    # keep only 10 biggest masks
# optimize the loss
# create the model (efficientvit + pixelshuffle)
# create the dataloader
# use pl for training
