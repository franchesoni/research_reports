from __future__ import print_function, division

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from itertools import filterfalse as ifilterfalse
from extras.losses_utils import preprocess_masks_features, symlog

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts.float() - gt_sorted.float().cumsum(0)
    union = gts.float() + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


# --------------------------- HELPER FUNCTIONS ---------------------------


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc

    return acc / n


class SpatialEmbLoss(nn.Module):
    def __init__(
        self, to_center=True, n_sigma=1, foreground_weight=1, img_size=(1024, 2048), device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()

        print(
            "Created spatial emb loss function with: to_center: {}, n_sigma: {}, foreground_weight: {}".format(
                to_center, n_sigma, foreground_weight
            )
        )

        self.to_center = to_center
        self.n_sigma = n_sigma
        self.foreground_weight = foreground_weight

        # coordinate map
        n_rows, n_cols = img_size
        cols = torch.linspace(0, 1, n_cols).view(1, 1, -1).expand(1, n_rows, n_cols)
        rows = torch.linspace(0, 1, n_rows).view(1, -1, 1).expand(1, n_rows, n_cols)
        xym = torch.cat((rows, cols), 0)
        xym = xym.to(device)
        xym.requires_grad = False
        self.register_buffer("xym", xym)

    def forward(
        self,
        features,
        masks,
        per_instance=True,
        w_inst=1,
        w_var=0.01,
        w_seed=0.01,
        print_iou=False,
        exp_s=True,
        print_intermediate=False,
    ):
        masks, features, M, B, H, W, F = preprocess_masks_features(masks, features)
        # features: B, 1, F, H*W
        # masks: B, M, 1, H*W

        xym_s = self.xym[..., 0:H, 0:W].reshape(1, 1, 2, -1)  # 1, 1, 2, H*W
        spatial_emb = symlog(features[:, :, 0:2]) + xym_s  # B, 1, 2, H*W
        sigma = features[:, :, 2 : 2 + self.n_sigma]  # B, 1, n_sigma, H*W
        seed_map = torch.sigmoid(
            features[:, :, 2 + self.n_sigma : 2 + self.n_sigma + 1]
        )  # B, 1, 1, H*W

        if self.to_center:
            centers = (xym_s * masks).sum(dim=3, keepdim=True) / (
                masks.sum(dim=3, keepdim=True) + 1e-6
            )  # B, M, 2, 1
        else:
            centers = (spatial_emb * masks).sum(dim=3, keepdim=True) / (
                masks.sum(dim=3, keepdim=True) + 1e-6
            )  # B, M, 2, 1

        # calculate sigma
        s = (sigma * masks).sum(dim=3, keepdim=True) / (
            masks.sum(dim=3, keepdim=True) + 1e-6
        )  # B, M, n_sigma, 1

        var_loss = torch.mean(torch.pow(sigma * masks - s.detach(), 2))

        if print_intermediate and exp_s:
            print("s before exp", torch.norm(s))

        s = torch.exp(s * 10) if exp_s else s

        # calculate gaussian
        dist = torch.exp(
            -1 * torch.sum(torch.pow(spatial_emb - centers, 2) * s, 2, keepdim=True)
        )  # B, M, 1, H*W

        # apply lovasz-hinge loss
        logits, targets = (2 * dist - 1).reshape(B * M, H, W), masks.reshape(
            B * M, H, W
        )*1
        instance_loss = lovasz_hinge(logits, targets, per_image=per_instance)

        # seed loss
        seed_loss = self.foreground_weight * (
            torch.pow((seed_map - dist.detach()) * masks, 2)
        ).mean(
            dim=3, keepdim=True
        )  # B, M, 1, 1

        loss = w_inst * instance_loss + w_var * var_loss + w_seed * seed_loss.mean()

        if print_intermediate:
            print("instance_loss", torch.norm(instance_loss))
            print("seed_loss", torch.norm(seed_loss))
            print("var_loss", torch.norm(var_loss))
            print("iou", calculate_iou(dist > 0.5, masks))
            print("total loss", loss)
            breakpoint()

        if print_iou:
            print(calculate_iou(dist > 0.5, masks))

        return loss


def calculate_iou(pred, label):
    intersection = ((label == 1) & (pred == 1)).sum()
    union = ((label == 1) | (pred == 1)).sum()
    if not union:
        return 0
    else:
        iou = intersection.item() / union.item()
        return iou

def save(tensor, name):
    from PIL import Image

    def minmaxnorm(x):
        return (x - x.min()) / (x.max() - x.min())
    
    tensor = minmaxnorm(tensor)
    tensor = (tensor * 255).to(torch.uint8)
    tensor = tensor.squeeze()  # C, H*W
    tensor = tensor.reshape(-1, 224, 224)  # C, H, W
    if tensor.shape[0] == 1:
        tensor = tensor[0]
    elif tensor.shape[0] == 2:
        tensor = torch.stack([tensor[0], torch.zeros_like(tensor[0]), tensor[1]], dim=0)
        tensor = tensor.permute(1, 2, 0)
    elif tensor.shape[0] >= 3:
        tensor = tensor[:3]
        tensor = tensor.permute(1, 2, 0)
    tensor = tensor.cpu().numpy()
    Image.fromarray(tensor).save(name)

