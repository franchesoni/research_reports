import torch
from functools import partial
from extras.losses_utils import preprocess_masks_features, get_row_col
from extras.van_gool_loss import SpatialEmbLoss
from extras.sing import SING


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


emdloss = SpatialEmbLoss(img_size=(224,224))



def simplest_loss(features, masks, alpha=1, reg_w=None):
    """Pull features of one mask to its center. Push features external to the mask away from the center."""
    features = torch.sigmoid(features) if reg_w is None else symlog(features)
    masks, features, M, B, H, W, F = preprocess_masks_features(masks, features)

    mask_features = features * masks  # B, M, F, H*W
    # now we have to sum and divide to get the mean correctly
    mean_features = mask_features.sum(dim=3, keepdim=True) / masks.sum(
        dim=3, keepdim=True
    )  # B, M, F, 1

    # now we can mask and compute
    distances = torch.norm(features - mean_features, dim=2, p=1)  # B, M, H*W
    # pull loss: pull features to the center
    pull_loss = (distances * masks.reshape(B, M, H * W)).sum(dim=2) / masks.reshape(
        B, M, H * W
    ).sum(dim=2)
    pull_loss = pull_loss.sum(dim=1) / M  # mean over masks
    # push loss: push features away from the center
    push_loss = (distances * (~masks.reshape(B, M, H * W))).sum(dim=2) / (
        ~masks.reshape(B, M, H * W)
    ).sum(dim=2)
    push_loss = push_loss.sum(dim=1) / M  # mean over masks

    reg_loss = (
        0
        if reg_w is None
        else reg_w * torch.norm(mean_features, dim=2, p=1).sum(dim=1) / M
    )  # mean over masks
    loss = alpha * pull_loss + push_loss + reg_loss
    loss = loss.mean()  # mean over batch
    return loss


def simplest_hinge(
    features, masks, ball_radius=0.15, reg_w=None, use_push_centers=True
):
    """Pull features of one mask to its center if they're outside pull ball. Push features external to the mask away from the center if they're inside push ball. Push centers apart if they're contained in the same push centers ball."""
    if reg_w is None:
        features = torch.sigmoid(features)
    else:
        features = symlog(features)
    masks, features, M, B, H, W, F = preprocess_masks_features(masks, features)
    # assuming the number of dimensions is N=4, then the number of balls that can fit in the big ball of radius 0.5 is 0.5**N / r**N, where r is the small radius. To get a number of balls of around 10^4 we can use r = 0.05.
    # assert ball_radius == 0.05

    # define ball radii
    pull_ball_radius = ball_radius
    push_ball_radius = 2 * ball_radius
    push_centers_ball_radius = 4 * ball_radius

    mask_features = features * masks  # B, M, F, H*W
    # now we have to sum and divide to get the mean correctly
    mean_features = mask_features.sum(dim=3, keepdim=True) / masks.sum(
        dim=3, keepdim=True
    )  # B, M, F, 1

    # now we can mask and compute
    distances = torch.norm(features - mean_features, dim=2, p=1)  # B, M, H*W
    # pull loss: pull features to the center if they're outside the pull ball
    pull_loss = torch.clamp(
        distances * masks.reshape(B, M, H * W) - pull_ball_radius, min=0
    ).sum(dim=2) / masks.reshape(B, M, H * W).sum(dim=2)
    pull_loss = pull_loss.sum(dim=1) / M  # mean over masks
    # push loss: push features away from the center if they're inside the push ball
    push_loss = (
        torch.clamp(push_ball_radius - distances, min=0) * (~masks.reshape(B, M, H * W))
    ).sum(dim=2) / (~masks.reshape(B, M, H * W)).sum(dim=2)
    push_loss = push_loss.sum(dim=1) / M  # mean over masks

    if use_push_centers:
        # push centers loss: push centers away from each other if they're inside the push centers ball and the masks don't overlap
        distinct_masks = (
            (masks.reshape(B, M, 1, H * W) * masks.reshape(B, 1, M, H * W)) != 0
        ).any(
            dim=3
        )  # B, M, M
        center_distances = torch.clamp(
            push_centers_ball_radius
            - torch.cdist(
                mean_features.reshape(B, M, F), mean_features.reshape(B, M, F)
            ),
            min=0,
        )  # B, M, M, hinged
        push_centers_loss = (
            (
                center_distances
                * distinct_masks
                * (torch.eye(M, device=masks.device)[None] == 0)
            ).sum(dim=(1, 2))
            / (M * (M - 1))
            if M > 1
            else 0
        )
    else:
        push_centers_loss = 0

    ## comment out regularization loss

    reg_loss = (
        0
        if reg_w is None
        else reg_w * torch.norm(mean_features, dim=2, p=1).sum(dim=1) / M
    )  # mean over masks
    loss = pull_loss + push_loss + push_centers_loss + reg_loss
    loss = loss.mean()  # mean over batch
    return loss


def offset_to_center(features, masks, reg_w=None):
    """Try to predict as feature the center of the mask"""
    features = symlog(features)
    masks, features, M, B, H, W, F = preprocess_masks_features(masks, features)

    # features: B, 1, F, H*W
    # masks: B, M, 1, H*W
    # everything will be: B, M, F, H*W
    # get position of pixels in [0, 1]
    row, col = get_row_col(H, W, features.device)
    row = row.reshape(1, 1, 1, H, 1).expand(B, 1, 1, H, W).reshape(B, 1, 1, H * W)
    col = col.reshape(1, 1, 1, 1, W).expand(B, 1, 1, H, W).reshape(B, 1, 1, H * W)
    positional_features = torch.cat(
        [row, col] + [torch.zeros_like(row)] * (max(0, F - 2)), dim=2
    )  # B, 1, F, H*W

    # now we can mask and compute target masks which are centers of masks
    target_features = positional_features * masks  # B, M, F, H*W
    target_features = target_features.sum(dim=3, keepdim=True) / masks.sum(
        dim=3, keepdim=True
    )  # B, M, F, 1  # mean over pixels

    # now we can compute the loss
    preds = positional_features - features
    distances = torch.norm(preds - target_features, dim=2, p=1)  # B, M, H*W, distance
    loss = distances * masks.reshape(B, M, H * W)  # B, M, H*W, distance
    loss = loss.sum(dim=2) / masks.reshape(B, M, H * W).sum(dim=2)  # B, M, distance
    loss = loss.sum(dim=1) / M  # mean over masks
    loss = loss.mean()  # mean over batch
    reg_loss = 0 if reg_w is None else reg_w * torch.norm(features, dim=2, p=1).mean()
    loss = loss + reg_loss
    return loss


def global_variance(features, masks):
    features = symlog(features)
    masks, features, M, B, H, W, F = preprocess_masks_features(masks, features)
    loss = -torch.var(
        features, dim=(3)
    ).mean()  # negative variance as loss increases variance hence diversity
    return loss



losses_dict = {
    "simplest": simplest_loss,
    "hinge": simplest_hinge,
    "offset": offset_to_center,
    "vangool": emdloss,
    "global_var": global_variance,
}


def try_loss(
    loss_name,
    runname=None,
    datadir="ofdata",
    sample_index=3,
    n_iter=1000,
    out_channels=3,
    loss_kwargs={},
    clean=False,
    per_channel_optim=False,
    sing=False,
):
    assert type(loss_kwargs) == dict, "loss_kwargs must be a dict, use quotes"
    from pathlib import Path
    import json
    from extras.utils import get_current_git_commit

    if runname is None:
        from haikunator import Haikunator

        runname = Haikunator().haikunate()
    import datetime

    runname = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{runname}"
    if clean:
        clean_ascent()

    dstdir = Path("ascent") / runname
    dstdir.mkdir(exist_ok=False, parents=True)
    hparams = {
        "loss_name": loss_name,
        "runname": runname,
        "datadir": datadir,
        "sample_index": sample_index,
        "n_iter": n_iter,
        "out_channels": out_channels,
        "loss_kwargs": loss_kwargs,
        "git_commit": get_current_git_commit(),
    }
    with open(dstdir / "hparams.json", "w") as f:
        json.dump(hparams, f, indent=4)

    loss_fn = losses_dict[loss_name]
    from data import get_train_val_ds, custom_collate
    from trainer import save_tensor_as_image

    print("getting dataloaders")
    train_ds, _ = get_train_val_ds(datadir)
    train_batch = custom_collate([train_ds[sample_index]])
    image, masks = train_batch
    save_tensor_as_image(dstdir / "image.jpg", image[0], 0)

    # do gradient ascent from tensor
    if per_channel_optim:
        channels_as_params = [torch.randn(size=(1, 1, 224, 224), requires_grad=True) for c in range(out_channels)]
        params_list = [{"params": channel} for channel in channels_as_params]
        optimizer = SING(params_list, lr=1) if sing else torch.optim.AdamW(params_list, lr=1)
        output = torch.cat(channels_as_params, dim=1)
    else:
        output = torch.randn(size=(1, out_channels, 224, 224), requires_grad=True)
        optimizer = torch.optim.Adam([output], lr=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=n_iter // 2, verbose=True
    )

    for i in range(n_iter):
        loss = loss_fn(output, masks, **loss_kwargs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        print(f"{i} / {n_iter}, loss: {loss.item():.4e}", end="\r")

        if i % (n_iter // 10) == 0:
            save_tensor_as_image(dstdir / "ascent.jpg", output[0], i)

    print("Done")
    with open(dstdir / "done.txt", "w") as f:
        f.write("Done")


def clean_ascent():
    import shutil
    import datetime
    from pathlib import Path

    dstdir = Path("ascent")
    for f in dstdir.iterdir():
        # if the directory doesn't have a done.txt file remove it
        if f.is_dir() and not (f / "done.txt").exists():
            shutil.rmtree(f)


if __name__ == "__main__":
    from fire import Fire

    Fire(try_loss)
