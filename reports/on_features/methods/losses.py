import torch
from functools import partial

def preprocess_masks_features(masks, features):
    # Get shapes right
    B, M, H, W = masks.shape
    Bf, F, Hf, Wf = features.shape
    assert H == Hf and W == Wf and B == Bf
    masks = masks.reshape(B, M, 1, H * W)
    assert masks.dtype == torch.bool

    # Reduce M if there are empty masks
    mask_areas = masks.sum(dim=3)  # B, M, 1
    if mask_areas.min() == 0:
        m_ind_is_zero = (mask_areas == 0).any(dim=0).squeeze()  # M
        masks = masks[:, ~m_ind_is_zero]
        M = masks.shape[1]
        del mask_areas

    features = features.reshape(B, 1, F, H * W)
    # output shapes
    # features: B, 1, F, H*W
    # masks: B, M, 1, H*W

    return masks, features, M, B, H, W, F

def get_row_col(H, W, device):
    # get position of pixels in [0, 1]
    row = torch.linspace(0, 1, H, device=device)
    col = torch.linspace(0, 1, W, device=device)
    return row, col
    


def simplest_loss(features, masks, alpha=1e-3):
    """Pull features of one mask to its center. Push features external to the mask away from the center."""
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

    loss = alpha * pull_loss + push_loss
    loss = loss.mean()  # mean over batch
    return loss


def simplest_hinge(features, masks, ball_radius=0.05):
    """Pull features of one mask to its center if they're outside pull ball. Push features external to the mask away from the center if they're inside push ball. Push centers apart if they're contained in the same push centers ball."""
    masks, features, M, B, H, W, F = preprocess_masks_features(masks, features)
    # assuming the number of dimensions is N=4, then the number of balls that can fit in the big ball of radius 0.5 is 0.5**N / r**N, where r is the small radius. To get a number of balls of around 10^4 we can use r = 0.05.
    assert ball_radius == 0.05
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
    # push centers loss: push centers away from each other if they're inside the push centers ball and the masks don't overlap
    distinct_masks = (
        (masks.reshape(B, M, 1, H * W) * masks.reshape(B, 1, M, H * W)) != 0
    ).any(
        dim=3
    )  # B, M, M
    center_distances = torch.clamp(
        push_centers_ball_radius
        - torch.cdist(mean_features.reshape(B, M, F), mean_features.reshape(B, M, F)),
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

    ## comment out regularization loss
    # reg_loss = torch.norm(mean_features, dim=2, p=1).sum(dim=1) / M  # mean over masks
    loss = pull_loss + push_loss + push_centers_loss  # + 0.001 * reg_loss
    loss = loss.mean()  # mean over batch
    return loss

def offset_to_center(features, masks):
    """Try to predict as feature the center of the mask"""
    masks, features, M, B, H, W, F = preprocess_masks_features(masks, features)

    # features: B, 1, F, H*W
    # masks: B, M, 1, H*W
    # everything will be: B, M, F, H*W
    # get position of pixels in [0, 1]
    row, col = get_row_col(H, W, features.device)  
    row = (row.reshape(1, 1, 1, H, 1)
        .expand(B, 1, 1, H, W)
        .reshape(B, 1, 1, H * W))
    col = (col.reshape(1, 1, 1, 1, W)
        .expand(B, 1, 1, H, W)
        .reshape(B, 1, 1, H * W))
    positional_features = torch.cat([row, col] + [torch.zeros_like(row)]*(max(0, F-2)), dim=2)  # B, 1, F, H*W

    # now we can mask and compute target masks which are centers of masks
    target_features = positional_features * masks  # B, M, F, H*W
    target_features = target_features.sum(dim=3, keepdim=True) / masks.sum(
        dim=3, keepdim=True
    )  # B, M, F, 1  # mean over pixels

    # now we can compute the loss
    preds = (positional_features - features)
    distances = torch.norm(preds - target_features, dim=2, p=1)  # B, M, H*W, distance
    loss = distances * masks.reshape(B, M, H * W)  # B, M, H*W, distance
    loss = loss.sum(dim=2) / masks.reshape(B, M, H * W).sum(dim=2)  # B, M, distance
    loss = loss.sum(dim=1) / M  # mean over masks
    loss = loss.mean()  # mean over batch
    return loss

def global_variance(features, masks):
    masks, features, M, B, H, W, F = preprocess_masks_features(masks, features)
    loss = - torch.var(features, dim=(3)).mean()  # negative variance as loss increases variance hence diversity
    return loss




##########################################################

# def multiscale_loss(features, masks, shades_of_gray=5, use_min=True):
#     push_ball_radius = 1 / shades_of_gray / 2  # determines the size of balls
#     pull_ball_radius = push_ball_radius / 2

#     masks, features, M, B, H, W, F = preprocess_masks_features(masks, features)
#     # sqrt(2) / 2**i * s = pull_radius * 2 for channel i. This means that a proportion of the max sized object is the diameter of the pull ball. Therefore s = pull_radius * 2**(i+1) / sqrt(2)
#     scales = torch.Tensor(
#         [pull_ball_radius * 2 ** (i + 1) / (2**0.5) for i in range(F)]
#     ).to(
#         masks.device
#     )  # F
#     # get position of pixels in [0, 1]
#     row = torch.linspace(0, 1, H, device=masks.device)
#     col = torch.linspace(0, 1, W, device=masks.device)
#     # now create features of shape B, 1, F, 3, H*W where 3 contains (f, row, col)
#     features = features.reshape(B, 1, F, 1, H * W)
#     row = (
#         row.reshape(1, 1, 1, 1, H, 1)
#         .expand(B, 1, F, 1, H, W)
#         .reshape(B, 1, F, 1, H * W)
#     )
#     col = (
#         col.reshape(1, 1, 1, 1, 1, W)
#         .expand(B, 1, F, 1, H, W)
#         .reshape(B, 1, F, 1, H * W)
#     )
#     row, col = row * scales.reshape(1, 1, F, 1, 1), col * scales.reshape(1, 1, F, 1, 1)
#     features = torch.cat([features, row, col], dim=3)  # B, 1, F, 3, H*W
#     masks = masks.reshape(B, M, 1, 1, H * W)  # expand to account for position

#     # now compute the loss on each feature channel
#     mask_features = features * masks  # B, M, F, 3, H*W
#     mean_features = mask_features.sum(dim=4, keepdim=True) / masks.sum(
#         dim=4, keepdim=True
#     )  # B, M, F, 3, 1
#     # now we can mask and compute
#     distances = torch.norm(features - mean_features, dim=3, p=1)  # B, M, F, H*W
#     # pull loss: pull features to the center if they're outside the pull ball
#     pull_loss = torch.clamp(
#         distances * masks.reshape(B, M, 1, H * W) - pull_ball_radius, min=0
#     ).sum(dim=3) / masks.reshape(B, M, 1, H * W).sum(
#         dim=3
#     )  # B, M, F
#     # push loss: push features away from the center if they're inside the push ball
#     push_loss = (
#         torch.clamp(push_ball_radius - distances, min=0)
#         * (~masks.reshape(B, M, 1, H * W))
#     ).sum(dim=3) / (~masks.reshape(B, M, 1, H * W)).sum(
#         dim=3
#     )  # B, M, F

#     loss = pull_loss + push_loss  # B, M, F
#     if use_min:
#         loss = torch.min(loss, dim=2)[
#             0
#         ]  # B, M  # take the best channel for each mask over the feature channels
#     else:
#         loss = torch.log(loss).sum(dim=2) / F  # B, M  # mean over feature channels
#     loss = loss.sum(dim=1) / M  # mean over masks
#     loss = loss.mean()  # mean over batch
#     return loss + 10  # translate to make it positive


# def hier_eucl_loss(features, masks):
#     """Pull features of one mask to its center if they're outside pull ball. Push features external to the mask away from the center if they're inside push ball. Push centers apart if they're contained in the same push centers ball. Change radii proportionally to area."""
#     masks, features, M, B, H, W, F = preprocess_masks_features(masks, features)

#     # define ball radii
#     ball_radius = 0.25  # this is the internal radius when the size is max, i.e. 512
#     norm_areas = masks.sum(dim=(2, 3)) / (512**2)
#     pull_ball_radius = ball_radius * norm_areas  # B, M
#     push_ball_radius = 2 * ball_radius * norm_areas
#     push_centers_ball_radius = 4 * ball_radius * norm_areas

#     mask_features = features * masks  # B, M, F, H*W
#     # now we have to sum and divide to get the mean correctly
#     mean_features = mask_features.sum(dim=3, keepdim=True) / masks.sum(
#         dim=3, keepdim=True
#     )  # B, M, F, 1

#     # now we can mask and compute
#     distances = torch.norm(features - mean_features, dim=2, p=1)  # B, M, H*W
#     # pull loss: pull features to the center if they're outside the pull ball
#     pull_loss = torch.clamp(
#         distances * masks.reshape(B, M, H * W) - pull_ball_radius.reshape(B, M, 1),
#         min=0,
#     ).sum(dim=2) / masks.reshape(B, M, H * W).sum(dim=2)
#     pull_loss = pull_loss.sum(dim=1) / M  # mean over masks
#     # push loss: push features away from the center if they're inside the push ball
#     push_loss = (
#         torch.clamp(push_ball_radius.reshape(B, M, 1) - distances, min=0)
#         * (~masks.reshape(B, M, H * W))
#     ).sum(dim=2) / (~masks.reshape(B, M, H * W)).sum(dim=2)
#     push_loss = push_loss.sum(dim=1) / M  # mean over masks
#     # push centers loss: push centers away from each other if they're inside the push centers ball and the masks don't overlap
#     distinct_masks = (
#         (masks.reshape(B, M, 1, H * W) * masks.reshape(B, 1, M, H * W)) != 0
#     ).any(
#         dim=3
#     )  # B, M, M
#     center_distances = torch.clamp(
#         push_centers_ball_radius.reshape(B, M, 1)
#         - torch.cdist(mean_features.reshape(B, M, F), mean_features.reshape(B, M, F)),
#         min=0,
#     )  # B, M, M, hinged
#     push_centers_loss = (
#         (
#             center_distances
#             * distinct_masks
#             * (torch.eye(M, device=masks.device)[None] == 0)
#         ).sum(dim=(1, 2))
#         / (M * (M - 1))
#         if M > 1
#         else 0
#     )

#     # reg_loss = torch.norm(mean_features, dim=2, p=1).sum(dim=1) / M  # mean over masks
#     loss = pull_loss + push_loss + push_centers_loss  # + 0.001 * reg_loss
#     loss = loss.mean()  # mean over batch
#     return loss


# losses_dict = {'basic': basic_loss,
#                'hier_eucl': hier_eucl_loss,
#                'multiscale_min': multiscale_loss,
#                'multiscale_mean': partial(multiscale_loss, use_min=False),
#                }

losses_dict = {"simplest": simplest_loss,
               "hinge": simplest_hinge,
               "offset": offset_to_center,
               "global_var": global_variance,}
