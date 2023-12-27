import torch

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

