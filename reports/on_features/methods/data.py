# synthetic data
import shutil
import copy
import pickle
from pathlib import Path
import random
import os
import tqdm
import cv2
import numpy as np
import torch
from PIL import ImageOps
from skimage import measure
from skimage.filters import gaussian


def generate_rectangles(image_size, num_rectangles):
    """
    Generate random rectangles within the given image size.

    Returns a list of rectangles where each rectangle is represented as
    (x, y, width, height).
    """
    rectangles = []
    for _ in range(num_rectangles):
        # get four random corners
        x1, x2 = np.random.randint(0, image_size[1], size=2)  # col
        y1, y2 = np.random.randint(0, image_size[0], size=2)  # row
        # deambiguate
        if x1 == x2:
            x1 = max(0, x1 - 10)
            x2 = min(image_size[1] - 1, x2 + 10)
        if y1 == y2:
            y1 = max(0, y1 - 10)
            y2 = min(image_size[0] - 1, y2 + 10)
        # sort
        xl, xr = min(x1, x2), max(x1, x2)
        yt, yb = min(y1, y2), max(y1, y2)
        # add to list
        width = xr - xl
        height = yb - yt
        rectangles.append((xl, yt, width, height))
    return rectangles


def generate_dataset_v2(
    datadir, num_images, image_size=518, num_rect=[6, 7, 8, 9], sigma_blur=5, reset=False
):
    """
    Generate a synthetic dataset of images with rectangles and their corresponding masks.
    The number of rectangles per image is chosen randomly from 'num_rectangles_choices'.
    Occasionally, rectangles will have the exact same color.
    """
    image_size = (image_size, image_size)
    num_rectangles_choices = num_rect
    datadir = Path(datadir)
    if reset and datadir.exists():
        shutil.rmtree(datadir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    strlen = len(str(num_images))
    for i in tqdm.tqdm(range(num_images)):
        # Create an empty image and mask
        image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
        mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)

        # Randomly choose the number of rectangles for this image
        num_rectangles = random.choice(num_rectangles_choices)
        rectangles = generate_rectangles(image_size, num_rectangles)

        common_color = (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
        )  # for some repeated rectangles
        common_color = (
            common_color / np.linalg.norm(common_color, ord=1) * 255
        ).astype(int)
        # keep track of where same color can't be used
        for j, rect in enumerate(rectangles):
            x, y, width, height = rect
            if (
                image[max(0, y - 1) : y + height + 1, max(0, x - 1) : x + width + 1]
                != common_color[None, None]
            ).all():  # if common color is not present, use same color
                color = common_color
            else:
                color = (
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                    np.random.randint(0, 256),
                )
                color = (color / np.linalg.norm(color, ord=1) * 255).astype(int)
            color = tuple([int(c) for c in color])

            # Draw the rectangle on the image and mask
            cv2.rectangle(image, (x, y), (x + width, y + height), color, -1)
            cv2.rectangle(
                mask, (x, y), (x + width, y + height), (j + 1), -1
            )  # Each rectangle gets a unique mask value

        # Remove mask regions for overlapped rectangles
        labels = measure.label(mask)
        if sigma_blur > 0:
            image = (gaussian(image, sigma=sigma_blur) * 255).astype(np.uint8)
        # Save the image and mask
        cv2.imwrite(os.path.join(datadir, f"image_{str(i).zfill(strlen)}.png"), image)
        cv2.imwrite(os.path.join(datadir, f"mask_{str(i).zfill(strlen)}.png"), labels)


class RectangleDataset(torch.utils.data.Dataset):
    def __init__(self, datadir):
        self.datadir = Path(datadir)
        image_files = sorted(self.datadir.glob("image_*.png"))
        mask_files = sorted(self.datadir.glob("mask_*.png"))
        assert len(image_files) == len(mask_files)
        # check that the image and mask files match
        assert all(
            [
                image_file.stem.split("_")[-1] == mask_file.stem.split("_")[-1]
                for image_file, mask_file in zip(image_files, mask_files)
            ]
        )
        self.sample_paths = list(zip(image_files, mask_files))

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        img_path, mask_path = self.sample_paths[idx]
        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        sample = self.to_tensor((image, mask))
        return sample

    def mask_as_batch(self, mask):
        """Convert mask from a gray image to a batch of binary masks"""
        return (
            torch.nn.functional.one_hot(torch.from_numpy(mask).to(torch.int64)) > 0
        ).permute(2, 0, 1)[1:]

    def to_tensor(self, sample):
        """Convert sample from numpy to torch"""
        image, mask = sample
        image = torch.from_numpy(image / 255).permute(2, 0, 1).float()
        mask = self.mask_as_batch(mask)
        return (image, mask)


def get_train_val_ds(datadir):
    traindsfile, valdsfile = Path(f"runs/trainds.pkl"), Path(
        f"runs/valds.pkl"
    )
    traindsfile.parent.mkdir(exist_ok=True, parents=True)
    if traindsfile.exists() and valdsfile.exists():
        with open(traindsfile, "rb") as f:
            train_ds = pickle.load(f)
        with open(valdsfile, "rb") as f:
            val_ds = pickle.load(f)
        return train_ds, val_ds

    train_ds = RectangleDataset(datadir)
    val_ds = copy.deepcopy(train_ds)
    train_ds.sample_paths = train_ds.sample_paths[
        : -min(len(train_ds.sample_paths) // 10, 1000)
    ]
    val_ds.sample_paths = val_ds.sample_paths[
        -min(len(val_ds.sample_paths) // 10, 1000) :
    ]
    with open(traindsfile, "wb") as f:
        pickle.dump(train_ds, f)
    with open(valdsfile, "wb") as f:
        pickle.dump(val_ds, f)
    return train_ds, val_ds


def pad_resized_img(pil_img, size=224):
    h, w = pil_img.height, pil_img.width
    longest_side = max(h, w)
    scale_ratio = size / longest_side
    new_h, new_w = int(h * scale_ratio), int(w * scale_ratio)
    pil_img = pil_img.resize(
        size=(new_w, new_h)
    )  # bicubic, default antialiasing (idk which)
    pil_img = ImageOps.expand(
        pil_img, border=(0, 0, size - new_w, size - new_h), fill=0
    )  # 0 pad
    return pil_img


def custom_collate(batch):
    images, masks = zip(*batch)
    n_masks = min([mask.shape[0] for mask in masks])
    masks = [mask[:n_masks] for mask in masks]
    images = torch.stack(images)
    masks = torch.stack(masks)
    return (images, masks)

if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset_v2)

# Example usage
# `python data.py datadir 99999 --reset`
