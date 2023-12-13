# synthetic data
from pathlib import Path
import random
import os
import tqdm
import cv2
import numpy as np
from skimage import measure

def generate_rectangles(image_size, num_rectangles):
    """
    Generate random rectangles within the given image size.

    Returns a list of rectangles where each rectangle is represented as 
    (x, y, width, height).
    """
    rectangles = []
    for _ in range(num_rectangles):
        x = np.random.randint(0, image_size[1])
        y = np.random.randint(0, image_size[0])
        width = np.random.randint(10, image_size[1] // 4)
        height = np.random.randint(10, image_size[0] // 4)
        rectangles.append((x, y, width, height))
    return rectangles


def generate_dataset_v2(datadir, num_images, image_size=(224, 224), num_rectangles_choices=[3, 4, 5, 6]):
    """
    Generate a synthetic dataset of images with rectangles and their corresponding masks.
    The number of rectangles per image is chosen randomly from 'num_rectangles_choices'.
    Occasionally, rectangles will have the exact same color.
    """
    datadir = Path(datadir)
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

        # Occasionally use the same color for different rectangles
        use_same_color = np.random.choice([True, False], p=[0.2, 0.8])  # 20% chance to use the same color
        if use_same_color:
            common_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

        for j, rect in enumerate(rectangles):
            x, y, width, height = rect
            color = common_color if use_same_color else (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
            
            # Draw the rectangle on the image and mask
            cv2.rectangle(image, (x, y), (x + width, y + height), color, -1)
            cv2.rectangle(mask, (x, y), (x + width, y + height), (j + 1), -1)  # Each rectangle gets a unique mask value

        # Remove mask regions for overlapped rectangles
        labels = measure.label(mask)
        # Save the image and mask
        cv2.imwrite(os.path.join(datadir, f'image_{str(i).zfill(strlen)}.png'), image)
        cv2.imwrite(os.path.join(datadir, f'mask_{str(i).zfill(strlen)}.png'), labels)

if __name__ == '__main__':
    from fire import Fire
    Fire(generate_dataset_v2)

# Example usage
# `python data.py datadir 10000``
