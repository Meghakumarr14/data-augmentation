import os
import cv2
import numpy as np
from albumentations import (
    HorizontalFlip, VerticalFlip, RandomRotate90, ElasticTransform,
    GridDistortion, OpticalDistortion, RandomBrightnessContrast,
    ShiftScaleRotate, Compose
)
from albumentations.augmentations.transforms import Normalize
from albumentations.core.composition import OneOf

# Set paths
IMAGE_DIR = 'images/'
MASK_DIR = 'masks/'
AUG_IMAGE_DIR = 'augmented/images/'
AUG_MASK_DIR = 'augmented/masks/'

# Create directories if not exist
os.makedirs(AUG_IMAGE_DIR, exist_ok=True)
os.makedirs(AUG_MASK_DIR, exist_ok=True)

# Define augmentation pipeline
augmentation = Compose([
    HorizontalFlip(p=0.5),
    VerticalFlip(p=0.5),
    RandomRotate90(p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
    OneOf([
        ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        GridDistortion(),
        OpticalDistortion(distort_limit=1.0, shift_limit=0.5)
    ], p=0.5),
    RandomBrightnessContrast(p=0.2)
])

def augment_image_and_mask(image_path, mask_path, count=5):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for i in range(count):
        augmented = augmentation(image=image, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']

        img_out_path = os.path.join(AUG_IMAGE_DIR, f"{base_name}_aug_{i}.png")
        mask_out_path = os.path.join(AUG_MASK_DIR, f"{base_name}_aug_{i}.png")

        cv2.imwrite(img_out_path, aug_img)
        cv2.imwrite(mask_out_path, aug_mask)

# Apply augmentation to all image-mask pairs
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    img_path = os.path.join(IMAGE_DIR, image_file)
    mask_path = os.path.join(MASK_DIR, image_file)
    if os.path.exists(mask_path):
        augment_image_and_mask(img_path, mask_path)
    else:
        print(f"Mask not found for image {image_file}")
