# utils/data_augmentation.py
import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

def create_augmentation_pipeline():
    """Create data augmentation pipeline for plants/weeds."""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.RandomShadow(p=0.3),
        A.Resize(224, 224, always_apply=True)  # Match model input size
    ])

def augment_dataset(src_dir, dst_dir, augmentations_per_image=5):
    """Augment images in source directory and save to destination.
    
    Args:
        src_dir: Source directory with original images
        dst_dir: Destination directory for augmented images
        augmentations_per_image: Number of augmentations per original image
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        
    transform = create_augmentation_pipeline()
    
    # Get all image files
    img_files = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_files.append(os.path.join(root, file))
    
    print(f"Found {len(img_files)} images to augment")
    
    for img_path in tqdm(img_files):
        # Determine class from directory structure
        relative_path = os.path.relpath(img_path, src_dir)
        img_class = os.path.dirname(relative_path)
        
        # Create output directory for this class
        class_dst_dir = os.path.join(dst_dir, img_class)
        if not os.path.exists(class_dst_dir):
            os.makedirs(class_dst_dir)
        
        # Load and augment image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Save original image
        filename = os.path.basename(img_path)
        original_dst = os.path.join(class_dst_dir, filename)
        cv2.imwrite(original_dst, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Create and save augmentations
        for i in range(augmentations_per_image):
            augmented = transform(image=img)['image']
            base_name, ext = os.path.splitext(filename)
            aug_filename = f"{base_name}_aug{i}{ext}"
            aug_dst = os.path.join(class_dst_dir, aug_filename)
            cv2.imwrite(aug_dst, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
    
    print(f"Augmentation complete. Generated {len(img_files) * augmentations_per_image} new images")

if __name__ == "__main__":
    # Example usage
    augment_dataset("dataset/original", "dataset/augmented", 5)