"""
Download and prepare multiple weed datasets from Hugging Face.
This script downloads selected weed detection/classification datasets and organizes them 
for training object detection models.

Datasets included:
1. Francesco/weed-crop-aerial - Aerial images with crop and weed detection annotations
2. AISeedCorp/weeds-updated - Updated weed dataset
3. zkdeng/weed_insect_50k - 50k weed and insect dataset
4. AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx - Weed risk dataset without bounding boxes
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
from tqdm import tqdm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Install required packages if not available
try:
    from datasets import load_dataset
    import huggingface_hub
    from dotenv import load_dotenv
except ImportError:
    print("Installing required libraries...")
    import subprocess
    subprocess.check_call(["pip", "install", "datasets", "huggingface_hub", "python-dotenv"])
    from datasets import load_dataset
    import huggingface_hub
    from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get HF token from environment variable (or use the one provided directly)
HF_TOKEN = os.getenv('HF_TOKEN', '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')

# Set the Hugging Face token for authentication
if HF_TOKEN:
    print("Setting up Hugging Face authentication token...")
    huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)
else:
    print("No Hugging Face token found. Only public datasets will be accessible.")

# Create necessary directories
def create_directories():
    """Create the required directories for storing the dataset."""
    base_dir = Path("weed_detection_dataset")
    
    # Create main dataset directories
    datasets = [
        "aerial_crop_weed",       # Francesco/weed-crop-aerial
        "seed_corp_weed",         # AISeedCorp/weeds-updated
        "weed_insect_50k",        # zkdeng/weed_insect_50k
        "weed_risk_dataset"       # AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx
    ]
    
    for dataset in datasets:
        for split in ['train', 'validation', 'test']:
            (base_dir / dataset / split / 'images').mkdir(parents=True, exist_ok=True)
            (base_dir / dataset / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at {base_dir.absolute()}")
    return base_dir

def save_image(img, dataset_dir, split_name, prefix, idx):
    """Helper function to save an image with a consistent naming convention."""
    img_filename = f"{prefix}_{split_name}_{idx:06d}.jpg"
    img_path = dataset_dir / split_name / 'images' / img_filename
    img.save(img_path)
    return img_filename

def save_empty_label(dataset_dir, split_name, img_filename):
    """Create an empty label file (for datasets without annotations)."""
    label_path = dataset_dir / split_name / 'labels' / f"{os.path.splitext(img_filename)[0]}.txt"
    with open(label_path, 'w') as f:
        pass  # Empty file
    return label_path

def save_yolo_annotations(dataset_dir, split_name, img_filename, annotations):
    """Save annotations in YOLO format."""
    if annotations:
        label_path = dataset_dir / split_name / 'labels' / f"{os.path.splitext(img_filename)[0]}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(annotations))
        return label_path
    return None

def download_aerial_crop_weed_dataset(base_dir):
    """Download the Francesco/weed-crop-aerial dataset from Hugging Face."""
    print("\n==== Downloading Aerial Crop Weed Dataset (Francesco/weed-crop-aerial) ====")
    dataset_dir = base_dir / "aerial_crop_weed"
    
    stats = {"images": 0, "crop_objects": 0, "weed_objects": 0}
    
    try:
        # Use datasets library with authentication
        for split_name in ['train', 'validation', 'test']:
            print(f"Downloading {split_name} split...")
            try:
                ds = load_dataset("Francesco/weed-crop-aerial", split=split_name, token=HF_TOKEN)
                print(f"  Successfully downloaded {split_name} split with {len(ds)} entries")
                
                # Process entries
                print(f"Processing {split_name} split...")
                for idx, entry in enumerate(tqdm(ds, desc=f"Processing {split_name}")):
                    try:
                        if 'image' in entry:
                            # Convert image bytes to PIL Image
                            img = Image.open(io.BytesIO(entry['image']['bytes']))
                            
                            # Save image
                            img_filename = save_image(img, dataset_dir, split_name, "aerial", idx)
                            
                            # Get image dimensions for normalization
                            img_width, img_height = img.size
                            
                            # Extract annotations
                            annotations = []
                            if 'objects' in entry and entry['objects']:
                                for obj in entry['objects']:
                                    if 'bbox' in obj and 'category' in obj:
                                        # Extract bounding box (x1, y1, x2, y2 format)
                                        x1, y1, x2, y2 = obj['bbox']
                                        
                                        # Convert to YOLO format (centerx, centery, width, height) normalized
                                        center_x = (x1 + x2) / 2 / img_width
                                        center_y = (y1 + y2) / 2 / img_height
                                        width = (x2 - x1) / img_width
                                        height = (y2 - y1) / img_height
                                        
                                        # Class ID (0 for crop, 1 for weed)
                                        class_id = 0 if obj['category'] == 'crop' else 1
                                        
                                        # Count objects by class
                                        if class_id == 0:
                                            stats["crop_objects"] += 1
                                        else:
                                            stats["weed_objects"] += 1
                                        
                                        # Add to annotations
                                        annotations.append(f"{class_id} {center_x} {center_y} {width} {height}")
                            
                            # Save annotations in YOLO format
                            save_yolo_annotations(dataset_dir, split_name, img_filename, annotations)
                            
                            stats["images"] += 1
                    except Exception as e:
                        print(f"Error processing entry {idx} in {split_name}: {e}")
                
                print(f"Processed {stats['images']} images in {split_name} split")
            
            except Exception as e:
                print(f"Error with {split_name} split: {e}")
        
        print(f"Dataset Summary: {stats['images']} images with {stats['crop_objects']} crops and {stats['weed_objects']} weeds")
        return True, stats
    
    except Exception as e:
        print(f"Failed to download aerial crop weed dataset: {e}")
        return False, stats

def download_seed_corp_weed_dataset(base_dir):
    """Download the AISeedCorp/weeds-updated dataset from Hugging Face."""
    print("\n==== Downloading Seed Corp Weed Dataset (AISeedCorp/weeds-updated) ====")
    dataset_dir = base_dir / "seed_corp_weed"
    
    stats = {"images": 0, "annotations": 0}
    
    try:
        # Use datasets library with authentication
        for split_name in ['train', 'validation']:
            print(f"Downloading {split_name} split...")
            try:
                ds = load_dataset("AISeedCorp/weeds-updated", split=split_name, token=HF_TOKEN)
                print(f"  Successfully downloaded {split_name} split with {len(ds)} entries")
                
                # Process entries
                print(f"Processing {split_name} split...")
                for idx, entry in enumerate(tqdm(ds, desc=f"Processing {split_name}")):
                    try:
                        if 'image' in entry:
                            # Convert image bytes to PIL Image if needed
                            if isinstance(entry['image'], dict) and 'bytes' in entry['image']:
                                img = Image.open(io.BytesIO(entry['image']['bytes']))
                            else:
                                # Handle based on actual format
                                img = entry['image']
                            
                            # Save image
                            img_filename = save_image(img, dataset_dir, split_name, "seedcorp", idx)
                            
                            # Process annotations if available
                            annotations = []
                            if 'annotations' in entry and entry['annotations']:
                                # This is a placeholder - adapt to actual dataset format
                                # Process each annotation into YOLO format
                                pass
                            
                            # Save annotations or empty label file
                            if annotations:
                                save_yolo_annotations(dataset_dir, split_name, img_filename, annotations)
                                stats["annotations"] += len(annotations)
                            else:
                                save_empty_label(dataset_dir, split_name, img_filename)
                            
                            stats["images"] += 1
                    except Exception as e:
                        print(f"Error processing entry {idx} in {split_name}: {e}")
                
                print(f"Processed {stats['images']} images in {split_name} split")
            
            except Exception as e:
                print(f"Error with {split_name} split: {e}")
        
        print(f"Dataset Summary: {stats['images']} images with {stats['annotations']} annotations")
        return True, stats
    
    except Exception as e:
        print(f"Failed to download seed corp weed dataset: {e}")
        return False, stats

def download_weed_insect_dataset(base_dir):
    """Download the zkdeng/weed_insect_50k dataset from Hugging Face."""
    print("\n==== Downloading Weed Insect Dataset (zkdeng/weed_insect_50k) ====")
    dataset_dir = base_dir / "weed_insect_50k"
    
    stats = {"images": 0, "labels": {}}
    
    try:
        print(f"Downloading zkdeng/weed_insect_50k dataset...")
        ds = load_dataset("zkdeng/weed_insect_50k", token=HF_TOKEN)
        
        # Process each split
        for split_name in ds.keys():
            print(f"Processing {split_name} split with {len(ds[split_name])} entries")
            
            # Process entries
            for idx, entry in enumerate(tqdm(ds[split_name], desc=f"Processing {split_name}")):
                try:
                    if 'image' in entry:
                        # Get the image
                        img = entry['image']
                        
                        # Save image
                        img_filename = save_image(img, dataset_dir, split_name, "insect", idx)
                        
                        # Get label information if available
                        label = None
                        if 'label' in entry:
                            label = entry['label']
                            # Keep track of label counts
                            if label not in stats["labels"]:
                                stats["labels"][label] = 0
                            stats["labels"][label] += 1
                        
                        # Create an empty label file for object detection format consistency
                        save_empty_label(dataset_dir, split_name, img_filename)
                        
                        stats["images"] += 1
                except Exception as e:
                    print(f"Error processing entry {idx} in {split_name}: {e}")
            
            print(f"Processed {stats['images']} images in {split_name} split")
        
        print(f"Dataset Summary: {stats['images']} images with label distribution: {stats['labels']}")
        return True, stats
    
    except Exception as e:
        print(f"Failed to download weed insect dataset: {e}")
        return False, stats

def download_weed_risk_dataset(base_dir):
    """Download the AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx dataset from Hugging Face."""
    print("\n==== Downloading Weed Risk Dataset (AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx) ====")
    dataset_dir = base_dir / "weed_risk_dataset"
    
    stats = {"images": 0}
    
    try:
        # Use datasets library with authentication
        for split_name in ['train', 'validation', 'test']:
            print(f"Downloading {split_name} split...")
            try:
                ds = load_dataset("AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx", split=split_name, token=HF_TOKEN)
                print(f"  Successfully downloaded {split_name} split with {len(ds)} entries")
                
                # Process entries
                print(f"Processing {split_name} split...")
                for idx, entry in enumerate(tqdm(ds, desc=f"Processing {split_name}")):
                    try:
                        if 'image' in entry:
                            # Convert image bytes to PIL Image if needed
                            if isinstance(entry['image'], dict) and 'bytes' in entry['image']:
                                img = Image.open(io.BytesIO(entry['image']['bytes']))
                            else:
                                # Handle based on actual format
                                img = entry['image']
                            
                            # Save image
                            img_filename = save_image(img, dataset_dir, split_name, "weedrisk", idx)
                            
                            # This dataset has no bounding boxes as indicated by the name
                            # Create empty label file for consistency
                            save_empty_label(dataset_dir, split_name, img_filename)
                            
                            stats["images"] += 1
                    except Exception as e:
                        print(f"Error processing entry {idx} in {split_name}: {e}")
                
                print(f"Processed {stats['images']} images in {split_name} split")
            
            except Exception as e:
                print(f"Error with {split_name} split: {e}")
        
        print(f"Dataset Summary: {stats['images']} images (no annotations)")
        return True, stats
    
    except Exception as e:
        print(f"Failed to download weed risk dataset: {e}")
        return False, stats

def create_dataset_metadata(base_dir, results):
    """Create metadata file with information about all datasets."""
    metadata = {
        "datasets": {
            "aerial_crop_weed": {
                "source": "Francesco/weed-crop-aerial",
                "description": "Aerial images of crops and weeds for object detection",
                "classes": ["crop", "weed"],
                "has_annotations": True,
                "annotation_type": "bounding_box",
                "stats": results["aerial_crop_weed"][1] if "aerial_crop_weed" in results else {}
            },
            "seed_corp_weed": {
                "source": "AISeedCorp/weeds-updated",
                "description": "Updated weed dataset from AISeedCorp",
                "has_annotations": True,  # Update based on actual dataset
                "stats": results["seed_corp_weed"][1] if "seed_corp_weed" in results else {}
            },
            "weed_insect_50k": {
                "source": "zkdeng/weed_insect_50k",
                "description": "50k dataset of weeds and insects",
                "has_annotations": False,  # This dataset has image-level labels, not bounding boxes
                "stats": results["weed_insect_50k"][1] if "weed_insect_50k" in results else {}
            },
            "weed_risk_dataset": {
                "source": "AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx",
                "description": "Weed risk dataset without bounding boxes",
                "has_annotations": False,
                "stats": results["weed_risk_dataset"][1] if "weed_risk_dataset" in results else {}
            }
        },
        "download_results": {k: v[0] for k, v in results.items()}
    }
    
    # Save metadata
    with open(base_dir / "dataset_info.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Created dataset metadata file at {base_dir / 'dataset_info.json'}")

def main():
    """Main function to download and prepare the datasets."""
    print("=== Weed Detection Datasets Downloader ===")
    print("This script will download and prepare multiple weed datasets from Hugging Face.")
    
    # Create directory structure
    base_dir = create_directories()
    
    # Download and process each dataset
    results = {}
    
    # Dataset 1: Francesco/weed-crop-aerial
    results["aerial_crop_weed"] = download_aerial_crop_weed_dataset(base_dir)
    
    # Dataset 2: AISeedCorp/weeds-updated
    results["seed_corp_weed"] = download_seed_corp_weed_dataset(base_dir)
    
    # Dataset 3: zkdeng/weed_insect_50k
    results["weed_insect_50k"] = download_weed_insect_dataset(base_dir)
    
    # Dataset 4: AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx
    results["weed_risk_dataset"] = download_weed_risk_dataset(base_dir)
    
    # Create metadata
    create_dataset_metadata(base_dir, results)
    
    print("\n=== Download Summary ===")
    for dataset, (success, _) in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{dataset}: {status}")
    
    print("\nNext steps:")
    print("1. Review the downloaded datasets in the 'weed_detection_dataset' directory")
    print("2. Train an object detection model using the aerial_crop_weed dataset")
    print("3. Consider using the other datasets for additional training or fine-tuning")
    print("4. Make sure your model is configured for 2 classes: crop and weed")

if __name__ == "__main__":
    main()
