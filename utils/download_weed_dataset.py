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
import requests
import dotenv
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load environment variables from .env file
dotenv.load_dotenv()

# Get HF token from environment variable
HF_TOKEN = os.getenv('HF_TOKEN')
if not HF_TOKEN:
    print("Warning: No Hugging Face token found in .env file. Some datasets may not be accessible.")

# Import the datasets library
try:
    from datasets import load_dataset
    import huggingface_hub
except ImportError:
    print("Installing required libraries...")
    import subprocess
    subprocess.check_call(["pip", "install", "datasets", "huggingface_hub", "python-dotenv"])
    from datasets import load_dataset
    import huggingface_hub

# Set the Hugging Face token
if HF_TOKEN:
    print("Setting up Hugging Face authentication token...")
    huggingface_hub.login(token=HF_TOKEN, add_to_git_credential=False)

# Create necessary directories
def create_directories():
    """Create the required directories for storing the dataset."""
    base_dir = Path("weed_detection_dataset")
    for split in ['train', 'validation', 'test']:
        (base_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (base_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"Created directory structure at {base_dir.absolute()}")
    return base_dir

def download_aerial_crop_weed_dataset(base_dir):
    """Download the Francesco/weed-crop-aerial dataset from Hugging Face."""
    print("\n==== Downloading Aerial Crop Weed Dataset (Francesco/weed-crop-aerial) ====")
    dataset_dir = base_dir / "aerial_crop_weed"
    
    # Dataset split paths
    splits = {
        'train': 'data/train-00000-of-00001-7b13e309dc92c14f.parquet', 
        'validation': 'data/validation-00000-of-00001-f63d8aaca96c7a26.parquet', 
        'test': 'data/test-00000-of-00001-0ac8c539c96b8826.parquet'
    }
    
    # Create authentication headers if token is available
    headers = {'Authorization': f'Bearer {HF_TOKEN}'} if HF_TOKEN else None
    
    datasets = {}
    for split_name, file_path in splits.items():
        try:
            print(f"Downloading {split_name} split...")
            datasets[split_name] = pd.read_parquet(f"hf://datasets/Francesco/weed-crop-aerial/{file_path}", storage_options={'headers': headers} if headers else None)
            print(f"Successfully downloaded {split_name} split with {len(datasets[split_name])} entries")
        except Exception as e:
            print(f"Error downloading {split_name} split: {e}")
            
            # Fallback to direct download if needed
            try:
                print("Trying fallback download method...")
                from datasets import load_dataset
                dataset = load_dataset("Francesco/weed-crop-aerial", split=split_name)
                datasets[split_name] = pd.DataFrame(dataset)
                print(f"Successfully downloaded {split_name} split with {len(datasets[split_name])} entries")
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                print("Please install the Hugging Face datasets library: pip install datasets")
                return None
    
    return datasets

def extract_and_save_data(datasets, base_dir):
    """Extract images and annotations and save them in the appropriate format."""
    if not datasets:
        print("No datasets to process.")
        return
    
    # Create a metadata file for the dataset
    metadata = {
        "dataset_name": "weed-crop-aerial",
        "source": "Hugging Face - Francesco/weed-crop-aerial",
        "description": "Aerial images of crops and weeds for object detection",
        "classes": ["crop", "weed"],
        "splits": {}
    }
    
    for split_name, df in datasets.items():
        print(f"Processing {split_name} split...")
        
        # Count for statistics
        img_count = 0
        crop_count = 0
        weed_count = 0
        
        # Process each entry
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            try:
                # Extract image and save it
                img_data = row['image']['bytes']
                img = Image.open(io.BytesIO(img_data))
                
                # Create a unique filename
                img_filename = f"{split_name}_{idx:06d}.jpg"
                img_path = base_dir / split_name / 'images' / img_filename
                img.save(img_path)
                
                # Get image dimensions for normalization
                img_width, img_height = img.size
                
                # Extract annotations
                annotations = []
                if 'objects' in row and row['objects']:
                    for obj in row['objects']:
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
                                crop_count += 1
                            else:
                                weed_count += 1
                            
                            # Add to annotations
                            annotations.append(f"{class_id} {center_x} {center_y} {width} {height}")
                
                # Save annotations in YOLO format
                if annotations:
                    label_path = base_dir / split_name / 'labels' / f"{os.path.splitext(img_filename)[0]}.txt"
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(annotations))
                
                img_count += 1
                
            except Exception as e:
                print(f"Error processing entry {idx} in {split_name}: {e}")
        
        print(f"Processed {img_count} images with {crop_count} crops and {weed_count} weeds in {split_name} split")
        
        # Update metadata
        metadata["splits"][split_name] = {
            "image_count": img_count,
            "crop_count": crop_count,
            "weed_count": weed_count
        }
    
    # Save metadata
    with open(base_dir / "dataset_info.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("Dataset preparation complete!")
    print(f"Dataset stored at: {base_dir.absolute()}")
    print(f"Images are in the 'images' folder and labels in YOLO format are in the 'labels' folder for each split.")

def main():
    """Main function to download and prepare the datasets."""
    print("=== Weed Detection Datasets Downloader ===")
    print("This script will download and prepare multiple weed datasets from Hugging Face.")
    
    if HF_TOKEN:
        print("Hugging Face authentication token found. Will use for accessing restricted datasets.")
    else:
        print("No Hugging Face token found. Only public datasets will be accessible.")
    
    # Create directory structure
    base_dir = create_directories()
    
    # Download dataset
    datasets = download_aerial_crop_weed_dataset(base_dir)
    
    # Extract and save data
    extract_and_save_data(datasets, base_dir)
    
    print("\nNext steps:")
    print("1. Use the prepared dataset for training your object detection model")
    print("2. Update your model configuration to point to this dataset location")
    print("3. Make sure your model is configured for 2 classes: crop and weed")

if __name__ == "__main__":
    main()
