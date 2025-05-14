"""
Direct Dataset Downloader for Weed Detection

This script directly downloads specified weed datasets from Hugging Face
using dedicated approaches for each dataset's specific format.

Datasets:
1. Francesco/weed-crop-aerial - Aerial images with crop and weed detection annotations
2. AISeedCorp/weeds-updated - Updated weed dataset
3. zkdeng/weed_insect_50k - 50k weed and insect dataset
4. AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx - Weed risk dataset without bounding boxes
"""

import os
import io
import json
import shutil
import requests
import urllib.request
from pathlib import Path
from tqdm import tqdm

# Set up Hugging Face credentials
HF_TOKEN = "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"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

def create_directory(path):
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path

def setup_directories():
    """Create base directories for all datasets."""
    base_dir = Path("weed_datasets")
    
    datasets = {
        "aerial_crop_weed": "Francesco/weed-crop-aerial",
        "seed_corp_weed": "AISeedCorp/weeds-updated",
        "weed_insect_50k": "zkdeng/weed_insect_50k",
        "weed_risk_dataset": "AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx"
    }
    
    # Create directories for each dataset
    for dataset_name in datasets:
        create_directory(base_dir / dataset_name / "images")
        create_directory(base_dir / dataset_name / "labels")
    
    print(f"Created directory structure at {base_dir.absolute()}")
    return base_dir, datasets

def download_file(url, destination_path, headers=None):
    """Download a file with progress bar."""
    response = requests.get(url, headers=headers, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination_path, 'wb') as file, tqdm(
            desc=destination_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
    
    return destination_path

def download_hf_dataset_files(repo_id, local_dir, file_patterns=None):
    """Download files from a Hugging Face dataset repository."""
    try:
        # Import huggingface_hub (install if needed)
        try:
            from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
            import huggingface_hub
        except ImportError:
            print("Installing huggingface_hub...")
            import subprocess
            subprocess.check_call(["pip", "install", "huggingface_hub"])
            from huggingface_hub import hf_hub_download, snapshot_download, list_repo_files
            import huggingface_hub
        
        # Login with token
        huggingface_hub.login(token=HF_TOKEN)
        
        print(f"Listing files for {repo_id}...")
        try:
            all_files = list_repo_files(repo_id, token=HF_TOKEN)
            print(f"Found {len(all_files)} files in repository")
            
            # Filter files if patterns are provided
            if file_patterns:
                filtered_files = []
                for pattern in file_patterns:
                    filtered_files.extend([f for f in all_files if pattern in f])
                files_to_download = filtered_files
            else:
                files_to_download = all_files
            
            print(f"Downloading {len(files_to_download)} files...")
            
            # Download each file
            for file_path in tqdm(files_to_download, desc=f"Downloading {repo_id}"):
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file_path,
                        local_dir=local_dir,
                        token=HF_TOKEN
                    )
                except Exception as e:
                    print(f"Error downloading {file_path}: {e}")
            
            return True
        
        except Exception as e:
            print(f"Error listing files, trying snapshot download: {e}")
            # Alternative: download entire snapshot
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                token=HF_TOKEN
            )
            return True
            
    except Exception as e:
        print(f"Failed to download dataset {repo_id}: {e}")
        return False

def download_francesco_weed_crop_aerial(base_dir, dataset_info):
    """Download Francesco/weed-crop-aerial dataset."""
    dataset_name = "aerial_crop_weed"
    repo_id = dataset_info[dataset_name]
    dataset_dir = base_dir / dataset_name
    
    print(f"\n=== Downloading {repo_id} ===")
    
    try:
        # Try using datasets library
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets library...")
            import subprocess
            subprocess.check_call(["pip", "install", "datasets"])
            from datasets import load_dataset
        
        # Download using the datasets library
        print("Loading dataset with HF datasets library...")
        dataset = load_dataset(repo_id, token=HF_TOKEN)
        
        # Process and save the data
        stats = {"images": 0, "annotations": 0}
        
        for split_name, split_data in dataset.items():
            print(f"Processing {split_name} split with {len(split_data)} entries")
            
            # Create directories for this split
            split_img_dir = create_directory(dataset_dir / split_name / "images")
            split_label_dir = create_directory(dataset_dir / split_name / "labels")
            
            for idx, entry in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                try:
                    # Save image
                    img_name = f"aerial_{split_name}_{idx:06d}.jpg"
                    img_path = split_img_dir / img_name
                    
                    # Save image bytes to file
                    with open(img_path, 'wb') as f:
                        f.write(entry['image']['bytes'])
                    
                    # Extract image dimensions for bounding box normalization
                    from PIL import Image
                    with Image.open(img_path) as img:
                        img_width, img_height = img.size
                    
                    # Process annotations (if any)
                    if 'objects' in entry and entry['objects']:
                        annotations = []
                        
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
                                
                                # Add to annotations
                                annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
                                stats["annotations"] += 1
                        
                        # Save annotations to a file
                        label_path = split_label_dir / f"{os.path.splitext(img_name)[0]}.txt"
                        with open(label_path, 'w') as f:
                            f.write('\n'.join(annotations))
                    
                    stats["images"] += 1
                    
                except Exception as e:
                    print(f"Error processing entry {idx} in {split_name}: {e}")
        
        # Create dataset info file
        with open(dataset_dir / "dataset_info.json", 'w') as f:
            json.dump({
                "dataset_name": "Francesco/weed-crop-aerial",
                "description": "Aerial images of crops and weeds for object detection",
                "stats": stats,
                "classes": ["crop", "weed"]
            }, f, indent=4)
        
        print(f"Successfully processed {stats['images']} images with {stats['annotations']} annotations")
        return True
        
    except Exception as e:
        print(f"Error downloading Francesco/weed-crop-aerial: {e}")
        return False

def download_aiseedcorp_weeds(base_dir, dataset_info):
    """Download AISeedCorp/weeds-updated dataset."""
    dataset_name = "seed_corp_weed"
    repo_id = dataset_info[dataset_name]
    dataset_dir = base_dir / dataset_name
    
    print(f"\n=== Downloading {repo_id} ===")
    
    try:
        # Use HF datasets library approach
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets library...")
            import subprocess
            subprocess.check_call(["pip", "install", "datasets"])
            from datasets import load_dataset
        
        # Download using the datasets library
        print("Loading dataset with HF datasets library...")
        try:
            dataset = load_dataset(repo_id, token=HF_TOKEN)
            
            # Process and save the data
            stats = {"images": 0, "annotations": 0}
            
            for split_name, split_data in dataset.items():
                print(f"Processing {split_name} split with {len(split_data)} entries")
                
                # Create directories for this split
                split_img_dir = create_directory(dataset_dir / split_name / "images")
                split_label_dir = create_directory(dataset_dir / split_name / "labels")
                
                for idx, entry in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                    try:
                        # Save image
                        img_name = f"seed_{split_name}_{idx:06d}.jpg"
                        img_path = split_img_dir / img_name
                        
                        # Check for 'image' field format and save accordingly
                        if 'image' in entry:
                            if hasattr(entry['image'], 'save'):
                                # PIL Image object
                                entry['image'].save(img_path)
                            elif isinstance(entry['image'], dict) and 'bytes' in entry['image']:
                                # Bytes in dict
                                with open(img_path, 'wb') as f:
                                    f.write(entry['image']['bytes'])
                            elif isinstance(entry['image'], bytes):
                                # Raw bytes
                                with open(img_path, 'wb') as f:
                                    f.write(entry['image'])
                            else:
                                print(f"Unknown image format in entry {idx}, skipping")
                                continue
                        else:
                            print(f"No image field in entry {idx}, skipping")
                            continue
                        
                        # For now, create an empty label file since we don't know the format
                        # You can update this when you examine the actual dataset structure
                        label_path = split_label_dir / f"{os.path.splitext(img_name)[0]}.txt"
                        with open(label_path, 'w') as f:
                            f.write("# Placeholder for annotations\n")
                        
                        stats["images"] += 1
                        
                    except Exception as e:
                        print(f"Error processing entry {idx} in {split_name}: {e}")
            
            # Create dataset info file
            with open(dataset_dir / "dataset_info.json", 'w') as f:
                json.dump({
                    "dataset_name": repo_id,
                    "description": "Updated weed dataset from AISeedCorp",
                    "stats": stats
                }, f, indent=4)
            
            print(f"Successfully processed {stats['images']} images")
            return True
            
        except Exception as e:
            print(f"Error with datasets library: {e}")
            print("Trying direct file download...")
            
            # Fallback to direct file download
            return download_hf_dataset_files(repo_id, dataset_dir)
            
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")
        return False

def download_zkdeng_weed_insect(base_dir, dataset_info):
    """Download zkdeng/weed_insect_50k dataset."""
    dataset_name = "weed_insect_50k"
    repo_id = dataset_info[dataset_name]
    dataset_dir = base_dir / dataset_name
    
    print(f"\n=== Downloading {repo_id} ===")
    
    try:
        # Use HF datasets library approach
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets library...")
            import subprocess
            subprocess.check_call(["pip", "install", "datasets"])
            from datasets import load_dataset
        
        # Download using the datasets library
        print("Loading dataset with HF datasets library...")
        try:
            dataset = load_dataset(repo_id, token=HF_TOKEN)
            
            # Process and save the data
            stats = {"images": 0, "classes": {}}
            
            for split_name, split_data in dataset.items():
                print(f"Processing {split_name} split with {len(split_data)} entries")
                
                # Create directories for this split
                split_img_dir = create_directory(dataset_dir / split_name / "images")
                split_label_dir = create_directory(dataset_dir / split_name / "labels")
                
                for idx, entry in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                    try:
                        # Save image
                        img_name = f"insect_{split_name}_{idx:06d}.jpg"
                        img_path = split_img_dir / img_name
                        
                        # Check for 'image' field format and save accordingly
                        if 'image' in entry:
                            if hasattr(entry['image'], 'save'):
                                # PIL Image object
                                entry['image'].save(img_path)
                            elif isinstance(entry['image'], dict) and 'bytes' in entry['image']:
                                # Bytes in dict
                                with open(img_path, 'wb') as f:
                                    f.write(entry['image']['bytes'])
                            elif isinstance(entry['image'], bytes):
                                # Raw bytes
                                with open(img_path, 'wb') as f:
                                    f.write(entry['image'])
                            else:
                                print(f"Unknown image format in entry {idx}, skipping")
                                continue
                        else:
                            print(f"No image field in entry {idx}, skipping")
                            continue
                        
                        # Track class information if available
                        if 'label' in entry:
                            label = str(entry['label'])
                            if label not in stats["classes"]:
                                stats["classes"][label] = 0
                            stats["classes"][label] += 1
                            
                            # For object detection format, create a placeholder label file
                            label_path = split_label_dir / f"{os.path.splitext(img_name)[0]}.txt"
                            with open(label_path, 'w') as f:
                                f.write(f"# Class: {label}\n")
                        
                        stats["images"] += 1
                        
                    except Exception as e:
                        print(f"Error processing entry {idx} in {split_name}: {e}")
            
            # Create dataset info file
            with open(dataset_dir / "dataset_info.json", 'w') as f:
                json.dump({
                    "dataset_name": repo_id,
                    "description": "50k dataset of weeds and insects",
                    "stats": stats,
                    "classes": list(stats["classes"].keys())
                }, f, indent=4)
            
            print(f"Successfully processed {stats['images']} images")
            return True
            
        except Exception as e:
            print(f"Error with datasets library: {e}")
            print("Trying direct file download...")
            
            # Fallback to direct file download
            return download_hf_dataset_files(repo_id, dataset_dir)
            
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")
        return False

def download_weed_risk_dataset(base_dir, dataset_info):
    """Download AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx dataset."""
    dataset_name = "weed_risk_dataset"
    repo_id = dataset_info[dataset_name]
    dataset_dir = base_dir / dataset_name
    
    print(f"\n=== Downloading {repo_id} ===")
    
    try:
        # Use HF datasets library approach
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets library...")
            import subprocess
            subprocess.check_call(["pip", "install", "datasets"])
            from datasets import load_dataset
        
        # Download using the datasets library
        print("Loading dataset with HF datasets library...")
        try:
            dataset = load_dataset(repo_id, token=HF_TOKEN)
            
            # Process and save the data
            stats = {"images": 0}
            
            for split_name, split_data in dataset.items():
                print(f"Processing {split_name} split with {len(split_data)} entries")
                
                # Create directories for this split
                split_img_dir = create_directory(dataset_dir / split_name / "images")
                split_label_dir = create_directory(dataset_dir / split_name / "labels")
                
                for idx, entry in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
                    try:
                        # Save image
                        img_name = f"weedrisk_{split_name}_{idx:06d}.jpg"
                        img_path = split_img_dir / img_name
                        
                        # Check for 'image' field format and save accordingly
                        if 'image' in entry:
                            if hasattr(entry['image'], 'save'):
                                # PIL Image object
                                entry['image'].save(img_path)
                            elif isinstance(entry['image'], dict) and 'bytes' in entry['image']:
                                # Bytes in dict
                                with open(img_path, 'wb') as f:
                                    f.write(entry['image']['bytes'])
                            elif isinstance(entry['image'], bytes):
                                # Raw bytes
                                with open(img_path, 'wb') as f:
                                    f.write(entry['image'])
                            else:
                                print(f"Unknown image format in entry {idx}, skipping")
                                continue
                        else:
                            print(f"No image field in entry {idx}, skipping")
                            continue
                        
                        # This dataset has no bounding boxes as indicated by the name
                        # Create empty label file for consistency
                        label_path = split_label_dir / f"{os.path.splitext(img_name)[0]}.txt"
                        with open(label_path, 'w') as f:
                            pass  # Empty file
                        
                        stats["images"] += 1
                        
                    except Exception as e:
                        print(f"Error processing entry {idx} in {split_name}: {e}")
            
            # Create dataset info file
            with open(dataset_dir / "dataset_info.json", 'w') as f:
                json.dump({
                    "dataset_name": repo_id,
                    "description": "Weed risk dataset without bounding boxes",
                    "stats": stats
                }, f, indent=4)
            
            print(f"Successfully processed {stats['images']} images")
            return True
            
        except Exception as e:
            print(f"Error with datasets library: {e}")
            print("Trying direct file download...")
            
            # Fallback to direct file download
            return download_hf_dataset_files(repo_id, dataset_dir)
            
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")
        return False

def main():
    """Main function to download datasets."""
    print("=== Weed Detection Direct Dataset Downloader ===")
    print("This script will download multiple weed datasets from Hugging Face")
    print("using methods tailored to each dataset's structure.\n")
    
    # Setup directory structure
    base_dir, dataset_info = setup_directories()
    
    # Track download results
    results = {}
    
    # Download datasets
    # 1. Francesco/weed-crop-aerial
    results["aerial_crop_weed"] = download_francesco_weed_crop_aerial(base_dir, dataset_info)
    
    # 2. AISeedCorp/weeds-updated
    results["seed_corp_weed"] = download_aiseedcorp_weeds(base_dir, dataset_info)
    
    # 3. zkdeng/weed_insect_50k
    results["weed_insect_50k"] = download_zkdeng_weed_insect(base_dir, dataset_info)
    
    # 4. AdamStormed22/weed-risk-reg-vqa-dataset-no-bbx
    results["weed_risk_dataset"] = download_weed_risk_dataset(base_dir, dataset_info)
    
    # Create summary file
    with open(base_dir / "download_summary.json", 'w') as f:
        json.dump({
            "download_time": os.path.getmtime(base_dir) if os.path.exists(base_dir) else None,
            "results": results
        }, f, indent=4)
    
    # Print summary
    print("\n=== Download Summary ===")
    for dataset, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{dataset}: {status}")
    
    print("\nNext steps:")
    print("1. Review the downloaded datasets in the 'weed_datasets' directory")
    print("2. Train an object detection model using the datasets with bounding box annotations")
    print("3. Make sure your model is configured for the correct number of classes")

if __name__ == "__main__":
    main()
