# coco2sam.py - Multi-class version

import json
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# Base directory containing train, test, valid folders
BASE_DIR = '/home/vuedale/Desktop/sam2.1/sam3/spondy-24'
OUTPUT_DIR = 'sam_data_multiclass'

# Create output folders
for subset in ['train', 'val', 'test']:
    os.makedirs(f'{OUTPUT_DIR}/{subset}/images', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/{subset}/masks', exist_ok=True)

def process_subset(json_file, img_dir, subset_name, csv_file):
    """Process a single subset (train/test/valid) and generate multi-class masks."""
    
    if not os.path.exists(json_file):
        print(f"Warning: {json_file} not found, skipping {subset_name}")
        return
    
    print(f"Processing {subset_name}...")
    
    # Load the COCO JSON file
    coco = COCO(json_file)
    subset_paths = []
    
    # Print category information
    print(f"\nCategories in {subset_name}:")
    categories = coco.loadCats(coco.getCatIds())
    for cat in categories:
        print(f"  ID: {cat['id']}, Name: {cat['name']}")
    print()

    # Process each image
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        image = Image.open(img_path).convert("RGB")

        # Create an empty mask (background = 0)
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

        # Get annotations for the image
        # Get annotations for the image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Draw each annotation with its category_id
        for ann in anns:
            if 'segmentation' in ann:
                category_id = ann['category_id']
                rle = coco.annToRLE(ann)
                decoded_mask = maskUtils.decode(rle)
                
                # CHANGED: Instead of combining, assign category_id to pixels
                # Where decoded_mask is 1, set the pixel to category_id
                mask[decoded_mask > 0] = category_id

        # IMPORTANT: mask now contains class indices (0 for background, 1-6 for objects)
        # No need to convert to 255!

        # Generate filenames
        img_filename = img_info['file_name']
        mask_filename = os.path.splitext(img_filename)[0] + '.png'

        # Copy image to output directory
        new_img_path = f'{OUTPUT_DIR}/{subset_name}/images/{img_filename}'
        shutil.copy(img_path, new_img_path)

        # Save mask (will be saved as grayscale with pixel values 0-6)
        new_mask_path = f'{OUTPUT_DIR}/{subset_name}/masks/{mask_filename}'
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(new_mask_path)

        # Add to CSV data
        subset_paths.append({
            'ImageId': f'./{OUTPUT_DIR}/{subset_name}/images/{img_filename}',
            'MaskId': f'./{OUTPUT_DIR}/{subset_name}/masks/{mask_filename}'
        })
        # Add to CSV data
        subset_paths.append({
            'ImageId': f'./{OUTPUT_DIR}/{subset_name}/images/{img_filename}',
            'MaskId': f'./{OUTPUT_DIR}/{subset_name}/masks/{mask_filename}'
        })

    # Save CSV file
    df = pd.DataFrame(subset_paths)
    df.to_csv(csv_file, index=False)
    print(f"Processed {len(subset_paths)} images for {subset_name}")
    print(f"Unique class values in masks: {np.unique(mask)}")  # Debug info

# Process each subset
subsets = {
    'train': ('train', 'train.csv'),
    'test': ('test', 'test.csv'),
    'valid': ('val', 'val.csv')  # Map 'valid' directory to 'val' output
}

for source_dir, (output_name, csv_name) in subsets.items():
    json_file = os.path.join(BASE_DIR, source_dir, '_annotations.coco.json')
    img_dir = os.path.join(BASE_DIR, source_dir)
    csv_file = os.path.join(OUTPUT_DIR, csv_name)
    
    process_subset(json_file, img_dir, output_name, csv_file)

print("\nConversion complete!")
print(f"Output saved to: {OUTPUT_DIR}/")
print(f"CSV files: train.csv, val.csv, test.csv")
print("\nIMPORTANT: Masks now contain class indices (0-6), not binary 0/255!")