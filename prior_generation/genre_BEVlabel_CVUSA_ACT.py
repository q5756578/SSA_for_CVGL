import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
import glob
'''
This file is used to synthesize BEV labels

It uses sam_sat_building, sat_structre, sat_trees from each city dataset
sam_sat_building and sat_structre are combined using AND operation to get result D
The value 255 in sat_trees at corresponding positions in D is also changed to 255
'''
def ensure_empty_folder(path):
    """
    Check if the specified folder exists and is empty.
    If it doesn't exist or is not empty, create a new empty folder.
    
    Args:
        path (str): Folder path.
    """
    if os.path.exists(path):
        if os.path.isdir(path) and not os.listdir(path):
            print(f"Folder '{path}' exists and is empty.")
        else:
            print(f"Folder '{path}' is not empty, will be cleared and recreated.")
            shutil.rmtree(path)
            os.makedirs(path)
            print(f"Recreated empty folder: {path}")
    else:
        os.makedirs(path)
        print(f"Folder '{path}' doesn't exist, created new folder.")

def find_image_files_recursive(root_dir, extensions=('*.jpg', '*.jpeg', '*.png')):
    """
    Recursively find all image files in the specified directory
    
    Args:
        root_dir: Root directory path
        extensions: Image file extensions to search
        
    Returns:
        list: List containing paths of all image files
    """
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    return image_files

def process_binary_images(folder_A, folder_B, folder_C, output_folder):
    """
    Process binary images, perform AND operation and save results, maintaining original directory structure.

    Args:
        folder_A (str): Path to folder A containing binary images A.
        folder_B (str): Path to folder B containing binary images B.
        folder_C (str): Path to folder C containing binary images C.
        output_folder (str): Folder to save result images D.
    """
    # Ensure output folder exists
    ensure_empty_folder(output_folder)

    # Supported image formats
    image_extensions = ('*.png', '*.jpg', '*.jpeg')

    # Get list of subfolders
    subfolders = [f.path for f in os.scandir(folder_A) if f.is_dir()]
    
    # Process each subfolder
    for subfolder_A in tqdm(subfolders, desc="Processing subfolders"):
        subfolder_name = os.path.basename(subfolder_A)
        
        # Build corresponding B, C subfolder paths
        subfolder_B = os.path.join(folder_B, subfolder_name)
        subfolder_C = os.path.join(folder_C, subfolder_name)
        
        # Create output subfolder
        output_subfolder = os.path.join(output_folder, subfolder_name)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Get all images in current subfolder
        files_A = []
        for ext in image_extensions:
            files_A.extend(glob.glob(os.path.join(subfolder_A, ext)))
        files_A = sorted(files_A)  # Ensure file order is consistent
        
        # Process each image file
        for file_A in tqdm(files_A, desc=f"Processing {subfolder_name}"):
            filename = os.path.basename(file_A)
            file_B = os.path.join(subfolder_B, filename)
            file_C = os.path.join(subfolder_C, filename)
            
            # Check if all files exist
            if not (os.path.exists(file_B) and os.path.exists(file_C)):
                print(f"Warning: Missing corresponding files for {filename}")
                continue
            
            # Read images
            img_A = cv2.imread(file_A, cv2.IMREAD_GRAYSCALE)
            img_B = cv2.imread(file_B, cv2.IMREAD_GRAYSCALE)
            img_C = cv2.imread(file_C, cv2.IMREAD_GRAYSCALE)
            
            if img_A is None or img_B is None or img_C is None:
                print(f"Warning: Failed to read one or more images for {filename}")
                continue
            
            # Check image dimensions
            if img_A.shape != img_B.shape or img_A.shape != img_C.shape:
                print(f"Warning: Image dimensions do not match for {filename}")
                continue
            
            # AND operation between A and B
            result = cv2.bitwise_and(img_A, img_B) * 255
            
            # Set positions where C is 255 to 255
            result = cv2.bitwise_or(img_C, result) * 255
            
            # Save result
            output_path = os.path.join(output_subfolder, filename)
            cv2.imwrite(output_path, result)

# Specify input and output folders
inputA_folder = ''
inputB_folder = ''
inputC_folder = ''
output_folder = ''

# Execute processing
process_binary_images(inputA_folder, inputB_folder, inputC_folder, output_folder)
