import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
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
    # Check if folder exists
    if os.path.exists(path):
        # If exists, check if empty
        if os.path.isdir(path) and not os.listdir(path):
            print(f"Folder '{path}' exists and is empty.")
        else:
            print(f"Folder '{path}' is not empty, will be cleared and recreated.")
            shutil.rmtree(path)  # Delete existing folder and its contents
            os.makedirs(path)    # Recreate empty folder
            print(f"Recreated empty folder: {path}")
    else:
        # If doesn't exist, create new folder
        os.makedirs(path)
        print(f"Folder '{path}' doesn't exist, created new folder.")


def process_binary_images(folder_A, folder_B, folder_C, output_folder):
    """
    Process binary images, perform AND operation and save results.

    Args:
        folder_A (str): Path to folder A containing binary images A.
        folder_B (str): Path to folder B containing binary images B.
        folder_C (str): Path to folder C containing binary images C.
        output_folder (str): Folder to save result images D.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get all file names from three folders
    files_A = set(os.listdir(folder_A))
    files_B = set(os.listdir(folder_B))
    files_C = set(os.listdir(folder_C))

    print('The number of files is:', len(files_A), len(files_B), len(files_C))
        
    # Get common file names from three folders
    common_files = files_A & files_B & files_C
    
    # Use tqdm to show progress bar
    for filename in tqdm(common_files, desc="Processing images"):
        # Construct file paths
        path_A = os.path.join(folder_A, filename)
        path_B = os.path.join(folder_B, filename)
        path_C = os.path.join(folder_C, filename)

        # Read binary images
        img_A = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        img_B = cv2.imread(path_B, cv2.IMREAD_GRAYSCALE)
        img_C = cv2.imread(path_C, cv2.IMREAD_GRAYSCALE)

        if img_A is None or img_B is None or img_C is None:
            print(f"Warning: Failed to read one or more images for {filename}.")
            continue

        # Image size check
        if img_A.shape != img_B.shape or img_A.shape != img_C.shape:
            print(f"Warning: Image dimensions do not match for {filename}.")
            continue

        # AND operation between A and B
        result = cv2.bitwise_and(img_A, img_B) * 255

        # Set positions where C is 255 to 255
        result = cv2.bitwise_or(img_C,result) * 255 

        # Save result image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, result)

        print(f"Processed and saved: {filename}")

# Specify input and output folders
input_folder = '/home/jshen/Depth-Anything-V2/test_images_depth'  # Replace with your input folder path
#output_folder = '/home/jshen/data/CVACT/BEV_label'  # Replace with your output folder path

CITYS = ['NewYork', 'SanFrancisco']

# Run program
    
inputA_folder = '/test_samples_artificial_buildings/'
inputB_folder = '/CVACT_new/test_samples_structures/'
inputC_folder = '/CVACT_new/test_samples_trees/'
output_folder =  '/CVACT_new/sat_BEV_label/'
ensure_empty_folder(output_folder)
process_binary_images(inputA_folder, inputB_folder,inputC_folder,output_folder)
