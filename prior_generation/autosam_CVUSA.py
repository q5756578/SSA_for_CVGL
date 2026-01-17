import os 
from PIL import Image 
import torch
import glob
import math
import shutil
import time
import multiprocessing as mp
from multiprocessing import Process, Queue

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from samgeo import tms_to_geotiff 
from samgeo.text_sam import LangSAM
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  
import pandas as pd 

'''
This file is used to generate segmentation masks for satellite images using the SAMGeo model.

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

def find_image_files_by_subfolder(root_dir, extensions=('*.jpg', '*.jpeg', '*.png')):
    """
    Organize image files by subfolder
    
    Args:
        root_dir: Root directory path
        extensions: Image file extensions to search
        
    Returns:
        dict: Mapping from subfolder name to list of image files
    """
    subfolder_images = {}
    
    # Get all first-level subfolders
    subfolders = [f.path for f in os.scandir(root_dir) if f.is_dir()]
    
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(subfolder, ext)))
        subfolder_images[subfolder_name] = sorted(image_files)  # Ensure file order is consistent
        
    return subfolder_images

def create_directory_structure(source_dir, target_dir):
    """
    Create the same directory structure in the target directory as in the source directory
    
    Args:
        source_dir: Source directory path
        target_dir: Target directory path
    """
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Traverse all subdirectories in the source directory
    for root, dirs, files in os.walk(source_dir):
        # Calculate relative path
        rel_path = os.path.relpath(root, source_dir)
        if rel_path == '.':
            continue
            
        # Create corresponding directory in the target directory
        target_subdir = os.path.join(target_dir, rel_path)
        os.makedirs(target_subdir, exist_ok=True)

def process_batch(gpu_id, image_files, prompt, target_path, queue):
    """
    Process a batch of images using the specified GPU
    
    Args:
        gpu_id: GPU ID
        image_files: List of image files to process
        prompt: Text prompt
        target_path: Output directory
        queue: Queue for inter-process communication
    """
    try:
        # Set the GPU to be used by the current process
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = f"cuda:{gpu_id}"
        
        # Create SAM instance
        sam = LangSAM(checkpoint='./sam_vit_h_4b8939.pth', model_type='vit_h', device=device)
        
        print(f"\nGPU {gpu_id} processing {len(image_files)} images for prompt: {prompt}")
        
        # Create temporary directory
        temp_dir = os.path.join(target_path, f"temp_gpu_{gpu_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Process images
        sam.predict_batch_v2(
            images=image_files,
            out_dir=temp_dir,
            text_prompt=prompt,
            box_threshold=0.24,
            text_threshold=0.6,
            mask_multiplier=255,
            dtype="uint8",
            merge=False,
            verbose=True,
        )
        
        # Move results to main directory, maintaining original directory structure
        for filename in os.listdir(temp_dir):
            src = os.path.join(temp_dir, filename)
            # Get the relative path of the original file
            original_path = next((f for f in image_files if os.path.basename(f) == filename), None)
            if original_path:
                # Get the path relative to the root directory
                rel_path = os.path.relpath(original_path, os.path.dirname(image_files[0]))
                # Create target directory
                dst_dir = os.path.join(target_path, os.path.dirname(rel_path))
                os.makedirs(dst_dir, exist_ok=True)
                # Move file
                dst = os.path.join(dst_dir, filename)
                shutil.move(src, dst)
        
        print(f"GPU {gpu_id} completed processing")
        queue.put((True, f"GPU {gpu_id} completed successfully"))
        
    except Exception as e:
        print(f"Error on GPU {gpu_id}: {str(e)}")
        queue.put((False, f"GPU {gpu_id} failed: {str(e)}"))
    finally:
        # Clean up temporary directory
        temp_dir = os.path.join(target_path, f"temp_gpu_{gpu_id}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def process_prompt_by_subfolder(prompt, subfolder_images, target_base_path, num_gpus=4):
    """
    Process all images for a single prompt by subfolder
    
    Args:
        prompt: Text prompt
        subfolder_images: Mapping from subfolder to list of images
        target_base_path: Base output directory
        num_gpus: Number of GPUs
    """
    print(f"\nProcessing prompt: {prompt}")
    
    # Create main output directory for the current prompt
    target_path = os.path.join(target_base_path, f"samples_{prompt}")
    ensure_empty_folder(target_path)
    
    # Process each subfolder
    for subfolder_name, image_files in subfolder_images.items():
        print(f"\nProcessing subfolder: {subfolder_name}")
        
        # Create subfolder output path
        subfolder_target_path = os.path.join(target_path, subfolder_name)
        os.makedirs(subfolder_target_path, exist_ok=True)
        
        # Split image list into multiple batches
        batch_size = math.ceil(len(image_files) / num_gpus)
        image_batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
        
        # Create queue for inter-process communication
        queue = Queue()
        
        # Create and start processes
        processes = []
        for i in range(num_gpus):
            if i < len(image_batches):  # Ensure enough data for each GPU
                p = Process(
                    target=process_batch,
                    args=(i, image_batches[i], prompt, subfolder_target_path, queue)
                )
                processes.append(p)
                p.start()
        
        # Wait for all processes to complete and collect results
        success_count = 0
        error_count = 0
        for _ in range(len(processes)):
            success, message = queue.get()
            if success:
                success_count += 1
            else:
                error_count += 1
            print(message)
        
        # Wait for all processes to end
        for p in processes:
            p.join()
        
        print(f"Completed processing subfolder {subfolder_name}")
        print(f"Successfully processed: {success_count} batches")
        if error_count > 0:
            print(f"Failed to process: {error_count} batches")

def main():
    # Set parameters
    image_path = '/home/jshen/data/CVUSA/CVUSA_subset/bingmap'
    text_prompt_list = ['artificial buildings', 'structures', 'trees', 'construction', 'edifice']
    num_gpus = 4
    
    # Get image files by subfolder
    subfolder_images = find_image_files_by_subfolder(image_path)
    total_images = sum(len(files) for files in subfolder_images.values())
    print(f"Total images to process: {total_images}")
    print("Subfolders found:", list(subfolder_images.keys()))
    
    # Set base output directory
    output_base_path = "/home/jshen/data/CVUSA/CVUSA_subset"
    
    # Process each prompt
    start_time = time.time()
    for prompt in text_prompt_list:
        process_prompt_by_subfolder(prompt, subfolder_images, output_base_path, num_gpus)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/total_images:.2f} seconds")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main() 