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
        
        # Move results to main directory
        for filename in os.listdir(temp_dir):
            src = os.path.join(temp_dir, filename)
            dst = os.path.join(target_path, filename)
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

def process_prompt(prompt, image_files, target_path, num_gpus=4):
    """
    Process all images for a single prompt using multi-process parallel processing
    
    Args:
        prompt: Text prompt
        image_files: List of all image files
        target_path: Output directory
        num_gpus: Number of GPUs
    """
    print(f"\nProcessing prompt: {prompt}")
    ensure_empty_folder(target_path)
    
    # Split image list into multiple batches
    batch_size = math.ceil(len(image_files) / num_gpus)
    image_batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    
    # Create queue for inter-process communication
    queue = Queue()
    
    # Create and start processes
    processes = []
    for i in range(num_gpus):
        p = Process(
            target=process_batch,
            args=(i, image_batches[i], prompt, target_path, queue)
        )
        processes.append(p)
        p.start()
    
    # Wait for all processes to complete and collect results
    success_count = 0
    error_count = 0
    for _ in range(num_gpus):
        success, message = queue.get()
        if success:
            success_count += 1
        else:
            error_count += 1
        print(message)
    
    # Wait for all processes to end
    for p in processes:
        p.join()
    
    print(f"\nProcessing completed for prompt: {prompt}")
    print(f"Successfully processed: {success_count} batches")
    if error_count > 0:
        print(f"Failed to process: {error_count} batches")

def main():
    # Set parameters
    image_path = '/CVACT_new/satview_polish/'
    text_prompt_list = ['construction', 'edifice' ]
    num_gpus = 4
    
    # Get all image files
    image_files = glob.glob(os.path.join(image_path, "*.jpg"))  # Adjust based on actual file extension
    total_images = len(image_files)
    print(f"Total images to process: {total_images}")
    
    # Process each prompt
    start_time = time.time()
    for prompt in text_prompt_list:
        target_path = f"./test_samples_{prompt}/"
        process_prompt(prompt, image_files, target_path, num_gpus)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nTotal processing time: {total_time:.2f} seconds")
    print(f"Average time per image: {total_time/total_images:.2f} seconds")

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()