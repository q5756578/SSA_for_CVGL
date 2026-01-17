"""
This module provides evaluation and visualization tools for the VIGOR (Visual Geo-localization) model.
It includes functions for computing similarity scores, finding nearest neighbors, and visualizing results.
"""

import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict
import torch.distributed as dist
import datetime
import concurrent.futures
from torch.utils.data import DataLoader
import os
import shutil
import random
import cv2
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import torch.nn.functional as F
import einops

def decode_layout(image, num_classes=8):
    """
    Convert network output tensor to RGB visualization image.
    
    This function handles both binary classification (2 classes) and multi-class (8 classes) cases.
    For binary classification, it creates a grayscale image.
    For 8-class case, it maps each class to a specific color for visualization.
    
    Args:
        image (torch.Tensor): Network output tensor of shape [C, H, W] where C is number of classes
        num_classes (int): Number of classes, either 2 or 8
        
    Returns:
        torch.Tensor: RGB image tensor of shape [3, H, W] with values in range [0, 1]
    """
    if num_classes == 2:
        # Binary classification: convert to grayscale image
        binary = torch.argmax(image, dim=0)
        rgb_image = torch.zeros((3, binary.shape[0], binary.shape[1]), device=binary.device)
        rgb_image[0] = binary * 255  # R channel
        rgb_image[1] = binary * 255  # G channel
        rgb_image[2] = binary * 255  # B channel
        return rgb_image
    else:
        # 8-class case: map each class to a specific color
        color_dict = {
            (255., 178., 102.): 0,  # Building
            (64., 90., 255.): 1,    # Parking
            (102., 255., 102.): 2,  # Grass/Park/Playground
            (0., 153., 0.): 3,      # Forest
            (204., 229., 255.): 4,  # Water
            (192., 192., 192.): 5,  # Path
            (96., 96., 96.): 6,     # Road
            (255., 255., 255.): 7   # Background
        }

        # Get predicted class indices for each pixel
        indices = torch.argmax(image, dim=0)  # [H, W]
        
        # Initialize RGB image tensor
        rgb_image = torch.zeros((3, indices.shape[0], indices.shape[1]), device=indices.device)
        
        # Fill colors for each class
        for i, color in enumerate(color_dict.keys()):
            mask = indices == i
            for c in range(3):
                rgb_image[c][mask] = color[c] / 255.0
        
        return rgb_image

def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    """
    Evaluate model performance on reference and query datasets.
    
    This function computes recall@k metrics for image retrieval by:
    1. Extracting features from both reference and query images
    2. Computing similarity scores between query and reference features
    3. Calculating recall@k metrics for specified ranks
    
    Args:
        config: Configuration object containing model and evaluation parameters
        model: Trained model to evaluate
        reference_dataloader: DataLoader for reference/satellite images
        query_dataloader: DataLoader for query/ground images
        ranks (list): List of ranks to evaluate (e.g., [1, 5, 10] for Recall@1, Recall@5, Recall@10)
        step_size (int): Batch size for processing large datasets
        cleanup (bool): Whether to free GPU memory after evaluation
        
    Returns:
        list: Recall@k scores for each specified rank
    """
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader,'reference') 
    query_features, query_labels = predict(config, model, query_dataloader,'query')
    
    print("Compute Scores:")
    r1 = calculate_scores(query_features, reference_features, query_labels, reference_labels, 
                         step_size=step_size, ranks=ranks) 
        
    # Cleanup GPU memory if requested
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1


def calc_sim(config,
            model,
            reference_dataloader,
            query_dataloader, 
            ranks=[1, 5, 10],
            step_size=1000,
            cleanup=True):
    """
    Calculate similarity scores and nearest neighbors between query and reference features.
    
    This function performs two main tasks:
    1. Computes recall@k metrics for training data
    2. Finds nearest neighbors for each query image within a specified range
    
    Args:
        config: Configuration object containing model and evaluation parameters
        model: Trained model to evaluate
        reference_dataloader: DataLoader for reference/satellite images
        query_dataloader: DataLoader for query/ground images
        ranks (list): List of ranks to evaluate (e.g., [1, 5, 10] for Recall@1, Recall@5, Recall@10)
        step_size (int): Batch size for processing large datasets
        cleanup (bool): Whether to free GPU memory after evaluation
        
    Returns:
        tuple: (recall_score, nearest_dict)
            - recall_score: Recall@1 score for training data
            - nearest_dict: Dictionary mapping query labels to their nearest neighbors
    """
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader,img_type='reference') 
    query_features, query_labels = predict(config, model, query_dataloader,img_type='query')
    
    print("Compute Scores Train:")
    results = calculate_scores_train(query_features, reference_features, query_labels, reference_labels, 
                                   step_size=step_size, ranks=ranks) 
    
    # Find nearest neighbors for each query
    near_dict = calculate_nearest(query_features=query_features,
                                reference_features=reference_features,
                                query_labels=query_labels,
                                reference_labels=reference_labels,
                                neighbour_range=config.neighbour_range,
                                step_size=step_size)
            
    # Cleanup GPU memory if requested
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return results, near_dict



def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):
    """
    Calculate recall@k metrics for image retrieval evaluation.
    
    This function computes:
    1. Similarity scores between query and reference features
    2. Recall@k metrics for specified ranks
    3. Hit rate (percentage of queries where no incorrect matches rank higher than ground truth)
    
    Args:
        query_features (torch.Tensor): Feature vectors of query images
        reference_features (torch.Tensor): Feature vectors of reference images
        query_labels (torch.Tensor): Labels of query images, including ground truth and semi-positive samples
        reference_labels (torch.Tensor): Labels of reference images
        step_size (int): Batch size for processing large datasets
        ranks (list): List of ranks to evaluate (e.g., [1, 5, 10] for Recall@1, Recall@5, Recall@10)
        
    Returns:
        numpy.ndarray: Array of recall@k scores for each specified rank
    """
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    # Convert labels to numpy for faster processing
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    # Create mapping from reference label to index for quick lookup
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    # Calculate similarity matrix in batches to manage memory
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
     
    # Combine all batches into Q x R similarity matrix
    similarity = torch.cat(similarity, dim=0)
    
    # Add top 1% rank for additional evaluation
    topk.append(R//100)
    
    # Initialize results array and hit rate counter
    results = np.zeros([len(topk)])
    hit_rate = 0.0
    
    # Process each query image
    bar = tqdm(range(Q))
    
    for i in bar:
        # Get similarity score of ground truth reference
        gt_sim = similarity[i, ref2index[query_labels_np[i][0]]]
        
        # Count number of references with higher similarity than ground truth
        higher_sim = similarity[i,:] > gt_sim
         
        # Calculate ranking and update recall@k scores
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        # Create mask for semi-positive samples (exclude them from hit rate calculation)
        mask = torch.ones(R)
        for near_pos in query_labels_np[i][1:]:
            mask[ref2index[near_pos]] = 0
        
        # Calculate hit rate (no incorrect matches rank higher than ground truth)
        hit = (higher_sim * mask).sum()
        if hit < 1:
            hit_rate += 1.0
                
    
    # Convert counts to percentages
    results = results/ Q * 100.
    hit_rate = hit_rate / Q * 100
    
    bar.close()
    
    # Wait to close progress bar
    time.sleep(0.1)
    
    # Format and print results
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))
    string.append('Hit_Rate: {:.4f}'.format(hit_rate))             
        
    print(' - '.join(string)) 

    return results

def calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):
    """
    Calculate recall@k metrics for training data evaluation.
    
    This function is similar to calculate_scores but specifically for training data,
    where each query has a single ground truth reference (no semi-positive samples).
    
    Args:
        query_features (torch.Tensor): Feature vectors of query images
        reference_features (torch.Tensor): Feature vectors of reference images
        query_labels (torch.Tensor): Labels of query images (single ground truth per query)
        reference_labels (torch.Tensor): Labels of reference images
        step_size (int): Batch size for processing large datasets
        ranks (list): List of ranks to evaluate (e.g., [1, 5, 10] for Recall@1, Recall@5, Recall@10)
        
    Returns:
        float: Recall@1 score for training data
    """
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    # Get only the ground truth label for each query
    query_labels_np = query_labels[:,0].cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    # Create mapping from reference label to index for quick lookup
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    # Calculate similarity matrix in batches to manage memory
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
     
    # Combine all batches into Q x R similarity matrix
    similarity = torch.cat(similarity, dim=0)

    # Add top 1% rank for additional evaluation
    topk.append(R//100)
    
    # Initialize results array
    results = np.zeros([len(topk)])
    
    # Process each query image
    bar = tqdm(range(Q))
    
    for i in bar:
        # Get similarity score of ground truth reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # Count number of references with higher similarity than ground truth
        higher_sim = similarity[i,:] > gt_sim
         
        # Calculate ranking and update recall@k scores
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
        
    # Convert counts to percentages
    results = results/ Q * 100.

    bar.close()
    
    # Wait to close progress bar
    time.sleep(0.1)
    
    # Format and print results
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))           
        
    print(' - '.join(string)) 

    return results[0]
   

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):
    """
    Find nearest neighbors for each query image within a specified range.
    
    This function:
    1. Computes similarity scores between query and reference features
    2. Finds top-k nearest neighbors for each query
    3. Excludes ground truth matches from the results
    4. Returns a dictionary mapping query labels to their nearest neighbors
    
    Args:
        query_features (torch.Tensor): Feature vectors of query images
        reference_features (torch.Tensor): Feature vectors of reference images
        query_labels (torch.Tensor): Labels of query images
        reference_labels (torch.Tensor): Labels of reference images
        neighbour_range (int): Number of nearest neighbors to find for each query
        step_size (int): Batch size for processing large datasets
        
    Returns:
        dict: Dictionary mapping query labels to lists of their nearest neighbor labels
    """
    # Get only the ground truth label for each query
    query_labels = query_labels[:,0]
    
    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    # Calculate similarity matrix in batches to manage memory
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
     
    # Combine all batches into Q x R similarity matrix
    similarity = torch.cat(similarity, dim=0)

    # Get top-k nearest neighbors for each query
    # Add 2 to neighbour_range to account for potential ground truth matches
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+2, dim=1)

    # Get reference labels for top-k neighbors
    topk_references = []
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

    # Create mask to exclude ground truth matches
    mask = topk_references != query_labels.unsqueeze(1)
    
    # Convert to numpy for easier processing
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    
    # Create dictionary storing nearest neighbors for each query
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        # Get nearest neighbors excluding ground truth matches
        nearest = topk_references[i][mask[i]][:neighbour_range]
        nearest_dict[query_labels[i].item()] = list(nearest)
    
    return nearest_dict


def copy_matching_images(gt_ref_path, gt_ref_name, source_dirs, query_dir, data_dir):
    """
    Copy images matching a reference image name from source folders to query directory.
    
    This function:
    1. Determines the city from the reference image path
    2. Searches for matching images in specified source directories
    3. Copies found images to the query directory with appropriate naming
    
    Args:
        gt_ref_path (str): Full path of the reference image
        gt_ref_name (str): Name of the reference image (without extension)
        source_dirs (list or str): List of source folder paths or single source folder path
        query_dir (str): Directory to save copied images
        data_dir (str): Root directory containing all city folders
        
    Note:
        - Source directories should be subfolders under city folders
        - Supports both .jpg and .npy file formats
        - For .npy files, extracts and saves the satellite image
    """
    
    # Determine which city the image belongs to from gt_ref_path
    city = None
    for city_name in ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']:
        if city_name in gt_ref_path:
            city = city_name
            break
    
    if city is None:
        print(f"Warning: Could not determine city from path: {gt_ref_path}")
        return
    
    print(f"\nCopying matching images for {gt_ref_name} from {city}:")
    
    # Ensure source_dirs is a list
    if isinstance(source_dirs, str):
        source_dirs = [source_dirs]
    
    for source_dir in source_dirs:
        # Build complete source file path
        source_path = os.path.join(data_dir, city, source_dir, gt_ref_name+'.jpg')
        print(source_path)
        # Check if file exists
        if not os.path.exists(source_path):
            print(f"Warning: File not found in {data_dir}/{city}/{source_dir}: {gt_ref_name}")
            continue
            
        # Determine output filename (use source folder name as prefix)
        source_name = os.path.basename(source_dir)
        output_name = f"matching_{source_name}_{gt_ref_name}.jpg"
        output_path = os.path.join(query_dir, output_name)
        
        # Copy image
        if source_path.endswith('.npy'):
            # Load from npy file and save as jpg
            img = np.load(source_path, allow_pickle=True).item()['satellite']
            # Ensure image is in RGB order (convert if BGR)
            if img.shape[-1] == 3:
                img = img[..., ::-1]  # BGR to RGB
            img = (img).astype(np.uint8)
            cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            # Copy file directly
            shutil.copy2(source_path, output_path)
        
        print(f"Copied from {data_dir}/{city}/{source_dir}: {output_name}")

def find_and_copy_matching_images(query_dir, ref_paths, ref_names, source_folders, data_dir, output_prefix="matching_"):
    """
    Find and copy images matching reference image names from source folders to query directory.
    
    This function:
    1. Processes each reference image
    2. Searches for matching images in specified source folders
    3. Tries different file extensions (.jpg, .png)
    4. Copies found images to query directory with appropriate naming
    
    Args:
        query_dir (str): Directory to save copied images
        ref_paths (list): List of reference image paths
        ref_names (list): List of reference image names (without extension)
        source_folders (list): List of source folder paths to search in
        data_dir (str): Root directory containing all city folders
        output_prefix (str): Prefix for output filenames
        
    Note:
        - Supports both .jpg and .png file formats
        - Uses both OpenCV and PIL for image reading
        - Maintains original image format when copying
    """
    print(f"\nSearching for matching images:")
    
    for ref_path, ref_name in zip(ref_paths, ref_names):
        # Determine which city the image belongs to from ref_path
        city = None
        for city_name in ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']:
            if city_name in ref_path:
                city = city_name
                break
        
        if city is None:
            print(f"Warning: Could not determine city from path: {ref_path}")
            continue
            
        print(f"\nProcessing {ref_name} from {city}:")
        
        # Track if file was found in any folder
        found_in_any_folder = False
        
        # Iterate through each source folder
        for source_folder in source_folders:
            print(f"\nSearching in folder: {source_folder}")
            
            # Try different file extensions
            for ext in ['.jpg', '.png']:
                # Build target file path using data_dir, city, source_folder and ref_name
                target_path = os.path.join(data_dir, city, source_folder, ref_name + ext)
                print('target_path is',target_path)
                if os.path.exists(target_path):
                    # Build output filename using source folder name as prefix
                    source_name = os.path.basename(source_folder)
                    output_name = f"{output_prefix}{source_name}_{ref_name}{ext}"
                    output_path = os.path.join(query_dir, output_name)
                    print('output_path',output_path)
                    # Copy file
                    try:
                        # Read image and save as jpg
                        img = cv2.imread(target_path)
                        if img is not None:
                            cv2.imwrite(output_path, img)
                        else:
                            # If cv2 fails to read, try using PIL
                            img = Image.open(target_path)
                            img.save(output_path, 'JPEG')
                        
                        print(f"Copied: {ref_name}{ext} -> {output_name}")
                        found_in_any_folder = True
                    except Exception as e:
                        print(f"Error copying {ref_name}{ext}: {e}")
                        continue
        
        if not found_in_any_folder:
            print(f"No matching file found for {ref_name} in any source folder")

def save_top_matches(config,
                    model,
                    reference_dataloader,
                    query_dataloader,
                    num_queries=5,  # Number of randomly selected query images
                    top_n=10,       # Number of most similar satellite images to return for each query
                    output_dir="top_matches",  # Output directory
                    step_size=1000,
                    ensure_gt_in_top=False,  # Whether to ensure ground truth is in top_n
                    find_matching_images=False,  # Whether to find matching images
                    matching_source_folder=None):  # Source folder path for matching images
    """
    Save visualization results for top matching images between query and reference datasets.
    
    This function:
    1. Randomly selects query images
    2. Computes similarity scores between query and reference features
    3. Saves visualization results including:
       - Query images
       - Ground truth reference images
       - Top-N most similar reference images
       - BEV (Bird's Eye View) outputs
       - Heatmaps (for binary classification)
    4. Optionally finds and copies matching images from source folders
    
    Args:
        config: Configuration object containing model and evaluation parameters
        model: Trained model to evaluate
        reference_dataloader: DataLoader for reference/satellite images
        query_dataloader: DataLoader for query/ground images
        num_queries (int): Number of randomly selected query images to process
        top_n (int): Number of most similar satellite images to return for each query
        output_dir (str): Directory to save visualization results
        step_size (int): Batch size for processing large datasets
        ensure_gt_in_top (bool): Whether to ensure ground truth is in top_n matches
        find_matching_images (bool): Whether to find and copy matching images
        matching_source_folder (str): Source folder path for finding matching images
        
    Note:
        - Creates a separate subdirectory for each query image
        - Saves both RGB and heatmap visualizations for BEV outputs
        - Uses seaborn for heatmap visualization
        - Supports both binary and multi-class classification
    """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set matplotlib style
    plt.style.use('default')  # Use default style
    sns.set_theme()  # Use seaborn default theme
    
    # Get all available query image indices
    Q = len(query_dataloader.dataset.images)
    all_indices = list(range(Q))
    
    steps = Q // step_size + 1
    # Initialize processed query count
    processed_queries = 0
    valid_queries = 0
    
    print(f"\nStarting to process queries (target: {num_queries} valid queries)")
    
    # Extract features for all query images
    print("\nExtract Features for All Queries:")
    query_features, query_labels = predict(config, model, query_dataloader, 'query')
    
    # Extract features for all reference images
    print("\nExtract Reference Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader, 'reference')
    
    # Compute similarity matrix
    print("Compute Similarity Matrix:")

    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
     
    # Combine all batches into Q x R similarity matrix
    similarity = torch.cat(similarity, dim=0)
    
    # Get model's number of classes
    num_classes = model.module.classes if hasattr(model, 'module') else model.classes
    
    while valid_queries < num_queries and processed_queries < Q:
        # Randomly select an unprocessed query image
        if len(all_indices) == 0:
            break
            
        query_idx = random.choice(all_indices)
        all_indices.remove(query_idx)
        processed_queries += 1
        
        # Get query image path
        query_path = query_dataloader.dataset.images[query_idx]
        query_name = os.path.basename(query_path).replace('.npy', '')
        
        # Get top N most similar reference images
        top_k_scores, top_k_indices = torch.topk(similarity[query_idx], k=top_n)
        
        # Get correct reference image (ground truth)
        gt_ref_idx = query_labels[query_idx][0].item()  # Get first correct reference image index
        
        # If ensure_gt_in_top is set, check if ground truth is in top_n
        if ensure_gt_in_top and gt_ref_idx not in top_k_indices:
            print(f"Skipping query {query_name} as ground truth not in top {top_n}")
            continue
            
        # Create output directory for query image
        query_dir = os.path.join(output_dir, query_name)
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
        
        # Save query image
        query_img = np.load(query_path, allow_pickle=True).item()['panorama']
        query_img = (query_img).astype(np.uint8)
        cv2.imwrite(os.path.join(query_dir, f"query_{query_name}.jpg"), cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR))
        
        # Get and save BEV output for query image
        query_img_tensor = torch.from_numpy(query_img).permute(2, 0, 1).unsqueeze(0).float()
        query_img_tensor = query_img_tensor.to(config.device)
        with torch.no_grad():
            _, _, bev_output = model(query_img_tensor, query_img_tensor, None, True)
            bev_output = F.softmax(bev_output, dim=1)
            bev_output = bev_output.squeeze(0).cpu()
            
            # Use new color mapping function
            rgb_image = decode_layout(bev_output, num_classes)
            
            # Save RGB image
            rgb_image = rgb_image.permute(1, 2, 0).numpy()
            rgb_image = (rgb_image * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(query_dir, f"query_bev_rgb_{query_name}.jpg"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            
            # Only save heatmap for binary classification case
            if num_classes == 2:
                bev_probs = torch.exp(bev_output) / torch.sum(torch.exp(bev_output), dim=0, keepdim=True)
                bev_heatmap = bev_probs[1].numpy()
                
                # Create heatmap using seaborn
                plt.figure(figsize=(10, 10))
                sns.heatmap(bev_heatmap, 
                           cmap='RdYlBu_r',  # Use red-blue color scheme
                           vmin=0, vmax=1,    # Set value range
                           square=True,       # Keep square shape
                           cbar_kws={'label': 'Probability'},  # Add colorbar label
                           xticklabels=False,  # Remove x-axis labels
                           yticklabels=False)  # Remove y-axis labels
                
                # Save heatmap
                plt.savefig(os.path.join(query_dir, f"query_bev_heatmap_{query_name}.jpg"), 
                           bbox_inches='tight',  # Remove extra margins
                           dpi=300,             # Set high resolution
                           pad_inches=0)        # Remove padding
                plt.close()  # Close figure to free memory
        
        # Save correct reference image (ground truth)
        gt_ref_path = reference_dataloader.dataset.idx2sat_path[gt_ref_idx]
        gt_ref_name = os.path.basename(gt_ref_path).replace('.npy', '')
        gt_ref_img = np.load(gt_ref_path, allow_pickle=True).item()['satellite']
        gt_ref_img = (gt_ref_img).astype(np.uint8)
        cv2.imwrite(os.path.join(query_dir, f"ground_truth_{gt_ref_name}.jpg"), cv2.cvtColor(gt_ref_img, cv2.COLOR_RGB2BGR))
        
        # Save most similar satellite images
        ref_names = []  # Store reference image names
        ref_paths = []
        for j, (score, ref_idx) in enumerate(zip(top_k_scores, top_k_indices)):
            ref_path = reference_dataloader.dataset.idx2sat_path[ref_idx.item()]
            ref_name = os.path.basename(ref_path).replace('.npy', '')
            ref_names.append(ref_name)  # Add to name list
            ref_paths.append(ref_path)
            
            # Save satellite image
            ref_img = np.load(ref_path, allow_pickle=True).item()['satellite']
            ref_img = (ref_img).astype(np.uint8)
            cv2.imwrite(os.path.join(query_dir, f"rank{j+1}_score{score:.4f}_{ref_name}.jpg"), cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
        
        if find_matching_images and matching_source_folder:
            find_and_copy_matching_images(query_dir, ref_paths, ref_names, matching_source_folder, config.source_data_path)
        
        valid_queries += 1
        print(f"Processed valid query {valid_queries}/{num_queries}: {query_name}")
    
    # Clean up memory
    del reference_features, reference_labels, query_features, query_labels
    gc.collect()
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Total queries processed: {processed_queries}")
    print(f"Valid queries saved: {valid_queries}")


