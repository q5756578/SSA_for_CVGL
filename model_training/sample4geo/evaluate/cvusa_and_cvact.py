import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict
import os
import cv2
import random
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
import einops
from PIL import Image


def find_and_copy_matching_images_cvusa(query_dir, ref_paths, ref_names, source_folders, data_dir, output_prefix="matching_"):
    """
    Find and copy images matching reference image names from specified folders to query directory for CVUSA dataset,
    maintaining the original folder structure.
    
    Args:
        query_dir: Directory for query images
        ref_paths: List of reference image paths
        ref_names: List of reference image names
        source_folders: List of source folder paths containing images to search
        data_dir: Root directory of the dataset
        output_prefix: Prefix for output filenames
    """
    print(f"\nSearching for matching images:")
    
    # Ensure source_folders is a list
    if isinstance(source_folders, str):
        source_folders = [source_folders]
    
    for ref_path, ref_name in zip(ref_paths, ref_names):
        print(f"\nProcessing {ref_name}:")
        
        # Track if file is found in any folder
        found_in_any_folder = False
        
        # Iterate through each source folder
        for source_folder in source_folders:
            print(f"\nSearching in folder: {source_folder}")
            
            # Extract subfolder structure from original path and remove extra top-level directory
            rel_path = os.path.relpath(ref_path, data_dir)
            path_parts = rel_path.split(os.sep)
            if len(path_parts) > 2:  # If path contains more than two levels
                subfolder = os.path.join(*path_parts[1:-1])  # Skip first level directory
            else:
                subfolder = ""
            
            # Build target file path, maintaining original folder structure
            target_path = os.path.join(data_dir, source_folder, subfolder, f"{ref_name}.jpg")
            print(f"Looking for file: {target_path}")
            
            if os.path.exists(target_path):
                # Build output filename (using source folder name as prefix)
                source_name = os.path.basename(source_folder)
                output_name = f"{output_prefix}{source_name}_{ref_name}.jpg"
                output_path = os.path.join(query_dir, output_name)
                
                # Copy file
                try:
                    # Read image and save as jpg
                    img = cv2.imread(target_path)
                    if img is not None:
                        cv2.imwrite(output_path, img)
                    else:
                        # If cv2 fails to read, try PIL
                        img = Image.open(target_path)
                        img.save(output_path, 'JPEG')
                    
                    print(f"Copied: {ref_name} -> {output_name}")
                    found_in_any_folder = True
                except Exception as e:
                    print(f"Error copying {ref_name}: {e}")
                    continue
        
        if not found_in_any_folder:
            print(f"No matching file found for {ref_name} in any source folder")



def decode_layout(image, num_classes=8):
    """
    Convert network output to RGB image
    
    Args:
        image: Network output tensor
        num_classes: Number of classes, can be 2 or 8
    """
    if num_classes == 2:
        # Binary classification case: return binary image directly
        binary = torch.argmax(image, dim=0)
        rgb_image = torch.zeros((3, binary.shape[0], binary.shape[1]), device=binary.device)
        rgb_image[0] = binary * 255  # R channel
        rgb_image[1] = binary * 255  # G channel
        rgb_image[2] = binary * 255  # B channel
        return rgb_image
    else:
        # 8-class case: use color mapping
        color_dict = {
            (255., 178., 102.): 0,  # Building
            (64., 90., 255.): 1,    # Parking
            (102., 255., 102.): 2,  # Grass park playground
            (0., 153., 0.): 3,      # Forest
            (204., 229., 255.): 4,  # Water
            (192., 192., 192.): 5,  # Path
            (96., 96., 96.): 6,     # Road
            (255., 255., 255.): 7   # Background
        }

        # Get predicted class indices
        indices = torch.argmax(image, dim=0)  # [H, W]
        
        # Create RGB image
        rgb_image = torch.zeros((3, indices.shape[0], indices.shape[1]), device=indices.device)
        
        # Fill corresponding colors for each class
        for i, color in enumerate(color_dict.keys()):
            mask = indices == i
            for c in range(3):
                rgb_image[c][mask] = color[c] / 255.0
        
        return rgb_image


def find_and_copy_matching_images(query_dir, ref_paths, ref_names, source_folders, data_dir, output_prefix="matching_", dataset=None):
    """
    Find and copy images matching reference image names from specified folders to query directory
    
    Args:
        query_dir: Directory for query images
        ref_paths: List of reference image paths
        ref_names: List of reference image names
        source_folders: List of source folder paths containing images to search
        data_dir: Root directory of the dataset
        output_prefix: Prefix for output filenames
        dataset: Dataset name (e.g., 'CVACT')
    """
    print(f"\nSearching for matching images:")
    
    # Ensure source_folders is a list
    if isinstance(source_folders, str):
        source_folders = [source_folders]
    
    for ref_path, ref_name in zip(ref_paths, ref_names):
        print(f"\nProcessing {ref_name}:")
        
        # Track if file is found in any folder
        found_in_any_folder = False
        
        # Iterate through each source folder
        for source_folder in source_folders:
            print(f"\nSearching in folder: {source_folder}")
            
            # Try different file extensions
            for ext in ['.jpg', '.png', '.jpeg']:
                # Build target file path
                if dataset == 'CVACT':
                    target_path = os.path.join(data_dir, source_folder, ref_name +'_satView_polish'+ ext)
                else:
                    target_path = os.path.join(data_dir, source_folder, ref_name + ext)
                print('target path', target_path)
                if os.path.exists(target_path):
                    # Build output filename (using source folder name as prefix)
                    source_name = os.path.basename(source_folder)
                    output_name = f"{output_prefix}{source_name}_{ref_name}{ext}"
                    output_path = os.path.join(query_dir, output_name)
                    
                    # Copy file
                    try:
                        # Read image and save as jpg
                        img = cv2.imread(target_path)
                        if img is not None:
                            cv2.imwrite(output_path, img)
                        else:
                            # If cv2 fails to read, try PIL
                            img = Image.open(target_path)
                            img.save(output_path, 'JPEG')
                        
                        print(f"Copied: {ref_name}{ext} -> {output_name}")
                        found_in_any_folder = True
                    except Exception as e:
                        print(f"Error copying {ref_name}{ext}: {e}")
                        continue
        
        if not found_in_any_folder:
            print(f"No matching file found for {ref_name} in any source folder")



def save_top_matches_cvact(config,
                          model,
                          reference_dataloader,
                          query_dataloader,
                          num_queries=5,
                          top_n=10,
                          output_dir="cvact_top_matches",
                          step_size=1000,
                          ensure_gt_in_top=False,  # Whether to ensure ground truth is in top_n
                          find_matching_images=False,  # Whether to find matching images
                          matching_source_folder=None):  # Source folder path for matching images
    """
    Save visualization results for CVACT dataset, handling binary BEV output
    
    Args:
        config: Configuration object
        model: Model object
        reference_dataloader: DataLoader for reference images
        query_dataloader: DataLoader for query images
        num_queries: Number of randomly selected query images
        top_n: Number of most similar satellite images to return for each query
        output_dir: Output directory
        step_size: Step size for batch processing
        ensure_gt_in_top: Whether to ensure ground truth is in top_n
        find_matching_images: Whether to find matching images
        matching_source_folder: List of source folder paths for matching images
    """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set matplotlib style
    plt.style.use('default')
    sns.set_theme()
    
    # Get all available query image indices
    Q = len(query_dataloader.dataset)
    all_indices = list(range(Q))
    
    steps = Q // step_size + 1
    processed_queries = 0
    valid_queries = 0
    
    print(f"\nStarting to process queries (target: {num_queries} valid queries)")
    
    # Extract features
    print("\nExtract Features for All Queries:")
    query_features, query_labels = predict(config, model, query_dataloader, img_type='query')
    
    print("\nExtract Reference Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader, img_type='reference')
    
    # Compute similarity matrix
    print("Compute Similarity Matrix:")
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
    
    similarity = torch.cat(similarity, dim=0)
    
    # Determine dataset type and corresponding folder paths
    is_eval = hasattr(query_dataloader.dataset, 'samples')
    if is_eval:
        data_subfolder = 'CVACT_new'
    else:
        data_subfolder = 'CVACT'
    
    matching_cons_folder = os.path.join(config.data_folder, data_subfolder)
    
    while valid_queries < num_queries and processed_queries < Q:
        if len(all_indices) == 0:
            break
            
        query_idx = random.choice(all_indices)
        all_indices.remove(query_idx)
        processed_queries += 1
        
        # Get query image information
        if is_eval:
            # CVACTDatasetEval
            query_id = query_dataloader.dataset.samples[query_idx]
        else:
            # CVACTDatasetTest
            query_id = query_dataloader.dataset.test_ids[query_idx]
            
        query_path = os.path.join(config.data_folder, data_subfolder, 'streetview', f'{query_id}_grdView.jpg')
        query_name = query_id
        
        # Get top N most similar reference images
        top_k_scores, top_k_indices = torch.topk(similarity[query_idx], k=top_n)
        
        # Get correctly corresponding reference image (ground truth)
        gt_ref_idx = query_labels[query_idx].item()
        
        # If ensure_gt_in_top is set, check if ground truth is in top_n
        if ensure_gt_in_top and gt_ref_idx not in top_k_indices:
            print(f"Skipping query {query_name} as ground truth not in top {top_n}")
            continue
        
        # Create output directory for query image
        query_dir = os.path.join(output_dir, query_name)
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
        
        try:
            # Save query image
            query_img = cv2.imread(query_path)
            if query_img is None:
                print(f"Warning: Could not read query image: {query_path}")
                continue
                
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            query_img = (query_img).astype(np.uint8)
            cv2.imwrite(os.path.join(query_dir, f"query_{query_name}.jpg"), cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR))
            
            # Get and save BEV output for query image
            query_img_tensor = torch.from_numpy(query_img).permute(2, 0, 1).unsqueeze(0).float()
            query_img_tensor = query_img_tensor.to(config.device)
            with torch.no_grad():
                _, _, bev_output = model(query_img_tensor, query_img_tensor, None, True)
                bev_output = F.softmax(bev_output, dim=1)
                bev_output = bev_output.squeeze(0).cpu()
                
                # Generate binary image
                binary = torch.argmax(bev_output, dim=0)
                binary_img = binary.numpy().astype(np.uint8) * 255
                cv2.imwrite(os.path.join(query_dir, f"query_bev_binary_{query_name}.jpg"), binary_img)
                
                # Generate heatmap
                bev_probs = torch.exp(bev_output) / torch.sum(torch.exp(bev_output), dim=0, keepdim=True)
                bev_heatmap = bev_probs[1].numpy()
                
                plt.figure(figsize=(10, 10))
                sns.heatmap(bev_heatmap, 
                           cmap='RdYlBu_r',
                           vmin=0, vmax=1,
                           square=True,
                           cbar_kws={'label': 'Probability'},
                           xticklabels=False,
                           yticklabels=False)
                
                plt.savefig(os.path.join(query_dir, f"query_bev_heatmap_{query_name}.jpg"),
                           bbox_inches='tight',
                           dpi=300,
                           pad_inches=0)
                plt.close()
            
            # Save correctly corresponding reference image (ground truth)
            if is_eval:
                # CVACTDatasetEval
                gt_ref_id = reference_dataloader.dataset.samples[gt_ref_idx]
            else:
                # CVACTDatasetTest
                gt_ref_id = reference_dataloader.dataset.test_ids[gt_ref_idx]
                
            gt_ref_path = os.path.join(config.data_folder, data_subfolder, 'satview_polish', f'{gt_ref_id}_satView_polish.jpg')
            gt_ref_img = cv2.imread(gt_ref_path)
            if gt_ref_img is None:
                print(f"Warning: Could not read ground truth image: {gt_ref_path}")
                continue
                
            gt_ref_img = cv2.cvtColor(gt_ref_img, cv2.COLOR_BGR2RGB)
            gt_ref_img = (gt_ref_img).astype(np.uint8)
            cv2.imwrite(os.path.join(query_dir, f"ground_truth_{gt_ref_id}.jpg"), cv2.cvtColor(gt_ref_img, cv2.COLOR_RGB2BGR))
            
            # Save most similar satellite images
            ref_names = []  # Store reference image names
            ref_paths = []  # Store reference image paths
            
            for j, (score, ref_idx) in enumerate(zip(top_k_scores, top_k_indices)):
                if is_eval:
                    # CVACTDatasetEval
                    ref_id = reference_dataloader.dataset.samples[ref_idx.item()]
                else:
                    # CVACTDatasetTest
                    ref_id = reference_dataloader.dataset.test_ids[ref_idx.item()]
                    
                ref_path = os.path.join(config.data_folder, data_subfolder, 'satview_polish', f'{ref_id}_satView_polish.jpg')
                ref_img = cv2.imread(ref_path)
                if ref_img is None:
                    print(f"Warning: Could not read reference image: {ref_path}")
                    continue
                    
                ref_names.append(ref_id)  # Add to name list
                ref_paths.append(ref_path)  # Add to path list
                
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_img = (ref_img).astype(np.uint8)
                cv2.imwrite(os.path.join(query_dir, f"rank{j+1}_score{score:.4f}_{ref_id}.jpg"), cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
            
            # If matching image search is enabled
            if find_matching_images and matching_source_folder:
                find_and_copy_matching_images(query_dir, ref_paths, ref_names, matching_source_folder, matching_cons_folder, dataset='CVACT')
            
            valid_queries += 1
            print(f"Processed valid query {valid_queries}/{num_queries}: {query_name}")
            
        except Exception as e:
            print(f"Error processing query {query_name}: {str(e)}")
            continue
    
    # Clean up memory
    del reference_features, reference_labels, query_features, query_labels
    gc.collect()
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Total queries processed: {processed_queries}")
    print(f"Valid queries saved: {valid_queries}")

def save_top_matches_cvusa(config,
                          model,
                          reference_dataloader,
                          query_dataloader,
                          num_queries=5,
                          top_n=10,
                          output_dir="cvusa_top_matches",
                          step_size=1000,
                          ensure_gt_in_top=False,  # Whether to ensure ground truth is in top_n
                          find_matching_images=False,  # Whether to find matching images
                          matching_source_folder=None):  # Source folder path for matching images
    """
    Save visualization results for CVUSA dataset, handling BEV output
    
    Args:
        config: Configuration object
        model: Model object
        reference_dataloader: DataLoader for reference images
        query_dataloader: DataLoader for query images
        num_queries: Number of randomly selected query images
        top_n: Number of most similar satellite images to return for each query
        output_dir: Output directory
        step_size: Step size for batch processing
        ensure_gt_in_top: Whether to ensure ground truth is in top_n
        find_matching_images: Whether to find matching images
        matching_source_folder: List of source folder paths for matching images
    """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set matplotlib style
    plt.style.use('default')
    sns.set_theme()
    
    # Get all available query image indices
    Q = len(query_dataloader.dataset)
    all_indices = list(range(Q))
    
    steps = Q // step_size + 1
    processed_queries = 0
    valid_queries = 0
    
    print(f"\nStarting to process queries (target: {num_queries} valid queries)")
    print(f"Total available queries: {Q}")
    
    # Extract features
    print("\nExtract Features for All Queries:")
    query_features, query_labels = predict(config, model, query_dataloader, img_type='query')
    
    print("\nExtract Reference Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader, img_type='reference')
    
    # Compute similarity matrix
    print("Compute Similarity Matrix:")
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = min(start + step_size, Q)  # Ensure not exceeding range
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
    
    similarity = torch.cat(similarity, dim=0)
    
    while valid_queries < num_queries and processed_queries < Q:
        if len(all_indices) == 0:
            break
            
        query_idx = random.choice(all_indices)
        all_indices.remove(query_idx)
        processed_queries += 1
        
        try:
            # Verify index is within valid range
            if query_idx >= Q:
                print(f"Warning: Query index {query_idx} is out of bounds (max: {Q-1})")
                continue
                
            # Get query image information
            query_path = query_dataloader.dataset.images[query_idx]
            query_id = os.path.splitext(os.path.basename(query_path))[0]
            query_path = os.path.join(config.data_folder, query_path)
            query_name = query_id
            
            # Get top N most similar reference images
            top_k_scores, top_k_indices = torch.topk(similarity[query_idx], k=min(top_n, len(reference_dataloader.dataset)))
            
            # Get correctly corresponding reference image (ground truth)
            gt_ref_idx = query_labels[query_idx].item()
            
            # Verify ground truth index is within valid range
            if gt_ref_idx >= len(reference_dataloader.dataset):
                print(f"Warning: Ground truth index {gt_ref_idx} is out of bounds (max: {len(reference_dataloader.dataset)-1})")
                continue
            
            # If ensure_gt_in_top is set, check if ground truth is in top_n
            if ensure_gt_in_top and gt_ref_idx not in top_k_indices:
                print(f"Skipping query {query_name} as ground truth not in top {top_n}")
                continue
            
            # Create output directory for query image
            query_dir = os.path.join(output_dir, query_name)
            if not os.path.exists(query_dir):
                os.makedirs(query_dir)
            
            # Save query image
            query_img = cv2.imread(query_path)
            if query_img is None:
                print(f"Warning: Could not read query image: {query_path}")
                continue
                
            query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
            query_img = (query_img).astype(np.uint8)
            cv2.imwrite(os.path.join(query_dir, f"query_{query_name}.jpg"), cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR))
            
            # Get and save BEV output for query image
            query_img_tensor = torch.from_numpy(query_img).permute(2, 0, 1).unsqueeze(0).float()
            query_img_tensor = query_img_tensor.to(config.device)
            with torch.no_grad():
                _, _, bev_output = model(query_img_tensor, query_img_tensor, None, True)
                bev_output = F.softmax(bev_output, dim=1)
                bev_output = bev_output.squeeze(0).cpu()
                
                # Generate binary image
                binary = torch.argmax(bev_output, dim=0)
                binary_img = binary.numpy().astype(np.uint8) * 255
                cv2.imwrite(os.path.join(query_dir, f"query_bev_binary_{query_name}.jpg"), binary_img)
                
                # Generate heatmap
                bev_probs = torch.exp(bev_output) / torch.sum(torch.exp(bev_output), dim=0, keepdim=True)
                bev_heatmap = bev_probs[1].numpy()
                
                plt.figure(figsize=(10, 10))
                sns.heatmap(bev_heatmap, 
                           cmap='RdYlBu_r',
                           vmin=0, vmax=1,
                           square=True,
                           cbar_kws={'label': 'Probability'},
                           xticklabels=False,
                           yticklabels=False)
                
                plt.savefig(os.path.join(query_dir, f"query_bev_heatmap_{query_name}.jpg"),
                           bbox_inches='tight',
                           dpi=300,
                           pad_inches=0)
                plt.close()
            
            # Save correctly corresponding reference image (ground truth)
            gt_ref_path = reference_dataloader.dataset.images[gt_ref_idx]
            gt_ref_id = os.path.splitext(os.path.basename(gt_ref_path))[0]
            gt_ref_path = os.path.join(config.data_folder, gt_ref_path)
            gt_ref_img = cv2.imread(gt_ref_path)
            if gt_ref_img is None:
                print(f"Warning: Could not read ground truth image: {gt_ref_path}")
                continue
                
            gt_ref_img = cv2.cvtColor(gt_ref_img, cv2.COLOR_BGR2RGB)
            gt_ref_img = (gt_ref_img).astype(np.uint8)
            cv2.imwrite(os.path.join(query_dir, f"ground_truth_{gt_ref_id}.jpg"), cv2.cvtColor(gt_ref_img, cv2.COLOR_RGB2BGR))
            
            # Save most similar satellite images
            ref_names = []  # Store reference image names
            ref_paths = []  # Store reference image paths
            
            for j, (score, ref_idx) in enumerate(zip(top_k_scores, top_k_indices)):
                ref_idx = ref_idx.item()
                if ref_idx >= len(reference_dataloader.dataset):
                    print(f"Warning: Reference index {ref_idx} is out of bounds (max: {len(reference_dataloader.dataset)-1})")
                    continue
                    
                ref_path = reference_dataloader.dataset.images[ref_idx]
                ref_id = os.path.splitext(os.path.basename(ref_path))[0]
                ref_path = os.path.join(config.data_folder, ref_path)
                ref_img = cv2.imread(ref_path)
                if ref_img is None:
                    print(f"Warning: Could not read reference image: {ref_path}")
                    continue
                    
                ref_names.append(ref_id)  # Add to name list
                ref_paths.append(ref_path)  # Add to path list
                
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                ref_img = (ref_img).astype(np.uint8)
                cv2.imwrite(os.path.join(query_dir, f"rank{j+1}_score{score:.4f}_{ref_id}.jpg"), cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
            
            # If matching image search is enabled
            if find_matching_images and matching_source_folder:
                find_and_copy_matching_images_cvusa(query_dir, ref_paths, ref_names, matching_source_folder, config.data_folder)
            
            valid_queries += 1
            print(f"Processed valid query {valid_queries}/{num_queries}: {query_name}")
            
        except Exception as e:
            print(f"Error processing query {query_idx}: {str(e)}")
            continue
    
    # Clean up memory
    del reference_features, reference_labels, query_features, query_labels
    gc.collect()
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Total queries processed: {processed_queries}")
    print(f"Valid queries saved: {valid_queries}")


def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    """
    Evaluate model performance on the dataset
    
    Args:
        config: Configuration object
        model: Model object
        reference_dataloader: DataLoader for reference images
        query_dataloader: DataLoader for query images
        ranks: List of ranks to evaluate
        step_size: Step size for batch processing
        cleanup: Whether to clean up memory after evaluation
    """
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader, img_type='reference') 
    query_features, query_labels = predict(config, model, query_dataloader, img_type='query')
    
    print("Compute Scores:")
    r1 = calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
        
    # Clean up and free memory on GPU
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
    Calculate similarity scores and nearest neighbors
    
    Args:
        config: Configuration object
        model: Model object
        reference_dataloader: DataLoader for reference images
        query_dataloader: DataLoader for query images
        ranks: List of ranks to evaluate
        step_size: Step size for batch processing
        cleanup: Whether to clean up memory after calculation
    """
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader, img_type='reference') 
    query_features, query_labels = predict(config, model, query_dataloader, img_type='query')
    
    print("Compute Scores Train:")
    r1 = calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # Clean up and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, near_dict




def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):
    """
    Calculate recall scores at specified ranks
    
    Args:
        query_features: Features of query images
        reference_features: Features of reference images
        query_labels: Labels of query images
        reference_labels: Labels of reference images
        step_size: Step size for batch processing
        ranks: List of ranks to evaluate
    """
    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
     
    # Matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    
    topk.append(R//100)
    results = np.zeros([len(topk)])
    
    bar = tqdm(range(Q))
    
    for i in bar:
        # Similarity value of ground truth reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # Number of references with higher similarity than ground truth
        higher_sim = similarity[i,:] > gt_sim
        
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
    results = results/ Q * 100.
    
    bar.close()
    
    # Wait to close progress bar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))            
        
    print(' - '.join(string)) 

    return results
    

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):
    """
    Calculate nearest neighbors for each query
    
    Args:
        query_features: Features of query images
        reference_features: Features of reference images
        query_labels: Labels of query images
        reference_labels: Labels of reference images
        neighbour_range: Range to search for neighbors
        step_size: Step size for batch processing
    """
    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity = []
    
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
     
    # Matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+1, dim=1)

    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

    # Mask for ids without ground truth hits
    mask = topk_references != query_labels.unsqueeze(1)
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    
    # Dictionary that only stores ids where similarity is higher than the lowest ground truth hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        nearest = topk_references[i][mask[i]][:neighbour_range]
        nearest_dict[query_labels[i].item()] = list(nearest)
    
    return nearest_dict
