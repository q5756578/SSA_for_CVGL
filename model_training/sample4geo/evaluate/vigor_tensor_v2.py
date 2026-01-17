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

def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader,'reference') 
    query_features, query_labels = predict(config, model, query_dataloader,'query')
    
    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
        
    # cleanup and free memory on GPU
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
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader,img_type='reference') 
    query_features, query_labels = predict(config, model, query_dataloader,img_type='query')
    
    print("Compute Scores Train:")
    results =  calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return results, near_dict



def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

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
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    hit_rate = 0.0
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i][0]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        # mask for semi pos
        mask = torch.ones(R)
        for near_pos in query_labels_np[i][1:]:
            mask[ref2index[near_pos]] = 0
        
        # calculate hit rate
        hit = (higher_sim * mask).sum()
        if hit < 1:
            hit_rate += 1.0
                
    
    results = results/ Q * 100.
    hit_rate = hit_rate / Q * 100
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))
    string.append('Hit_Rate: {:.4f}'.format(hit_rate))             
        
    print(' - '.join(string)) 

    return results

def calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels[:,0].cpu().numpy()
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
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
        
    results = results/ Q * 100.

    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))           
        
    print(' - '.join(string)) 

    return results[0]
   

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):

    query_labels = query_labels[:,0]
    
    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)


    # there might be more ground views for same sat view
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+2, dim=1)


    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

     
    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
    
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        
        nearest = topk_references[i][mask[i]][:neighbour_range]
    
        nearest_dict[query_labels[i].item()] = list(nearest)
    

    return nearest_dict


def copy_matching_images(gt_ref_path, gt_ref_name, source_dirs, query_dir, data_dir):
    """
    Copy images matching the given reference image name from specified source folders to query directory
    
    Args:
        gt_ref_path: Full path of the reference image
        gt_ref_name: Name of the reference image (without extension)
        source_dirs: List of source folder paths, each path should be a subfolder under a city
        query_dir: Query image directory
        data_dir: Root directory of the dataset containing all city folders
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
            
        # Determine output filename (using source folder name as prefix)
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
    Find and copy images matching reference image names from specified folders to query directory
    
    Args:
        query_dir: Directory for query images
        ref_paths: List of reference image paths
        ref_names: List of reference image names
        source_folders: List of source folder paths containing images to search
        data_dir: Root directory of the dataset containing all city folders
        output_prefix: Prefix for output filenames
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
        
        # Track if file is found in any folder
        found_in_any_folder = False
        
        # Iterate through each source folder
        for source_folder in source_folders:
            print(f"\nSearching in folder: {source_folder}")
            
            # Try different file extensions
            for ext in ['.jpg', '.png']:
                # Build target file path (using data_dir, city, source_folder and ref_name)
                target_path = os.path.join(data_dir, city, source_folder, ref_name + ext)
                
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
                    source_dirs=None,  # List of source folder paths
                    ensure_gt_in_top=False,  # Whether to ensure ground truth is in top_n
                    find_matching_images=False,  # Whether to find matching images
                    matching_source_folder=None):  # Source folder path for matching images
    """
    Randomly select query images, compute similarities, and save results including correctly corresponding
    reference images and BEV outputs
    
    Args:
        config: Configuration object
        model: Model object
        reference_dataloader: DataLoader for reference images
        query_dataloader: DataLoader for query images
        num_queries: Number of randomly selected query images
        top_n: Number of most similar satellite images to return for each query
        output_dir: Output directory
        step_size: Step size for batch processing
        source_dirs: List of source folder paths, if None then no matching images are copied
        ensure_gt_in_top: Whether to ensure ground truth is in top_n, if True only save results containing ground truth
        find_matching_images: Whether to find matching images
        matching_source_folder: Source folder path for matching images
    """
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Set matplotlib style
    plt.style.use('default')  # Use default style
    sns.set_theme()  # Use seaborn default theme
    
    # Get all available query image indices
    # Check dataset type
    if hasattr(query_dataloader.dataset, 'images'):
        # VIGOR dataset
        Q = len(query_dataloader.dataset.images)
        all_indices = list(range(Q))
    else:
        # CVACT dataset
        Q = len(query_dataloader.dataset)
        all_indices = list(range(Q))
    
    steps = Q // step_size + 1
    # Initialize processed query count
    processed_queries = 0
    valid_queries = 0
    
    print(f"\nStarting to process queries (target: {num_queries} valid queries)")
    
    # Extract features for all query images only
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
     
    # Matrix Q x R
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
        if hasattr(query_dataloader.dataset, 'images'):
            # VIGOR dataset
            query_path = query_dataloader.dataset.images[query_idx]
            query_name = os.path.basename(query_path).replace('.npy', '')
        else:
            # CVACT dataset
            query_path = query_dataloader.dataset.ground_paths[query_idx]
            query_name = os.path.basename(query_path).replace('.jpg', '')
        
        # Get top N most similar reference images
        top_k_scores, top_k_indices = torch.topk(similarity[query_idx], k=top_n)
        
        # Get correctly corresponding reference image (ground truth)
        gt_ref_idx = query_labels[query_idx][0].item()  # Get first correctly corresponding reference image index
        
        # If ensure_gt_in_top is set, check if ground truth is in top_n
        if ensure_gt_in_top and gt_ref_idx not in top_k_indices:
            print(f"Skipping query {query_name} as ground truth not in top {top_n}")
            continue
            
        # Create output directory for query image
        query_dir = os.path.join(output_dir, query_name)
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
        
        # Save query image
        if hasattr(query_dataloader.dataset, 'images'):
            # VIGOR dataset
            query_img = np.load(query_path, allow_pickle=True).item()['panorama']
        else:
            # CVACT dataset
            query_img = cv2.imread(query_path)
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
            
            # Use new color mapping function
            rgb_image = decode_layout(bev_output, num_classes)
            
            # Save RGB image
            rgb_image = rgb_image.permute(1, 2, 0).numpy()
            rgb_image = (rgb_image * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(query_dir, f"query_bev_rgb_{query_name}.jpg"), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
            
            # Only save heatmap for binary classification
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
        
        # Save correctly corresponding reference image (ground truth)
        if hasattr(reference_dataloader.dataset, 'idx2sat_path'):
            # VIGOR dataset
            gt_ref_path = reference_dataloader.dataset.idx2sat_path[gt_ref_idx]
            gt_ref_name = os.path.basename(gt_ref_path).replace('.npy', '')
            gt_ref_img = np.load(gt_ref_path, allow_pickle=True).item()['satellite']
        else:
            # CVACT dataset
            gt_ref_path = reference_dataloader.dataset.sat_paths[gt_ref_idx]
            gt_ref_name = os.path.basename(gt_ref_path).replace('.jpg', '')
            gt_ref_img = cv2.imread(gt_ref_path)
            gt_ref_img = cv2.cvtColor(gt_ref_img, cv2.COLOR_BGR2RGB)
        
        gt_ref_img = (gt_ref_img).astype(np.uint8)
        cv2.imwrite(os.path.join(query_dir, f"ground_truth_{gt_ref_name}.jpg"), cv2.cvtColor(gt_ref_img, cv2.COLOR_RGB2BGR))
        
        # If source folder list is provided, copy matching images
        if source_dirs is not None:
            copy_matching_images(gt_ref_path, gt_ref_name, source_dirs, query_dir, config.source_data_path)
        
        # Save most similar satellite images
        ref_names = []  # Store reference image names
        ref_paths = []
        for j, (score, ref_idx) in enumerate(zip(top_k_scores, top_k_indices)):
            if hasattr(reference_dataloader.dataset, 'idx2sat_path'):
                # VIGOR dataset
                ref_path = reference_dataloader.dataset.idx2sat_path[ref_idx.item()]
                ref_name = os.path.basename(ref_path).replace('.npy', '')
                ref_img = np.load(ref_path, allow_pickle=True).item()['satellite']
            else:
                # CVACT dataset
                ref_path = reference_dataloader.dataset.sat_paths[ref_idx.item()]
                ref_name = os.path.basename(ref_path).replace('.jpg', '')
                ref_img = cv2.imread(ref_path)
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
            
            ref_names.append(ref_name)  # Add to name list
            ref_paths.append(ref_path)
            
            # Save satellite image
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


# 在 eval_vigor_same.py 或 eval_vigor_cross.py 中添加以下代码

# 在评估完成后添加
