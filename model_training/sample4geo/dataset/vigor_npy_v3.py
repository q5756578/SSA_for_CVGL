import cv2
import numpy as np
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
from collections import defaultdict
import time
import copy
import torch
from torch.utils.data import Dataset, DataLoader
import threading
from queue import Queue
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
#import einops
# Original version of the dataset without additional data caching mechanism, using VIGORv2 labels for training.

class VigorDatasetTrain(Dataset):
    """
    VIGOR dataset training class for ground-to-satellite image matching.
    
    Args:
        data_folder (str): Path to the data storage folder
        same_area (bool): Whether to use data from the same area
        transforms_query (callable, optional): Transformations for ground view images
        transforms_reference (callable, optional): Transformations for satellite view images
        transforms_depth (callable, optional): Transformations for ground view depth maps
        transforms_bev (callable, optional): Transformations for satellite view BEV label images
        prob_flip (float): Probability of horizontal flipping
        prob_rotate (float): Probability of rotation
        shuffle_batch_size (int): Batch size for shuffling
    """
    
    def __init__(self,
                 data_folder,                # Path to data storage folder
                 same_area=True,             # Whether to use same area data
                 transforms_query=None,      # Transformations for ground view images
                 transforms_reference=None,  # Transformations for satellite view images
                 transforms_depth=None,      # Transformations for ground view depth maps
                 transforms_bev=None,        # Transformations for satellite view BEV label images
                 prob_flip=0.0,              # Probability of horizontal flipping
                 prob_rotate=0.0,            # Probability of rotation
                 shuffle_batch_size=128,     # Batch size for shuffling
                 ):
        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference
        self.transforms_depth = transforms_depth
        self.transforms_bev = transforms_bev
        
        # Load different city lists based on same_area flag
        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            self.cities = ['NewYork', 'Seattle']

        self.layout_mode = 'discrete'
        
        # Load satellite image list
        sat_list = []
        for city in self.cities:
            # Read satellite image path file and build DataFrame with paths
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, sep=r'\s+')
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1).str.replace(".png", ".npy")
            df_tmp["path"] = df_tmp['path'].str.replace("VIGOR", "VIGOR_npy", regex=False)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)

        # Create satellite image index to path mapping
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))  # Map satellite image names to indices
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))  # Map indices to satellite image names
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))  # Map indices to satellite image paths

        # Load ground view image data
        ground_list = []
        for city in self.cities:
            # Select different data files based on same_area flag
            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_train.txt', header=None, sep=r'\s+')
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, sep=r'\s+')
            
            # Rename columns and add path column
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0: "ground",
                                                                     1: "sat",
                                                                     4: "sat_np1",
                                                                     7: "sat_np2",
                                                                     10: "sat_np3"})
            df_tmp["ground_npy"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/panorama/{x.ground}', axis=1).str.replace(".jpg", ".npy")
            df_tmp["ground_npy"] = df_tmp['ground_npy'].str.replace("VIGOR", "VIGOR_npy", regex=False)
            df_tmp['BEV_label'] = df_tmp.apply(lambda x: f'{data_folder}/{city}/sat_npy/{x.sat}', axis=1).str.replace(".png", ".npy")
            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
            
            ground_list.append(df_tmp)

        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)

        # Create ground data index to path mapping
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_npy_path = dict(zip(self.df_ground.index, self.df_ground.ground_npy))
        self.idx2bev_label = dict(zip(self.df_ground.index, self.df_ground.BEV_label))
        
        # Create ground image and satellite image pairs
        self.pairs = list(zip(self.df_ground.index, self.df_ground.sat))
        self.idx2pairs = defaultdict(list)
        for pair in self.pairs:
            self.idx2pairs[pair[1]].append(pair)

        # Extract satellite image indices related to ground images
        self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values
        
        # Deep copy pair list for shuffling
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):
        """
        Get a data sample at the specified index.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (query_img, reference_img, que_bevlab, label)
                - query_img: Ground view image
                - reference_img: Satellite view image
                - que_bevlab: BEV label image
                - label: Ground truth label
        """
        # Get ground and satellite image pair at the specified index
        idx_ground, idx_sat = self.samples[index]
        
        # Load ground view image and convert to RGB
        query_img = np.load(self.idx2ground_npy_path[idx_ground], allow_pickle=True)
        
        # Load satellite view image and convert to RGB
        reference_img = np.load(self.idx2sat_path[idx_sat], allow_pickle=True)
        
        # Load BEV label
        que_bevlab = np.load(self.idx2bev_label[idx_ground], allow_pickle=True).item()
        que_bevlab = que_bevlab['sat_label01']
        que_bevlab = np.expand_dims(que_bevlab, axis=-1)
        
        # Randomly flip ground and satellite images horizontally
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1)
            que_bevlab = cv2.flip(que_bevlab, 1)
            
        # Apply predefined image transformations
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']
        if self.transforms_bev is not None:
            que_bevlab = self.transforms_bev(image=que_bevlab)['image']
            que_bevlab = que_bevlab // 255
            
        # Randomly rotate ground and satellite images
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))  # Rotate satellite image
            que_bevlab = torch.rot90(que_bevlab, k=r, dims=(1, 2))  # Rotate BEV label
            c, h, w = query_img.shape
            shifts = -w // 4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)  # Roll ground image to simulate rotation
        
        label = torch.tensor(idx_sat, dtype=torch.long)  # Label is satellite image index
        que_bevlab = que_bevlab.long()
        
        return query_img, reference_img, que_bevlab, label
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.samples)
    
    def decode_layoutv2(self, image, layout_mode='discrete'):
        """
        Decode RGB image to label image using efficient NumPy operations.
        
        Args:
            image: Input RGB image with shape (height, width, channels)
            layout_mode: Output mode, either 'discrete' or 'one_hot'
            
        Returns:
            Decoded label image
        """
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

        # Ensure input is a 3D array
        if image.ndim != 3:
            raise ValueError("Input image must be a 3D array with shape (height, width, channels)")

        # Normalize input image to [0,1] range
        image = image.astype(np.float32) / 255.0
        
        # Create color lookup table
        color_keys = np.array(list(color_dict.keys())) / 255.0
        color_values = np.array(list(color_dict.values()))
        num_classes = len(color_dict)

        # Reshape image array for computation
        h, w = image.shape[:2]
        image_reshaped = image.reshape(-1, 3)  # (height*width, 3)

        # Calculate L1 distance between each pixel and all colors
        distances = np.sum(np.abs(image_reshaped[:, np.newaxis, :] - color_keys), axis=2)
        
        # Find index of minimum distance
        min_indices = np.argmin(distances, axis=1)
        
        if layout_mode == 'discrete':
            # Return class indices directly
            return min_indices.reshape(h, w)
        else:  # one_hot
            # Create one-hot encoding
            one_hot = np.zeros((h * w, num_classes), dtype=np.uint8)
            one_hot[np.arange(h * w), min_indices] = 1
            return one_hot.reshape(h, w, num_classes)

    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        """
        Custom shuffle function for unique class_id sampling in batch.
        
        Args:
            sim_dict (dict, optional): Dictionary containing similarity information
            neighbour_select (int): Number of neighbors to select
            neighbour_range (int): Range to search for neighbors
        """
        print("\nShuffling Dataset...")
        
        # Create deep copies of pairs and mappings
        pair_pool = copy.deepcopy(self.pairs)
        idx2pair_pool = copy.deepcopy(self.idx2pairs)
        
        neighbour_split = neighbour_select // 2
        
        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)
        
        # Shuffle pairs order
        random.shuffle(pair_pool)
        
        # Track used pairs and batch indices
        pairs_epoch = set()
        idx_batch = set()
        
        # Initialize batch containers
        batches = []
        current_batch = []
        
        # Initialize counter for failed additions
        break_counter = 0
        
        # Initialize progress bar
        pbar = tqdm()

        while True:
            pbar.update()
            
            if len(pair_pool) > 0:
                pair = pair_pool.pop(0)
                _, idx = pair
                
                if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)
                    
                    # Remove from pool used for similarity sampling
                    idx2pair_pool[idx].remove(pair)
                    
                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                        # Perform similarity sampling
                        near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])
                        near_always = copy.deepcopy(near_similarity[:neighbour_split])
                        near_random = copy.deepcopy(near_similarity[neighbour_split:])
                        random.shuffle(near_random)
                        near_random = near_random[:neighbour_split]
                        near_similarity_select = near_always + near_random
                        
                        for idx_near in near_similarity_select:
                            if len(current_batch) >= self.shuffle_batch_size:
                                break
                            
                            if idx_near not in idx_batch:
                                near_pairs = copy.deepcopy(idx2pair_pool[idx_near])
                                random.shuffle(near_pairs)
                                
                                for near_pair in near_pairs:
                                    idx_batch.add(idx_near)
                                    current_batch.append(near_pair)
                                    pairs_epoch.add(near_pair)
                                    
                                    idx2pair_pool[idx_near].remove(near_pair)
                                    similarity_pool[idx].remove(idx_near)
                                    
                                    # Only select one view
                                    break
                            
                        break_counter = 0
                        
                    else:
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)
                    
                    break_counter += 1
                    
                    if break_counter >= 1024:
                        break
                
                else:
                    break

                if len(current_batch) >= self.shuffle_batch_size:
                    # Add current batch to batches
                    batches.extend(current_batch)
                    idx_batch = set()
                    current_batch = []
        
        pbar.close()
        
        # Wait before closing progress bar
        time.sleep(0.3)
        
        # Check if we need to expand neighbour_range
        while len(batches) < 0.8 * len(self.pairs) and neighbour_range < len(self.pairs):
            print(f"\nExpanding neighbour_range from {neighbour_range} to {neighbour_range * 2}")
            neighbour_range *= 2
            
            # Reset pools for new iteration
            pair_pool = copy.deepcopy(self.pairs)
            idx2pair_pool = copy.deepcopy(self.idx2pairs)
            if sim_dict is not None:
                similarity_pool = copy.deepcopy(sim_dict)
            
            # Create new batches
            new_batches = []
            current_batch = []
            idx_batch = set()
            pairs_epoch = set(batches)  # Start with already selected pairs
            
            # Shuffle remaining pairs
            remaining_pairs = [p for p in pair_pool if p not in pairs_epoch]
            random.shuffle(remaining_pairs)
            
            for pair in remaining_pairs:
                _, idx = pair
                
                if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                    idx_batch.add(idx)
                    current_batch.append(pair)
                    pairs_epoch.add(pair)
                    
                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                        near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])
                        near_always = copy.deepcopy(near_similarity[:neighbour_split])
                        near_random = copy.deepcopy(near_similarity[neighbour_split:])
                        random.shuffle(near_random)
                        near_random = near_random[:neighbour_split]
                        near_similarity_select = near_always + near_random
                        
                        for idx_near in near_similarity_select:
                            if len(current_batch) >= self.shuffle_batch_size:
                                break
                            
                            if idx_near not in idx_batch:
                                near_pairs = copy.deepcopy(idx2pair_pool[idx_near])
                                random.shuffle(near_pairs)
                                
                                for near_pair in near_pairs:
                                    if near_pair not in pairs_epoch:
                                        idx_batch.add(idx_near)
                                        current_batch.append(near_pair)
                                        pairs_epoch.add(near_pair)
                                        break
                
                if len(current_batch) >= self.shuffle_batch_size:
                    new_batches.extend(current_batch)
                    current_batch = []
                    idx_batch = set()
            
            # Add any remaining pairs in current_batch
            if current_batch:
                new_batches.extend(current_batch)
            
            # Update batches with new samples
            batches.extend(new_batches)
            
            print(f"Added {len(new_batches)} new samples. Total samples now: {len(batches)}")
        
        self.samples = batches
        print("Pair pool size:", len(pair_pool))
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))
        print("Break Counter:", break_counter)
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))

class VigorDatasetEval(Dataset):
    """
    VIGOR dataset evaluation class for ground-to-satellite image matching.
    
    Args:
        data_folder (str): Path to the data storage folder
        split (str): Dataset split ('train' or 'test')
        img_type (str): Type of image ('query' or 'reference')
        same_area (bool): Whether to use data from the same area
        transforms (callable, optional): Transformations for images
        transforms_depth (callable, optional): Transformations for depth maps
    """
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 same_area=True,
                 transforms=None,
                 transforms_depth=None
                 ):
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        self.transforms_depth = transforms_depth
            
        if same_area:
            self.cities = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']
        else:
            if split == "train":
                self.cities = ['NewYork', 'Seattle'] 
            else:
                self.cities = ['Chicago', 'SanFrancisco'] 
               
        # Load satellite image list
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, sep=r'\s+')
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/satellite/{x.sat}', axis=1).str.replace(".png", ".npy")
            df_tmp["path"] = df_tmp['path'].str.replace("VIGOR", "VIGOR_npy", regex=False)
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # Create index mappings for complete train and test sets
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        # Load ground view data based on mode
        ground_list = []
        for city in self.cities:
            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_{split}.txt', header=None, sep=r'\s+')
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, sep=r'\s+')
  
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0: "ground",
                                                                     1: "sat",
                                                                     4: "sat_np1",
                                                                     7: "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["ground_npy"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/panorama/{x.ground}', axis=1).str.replace(".jpg", ".npy")
            df_tmp["ground_npy"] = df_tmp['ground_npy'].str.replace("VIGOR", "VIGOR_npy", regex=False)
            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)

            ground_list.append(df_tmp) 
            
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
        
        # Create index mappings for split
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.ground_npy))
        
        if self.img_type == "reference":
            if split == "train":
                # Only satellite images used in training
                self.label = self.df_ground["sat"].unique()
                self.images = []
                for idx in self.label:
                    self.images.append(self.idx2sat_path[idx])
            else:
                # All satellite images of cities in split
                self.images = self.df_sat["path"].values
                self.label = self.df_sat.index.values
            
        elif self.img_type == "query":
            self.images = self.df_ground["ground_npy"].values
            self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")

    def __getitem__(self, index):
        """
        Get a data sample at the specified index.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: (img, label)
                - img: Image data (either ground or satellite view)
                - label: Ground truth label
        """
        if self.img_type == "reference":
            img_path = self.images[index]
            label = self.label[index]
            
            img = np.load(img_path, allow_pickle=True)
            
            # Apply image transformations
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
                
            label = torch.tensor(label, dtype=torch.long)

            return img, label
        else:
            img_path = self.images[index]
            label = self.label[index]
            
            img_np = np.load(img_path, allow_pickle=True)
            
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
                
            label = torch.tensor(label, dtype=torch.long)
            
            return img, label
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images)

            





