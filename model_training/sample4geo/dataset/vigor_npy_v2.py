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
# Original version of dataset without additional data caching mechanism, using VIGORv2 labels for training.




class VigorDatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,                # Path to data storage folder
                 same_area=True,             # Whether to use data from the same area
                 transforms_query=None,      # Transformations for ground view images
                 transforms_reference=None,  # Transformations for satellite view images
                 transforms_depth= None,     # Transformations for ground view depth maps
                 transforms_bev =  None,     # Transformations for satellite view BEV label images
                 prob_flip=0.0,              # Probability of horizontal flip
                 prob_rotate=0.0,            # Probability of rotation
                 shuffle_batch_size=128,     # Batch size after shuffling
                 ):
        super().__init__()

        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        self.transforms_query = transforms_query
        self.transforms_reference = transforms_reference
        self.transforms_depth = transforms_depth
        self.transforms_bev  = transforms_bev
        # Load different city lists based on whether using same area
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
            
            df_tmp["path"]  = df_tmp.apply(lambda x: f'{data_folder}/{city}/sat_npy/{x.sat}', axis=1).str.replace(".png", ".npy") #png
            
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)

        # Create satellite image index to path mapping
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))  # Map satellite image names to indices
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))  # Map indices to satellite image names
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))  # Map indices to satellite image paths

        # Load ground view image data
        ground_list = []
        for city in self.cities:
            # Select different data files based on `same_area` flag
            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_train.txt', header=None,sep=r'\s+')
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, sep=r'\s+')
            
            # Rename columns and add path column                                
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0: "ground",
                                                                     1: "sat",
                                                                     4: "sat_np1",
                                                                     7: "sat_np2",
                                                                     10: "sat_np3"})
            df_tmp["ground_npy"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/ground_npy/{x.ground}', axis=1).str.replace(".jpg", ".npy") #jpg
            
            df_tmp['BEV_label'] = df_tmp.apply(lambda x: f'{data_folder}/{city}/map_repro/{x.sat}', axis=1).str.replace(".png", ".npy") 
            
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)
                
            
            ground_list.append(df_tmp)

        
        
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)

        # Create ground data index to path mapping
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_npy_path = dict(zip(self.df_ground.index, self.df_ground.ground_npy))
        self.idx2bev_label =  dict(zip(self.df_ground.index, self.df_ground.BEV_label))
        # Create ground image and satellite image pairs
        self.pairs = list(zip(self.df_ground.index, self.df_ground.sat))
        self.idx2pairs = defaultdict(list)
        for pair in self.pairs:
            self.idx2pairs[pair[1]].append(pair)

        # Extract satellite image indices related to ground images
    
        #self.label = self.df_ground["sat"].values # Only one-to-one correspondence sat_ground
        self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 
        
        # Deep copy pair list for shuffling
        self.samples = copy.deepcopy(self.pairs)

    def __getitem__(self, index):
        # Get ground and satellite image pair based on index
        idx_ground, idx_sat = self.samples[index]
        
        # Load ground view image and convert to RGB
        query_npy = np.load(self.idx2ground_npy_path[idx_ground],allow_pickle=True).item()
    
        query_img = query_npy['panorama']
        
        '''
        panorama   'depth_map_normal' 
        "satellite":
        "sat_label01"
        '''
       
        #que_depth  = query_npy['depth_map_normal']
        #que_depth = np.expand_dims(que_depth, axis=-1)
        # Load satellite view image and convert to RGB
        reference_npy =  np.load(self.idx2sat_path[idx_sat],allow_pickle=True).item()
        reference_img = reference_npy['satellite']
        #print('gouen',self.idx2ground_npy_path[idx_ground])
        #print('bev_lab',self.idx2bev_label[idx_ground])
        #que_bevlab   =  cv2.imread(self.idx2bev_label[idx_ground],cv2.COLOR_BGR2RGB)
        
        #que_bevlab  = reference_npy['sat_label01']
        
        que_bevlab =  np.load(self.idx2bev_label[idx_ground],allow_pickle=True) #VIGORv2 dataset labels
        que_bevlab = np.expand_dims(que_bevlab, axis=-1)
        
        #if self.layout_mode == 'RGB':
        #    pass
        #elif self.layout_mode == 'discrete':
        #    que_bevlab = self.decode_layoutv2(que_bevlab)
        #else:
        #    raise RuntimeError(f"layout mode {self.layout_mode} invalid please choose [RGB, discrete, one_hot]")
        
   
        # Randomly flip ground and satellite images horizontally
        if np.random.random() < self.prob_flip:
            
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1) 
            #que_depth   =  cv2.flip(que_depth,1)
            que_bevlab  =  cv2.flip(que_bevlab,1)
        # Apply predefined image transformations
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']
        #if self.transforms_depth is not None:
        #    que_depth  =  self.transforms_depth(image=que_depth)['image']
        if self.transforms_bev is not None:
            que_bevlab  =  self.transforms_bev(image=que_bevlab)['image']
            #que_bevlab  = que_bevlab // 255
            
        # Randomly rotate ground and satellite images simultaneously
        if np.random.random() < self.prob_rotate:
            r = np.random.choice([1, 2, 3])
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2))  # Rotate satellite image
            que_bevlab = torch.rot90(que_bevlab, k=r, dims=(1, 2))   # Rotate BEV label
            c, h, w = query_img.shape
            shifts = - w // 4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)  # Roll ground image to simulate rotation
            #que_depth = torch.roll(que_depth, shifts=shifts, dims=2)  # Roll ground image, depth image naturally needs to roll too
        
        label = torch.tensor(idx_sat, dtype=torch.long)  # Label is satellite image index
        que_bevlab = que_bevlab.long()
        
     
        return query_img, reference_img,que_bevlab, label
    
    def __len__(self):
        return len(self.samples)
    
    def decode_layoutv2(self, image, layout_mode='discrete'):
        """
        Decode RGB image to label image using efficient NumPy operations
        Args:
            image: Input RGB image with shape (height, width, channels)
            layout_mode: Output mode, 'discrete' or 'one_hot'
        Returns:
            Decoded label image
        """
        color_dict = {
            (255., 178., 102.): 0,  # Building
            (64., 90., 255.): 1,   # parking
            (102., 255., 102.): 2,    # grass park playground
            (0., 153., 0.): 3,      # forest
            (204., 229., 255.): 4,    # water
            (192., 192., 192.): 5,    # path
            (96., 96., 96.): 6,       # road
            (255., 255., 255.): 7     # background
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
        # Use broadcasting and vectorized operations to compute distances
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

    
    def decode_layout(self, image):
        color_dict = {
            (255., 178., 102.): 0,  #Building
            (64., 90., 255.): 255,    #parking
            (102., 255., 102.): 0,  #grass park playground
            (0., 153., 0.): 255,      #forest
            (204., 229., 255.): 0,  #water
            (192., 192., 192.): 0,  #path
            (96., 96., 96.): 0,     #road
            (255., 255., 255.): 0   #background
        }

        classes = torch.Tensor(list(color_dict.values()))
        color_to_class_tensor = torch.Tensor(list(color_dict.keys()))/255

        #image = einops.rearrange(image, 'c h w -> h w c')
        indices = torch.argmin(torch.sum(torch.abs(image.unsqueeze(dim=-2) - color_to_class_tensor), dim=-1), dim=-1)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=len(classes))
        #one_hot = einops.rearrange(one_hot, 'h w c -> c h w')
        
        if self.layout_mode == 'discrete':
            one_hot = torch.argmax(one_hot, dim=0)
            return one_hot.cpu().numpy()
        elif self.layout_mode == 'one_hot':
            return one_hot

    
    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        
        # Custom shuffle function for unique class_id sampling in batch
        
        
        print("\nShuffle Dataset:")  # Output message indicating dataset is being shuffled
        
        pair_pool = copy.deepcopy(self.pairs)  # Create deep copy of self.pairs
        idx2pair_pool = copy.deepcopy(self.idx2pairs)  # Create deep copy of self.idx2pairs
        
        neighbour_split = neighbour_select // 2  # Calculate half of neighbor selection
        
        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)  # If sim_dict provided, create deep copy
        
        # Shuffle pairs order
        random.shuffle(pair_pool)  # Randomly shuffle pair_pool elements
        
        # Lookup if already used in epoch
        pairs_epoch = set()  # Create set to track pairs used in current epoch
        idx_batch = set()  # Create set to track current batch indices
        
        # buckets
        batches = []  # Create list to store all batch pairs
        current_batch = []  # Create list to store current batch pairs
        
        # counter
        break_counter = 0  # Create counter to track failed pair additions
        
        # progressbar
        pbar = tqdm()  # Initialize progress bar

        while True:  # Start infinite loop until manually interrupted or conditions met
            
            pbar.update()  # Update progress bar each iteration
            
            if len(pair_pool) > 0:  # Check if pair pool has pairs
                pair = pair_pool.pop(0)  # Pop first pair
                
                _, idx = pair  # Extract index from pair
                
                if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                    # Check if current index not in idx_batch, pair not in pairs_epoch,
                    # and current batch length less than self.shuffle_batch_size
                    
                    idx_batch.add(idx)  # Add index to idx_batch
                    current_batch.append(pair)  # Add current pair to current_batch
                    pairs_epoch.add(pair)  # Add pair to pairs_epoch
                    
                    # remove from pool used for sim-sampling
                    idx2pair_pool[idx].remove(pair)  # Remove current pair from idx2pair_pool
                    
                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                        # If sim_dict provided and current batch length less than self.shuffle_batch_size, do similarity sampling
                        
                        near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])  # Extract similarities
                        near_always = copy.deepcopy(near_similarity[:neighbour_split])  # Always selected neighbors
                        near_random = copy.deepcopy(near_similarity[neighbour_split:])  # Randomly selected neighbors
                        random.shuffle(near_random)  # Randomly shuffle random neighbors
                        near_random = near_random[:neighbour_split]  # Select half of random neighbors
                        near_similarity_select = near_always + near_random  # Combine both neighbor sets
                        
                        for idx_near in near_similarity_select:  # Iterate through each neighbor index
                            
                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:  # Check current batch length
                                break  # Exit if batch size reached
                            
                            # no check for pair in epoch necessary cause all we add is removed from pool
                            if idx_near not in idx_batch:  # Check if neighbor index not in idx_batch
                                
                                near_pairs = copy.deepcopy(idx2pair_pool[idx_near])  # Get neighbor pairs
                                
                                # up to 2 for one sat view 
                                random.shuffle(near_pairs)  # Randomly shuffle neighbor pairs
                                
                                for near_pair in near_pairs:  # Iterate through neighbor pairs
                                    
                                    idx_batch.add(idx_near)  # Add neighbor index to idx_batch
                                    current_batch.append(near_pair)  # Add neighbor pair to current_batch
                                    pairs_epoch.add(near_pair)  # Add neighbor pair to pairs_epoch
                                    
                                    idx2pair_pool[idx_near].remove(near_pair)  # Remove from pool
                                    similarity_pool[idx].remove(idx_near)  # Remove from similarity pool
                                    
                                    # only select one view
                                    break  # Exit after selecting one view
                            
                        break_counter = 0  # Reset counter
                        
                    else:  # If pair not added to batch and not in pairs_epoch
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)  # Put pair back in pair_pool
                    
                    break_counter += 1  # Increment counter
                    
                    if break_counter >= 1024:  # Check counter
                        break  # Exit if limit reached
                
                else:  # If pair_pool empty, exit loop
                    break

                if len(current_batch) >= self.shuffle_batch_size:  # Check current batch
                    
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)  # Add current_batch pairs to batches
                    idx_batch = set()  # Reset idx_batch
                    current_batch = []  # Reset current_batch
        
        pbar.close()  # Close progress bar
        
        # wait before closing progress bar
        time.sleep(0.3)  # Wait 0.3 seconds to ensure user sees final progress
        
        self.samples = batches  # Assign final pair list to self.samples
        print("pair_pool:", len(pair_pool))  # Output pair pool length
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))  # Output length comparison
        print("Break Counter:", break_counter)  # Output failed pair count
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))  # Output unadded pairs
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))  # Output first and last element IDs
    
    """
    def shuffle(self, sim_dict=None, neighbour_select=8, neighbour_range=16):
        '''
        custom shuffle function for unique class_id sampling in batch
        '''
        
        print("\nShuffle Dataset:")  # Output message indicating dataset is being shuffled
        
        pair_pool = copy.deepcopy(self.pairs)  # Create deep copy of self.pairs
        idx2pair_pool = copy.deepcopy(self.idx2pairs)  # Create deep copy of self.idx2pairs
        
        neighbour_split = neighbour_select // 2  # Calculate half of neighbor selection
        
        if sim_dict is not None:
            similarity_pool = copy.deepcopy(sim_dict)  # If sim_dict provided, create deep copy
        
        # Shuffle pairs order
        random.shuffle(pair_pool)  # Randomly shuffle pair_pool elements
        
        # Lookup if already used in epoch
        pairs_epoch = set()  # Create set to track pairs used in current epoch
        idx_batch = set()  # Create set to track current batch indices
        
        # buckets
        batches = []  # Create list to store all batch pairs
        current_batch = []  # Create list to store current batch pairs
        
        # counter
        break_counter = 0  # Create counter to track failed pair additions
        
        # progressbar
        pbar = tqdm()  # Initialize progress bar

        while True:  # Start infinite loop until manually interrupted or conditions met
            
            pbar.update()  # Update progress bar each iteration
            
            if len(pair_pool) > 0:  # Check if pair pool has pairs
                pair = pair_pool.pop(0)  # Pop first pair
                
                _, idx = pair  # Extract index from pair
                
                if idx not in idx_batch and pair not in pairs_epoch and len(current_batch) < self.shuffle_batch_size:
                    # Check if current index not in idx_batch, pair not in pairs_epoch,
                    # and current batch length less than self.shuffle_batch_size
                    
                    idx_batch.add(idx)  # Add index to idx_batch
                    current_batch.append(pair)  # Add current pair to current_batch
                    pairs_epoch.add(pair)  # Add pair to pairs_epoch
                    
                    # remove from pool used for sim-sampling
                    idx2pair_pool[idx].remove(pair)  # Remove current pair from idx2pair_pool
                    
                    if sim_dict is not None and len(current_batch) < self.shuffle_batch_size:
                        # If sim_dict provided and current batch length less than self.shuffle_batch_size, do similarity sampling
                        
                        near_similarity = copy.deepcopy(similarity_pool[idx][:neighbour_range])  # Extract similarities
                        near_always = copy.deepcopy(near_similarity[:neighbour_split])  # Always selected neighbors
                        near_random = copy.deepcopy(near_similarity[neighbour_split:])  # Randomly selected neighbors
                        random.shuffle(near_random)  # Randomly shuffle random neighbors
                        near_random = near_random[:neighbour_split]  # Select half of random neighbors
                        near_similarity_select = near_always + near_random  # Combine both neighbor sets
                        
                        for idx_near in near_similarity_select:  # Iterate through each neighbor index
                            
                            # check for space in batch
                            if len(current_batch) >= self.shuffle_batch_size:  # Check current batch length
                                break  # Exit if batch size reached
                            
                            # no check for pair in epoch necessary cause all we add is removed from pool
                            if idx_near not in idx_batch:  # Check if neighbor index not in idx_batch
                                
                                near_pairs = copy.deepcopy(idx2pair_pool[idx_near])  # Get neighbor pairs
                                
                                # up to 2 for one sat view 
                                random.shuffle(near_pairs)  # Randomly shuffle neighbor pairs
                                
                                for near_pair in near_pairs:  # Iterate through neighbor pairs
                                    
                                    idx_batch.add(idx_near)  # Add neighbor index to idx_batch
                                    current_batch.append(near_pair)  # Add neighbor pair to current_batch
                                    pairs_epoch.add(near_pair)  # Add neighbor pair to pairs_epoch
                                    
                                    idx2pair_pool[idx_near].remove(near_pair)  # Remove from pool
                                    similarity_pool[idx].remove(idx_near)  # Remove from similarity pool
                                    
                                    # only select one view
                                    break  # Exit after selecting one view
                            
                        break_counter = 0  # Reset counter
                        
                    else:  # If pair not added to batch and not in pairs_epoch
                        if pair not in pairs_epoch:
                            pair_pool.append(pair)  # Put pair back in pair_pool
                    
                    break_counter += 1  # Increment counter
                    
                    if break_counter >= 1024:  # Check counter
                        break  # Exit if limit reached
                
                else:  # If pair_pool empty, exit loop
                    break

                if len(current_batch) >= self.shuffle_batch_size:  # Check current batch
                    
                    # empty current_batch bucket to batches
                    batches.extend(current_batch)  # Add current_batch pairs to batches
                    idx_batch = set()  # Reset idx_batch
                    current_batch = []  # Reset current_batch
        
        pbar.close()  # Close progress bar
        
        # wait before closing progress bar
        time.sleep(0.3)  # Wait 0.3 seconds to ensure user sees final progress
        
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
        
        self.samples = batches  # Assign final pair list to self.samples
        print("pair_pool:", len(pair_pool))  # Output pair pool length
        print("Original Length: {} - Length after Shuffle: {}".format(len(self.pairs), len(self.samples)))  # Output length comparison
        print("Break Counter:", break_counter)  # Output failed pair count
        print("Pairs left out of last batch to avoid creating noise:", len(self.pairs) - len(self.samples))  # Output unadded pairs
        print("First Element ID: {} - Last Element ID: {}".format(self.samples[0][1], self.samples[-1][1]))  # Output first and last element IDs
    """



       
class VigorDatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 same_area=True,
                 transforms=None,
                 transforms_depth = None
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
               
        # load sat list 
        sat_list = []
        for city in self.cities:
            df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/satellite_list.txt', header=None, sep=r'\s+')
            df_tmp = df_tmp.rename(columns={0: "sat"})
            df_tmp["path"]  = df_tmp.apply(lambda x: f'{data_folder}/{city}/sat_npy/{x.sat}', axis=1).str.replace(".png", ".npy") #png
            sat_list.append(df_tmp)
        self.df_sat = pd.concat(sat_list, axis=0).reset_index(drop=True)
        
        # idx for complete train and test independent of mode = train or test
        sat2idx = dict(zip(self.df_sat.sat, self.df_sat.index))
        self.idx2sat = dict(zip(self.df_sat.index, self.df_sat.sat))
        self.idx2sat_path = dict(zip(self.df_sat.index, self.df_sat.path))
        
        
        # ground dependent on mode 'train' or 'test'
        ground_list = []
        for city in self.cities:
            
            if same_area:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/same_area_balanced_{split}.txt', header=None, sep=r'\s+')
            else:
                df_tmp = pd.read_csv(f'{data_folder}/splits/{city}/pano_label_balanced.txt', header=None, sep=r'\s+')
  
            
            df_tmp = df_tmp.loc[:, [0, 1, 4, 7, 10]].rename(columns={0:  "ground",
                                                                     1:  "sat",
                                                                     4:  "sat_np1",
                                                                     7:  "sat_np2",
                                                                     10: "sat_np3"})
            
            df_tmp["ground_npy"] = df_tmp.apply(lambda x: f'{data_folder}/{city}/ground_npy/{x.ground}', axis=1).str.replace(".jpg", ".npy") #jpg
            for sat_n in ["sat", "sat_np1", "sat_np2", "sat_np3"]:
                df_tmp[f'{sat_n}'] = df_tmp[f'{sat_n}'].map(sat2idx)

            ground_list.append(df_tmp) 
            
        self.df_ground = pd.concat(ground_list, axis=0).reset_index(drop=True)
        #self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values 
        # idx for split train or test dependent on mode = train or test
        self.idx2ground = dict(zip(self.df_ground.index, self.df_ground.ground))
        self.idx2ground_path = dict(zip(self.df_ground.index, self.df_ground.ground_npy))
 
        '''
        panorama   'depth_map_normal' 
        "satellite":
        "sat_label01"
        '''
        
        if self.img_type == "reference":
            if split == "train":
                # only sat images we really train on
                self.label = self.df_ground["sat"].unique()
                self.images = []
                for idx in self.label:
                    self.images.append(self.idx2sat_path[idx])
            else:
                # all sat images of cities in split
                self.images = self.df_sat["path"].values
                self.label = self.df_sat.index.values
            
        elif self.img_type == "query":
            self.images = self.df_ground["ground_npy"].values
            #self.label = self.df_ground["sat"].values 
            self.label = self.df_ground[["sat", "sat_np1", "sat_np2", "sat_np3"]].values
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
                

    def __getitem__(self, index):
        if self.img_type == "reference":
            img_path = self.images[index]
            label = self.label[index]
            
            img  = np.load(img_path,allow_pickle=True).item()
            img =  img['satellite']
            
            # image transforms
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
                
            label = torch.tensor(label, dtype=torch.long)

            return img, label
        else:
            img_path = self.images[index]
            label = self.label[index]
            
            img_np = np.load(img_path,allow_pickle=True).item()
            img = img_np['panorama']
            
            que_depth = img_np['depth_map_normal']
    
            que_depth = np.expand_dims(que_depth, axis=-1)
            

            if self.transforms is not None:
                img = self.transforms(image=img)['image']
                depth = self.transforms(image=que_depth)['image']
            
            #img = torch.cat([img,depth],dim=0)
                
            label = torch.tensor(label, dtype=torch.long)
            
            return img,label
    
   
        
    
    def __len__(self):
        return len(self.images)

            





