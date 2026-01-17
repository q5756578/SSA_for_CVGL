import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
from tqdm import tqdm
import time
 
class CVUSADatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 transforms_bev =  None,     # 对卫星视角bev标签图像的变换
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 shuffle_batch_size=128,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.shuffle_batch_size = shuffle_batch_size
        
        self.transforms_query = transforms_query           # ground
        self.transforms_reference = transforms_reference   # satellite
        self.transforms_bev  = transforms_bev
        self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        
        self.df = self.df.rename(columns={0: "sat", 1: "ground", 2: "ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))
        
        #self.df['BEV_label'] = self.df.sat.str.replace('bingmap','sat_BEV_label')
        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
        
        self.pairs = list(zip(self.df.idx, self.df.sat, self.df.ground))
        
        self.idx2pair = dict()
        train_ids_list = list()
        
        # for shuffle pool
        for pair in self.pairs:
            idx = pair[0]
            self.idx2pair[idx] = pair
            train_ids_list.append(idx)
            
        self.train_ids = train_ids_list
        self.samples = copy.deepcopy(self.train_ids)
            

    def __getitem__(self, index):
        
        idx, sat, ground = self.idx2pair[self.samples[index]]
        bev_label = sat.replace('bingmap','sat_BEV_label')
        #print(bev_label)
        # load query -> ground image
        query_img = cv2.imread(f'{self.data_folder}/{ground}')
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        reference_img = cv2.imread(f'{self.data_folder}/{sat}')
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        reference_bev  = cv2.imread(f'{self.data_folder}/{bev_label}',cv2.IMREAD_GRAYSCALE)
        reference_bev = np.expand_dims(reference_bev,axis=-1)
        #print(reference_bev.shape,'asdasdasdasdasdasd')
        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1) 
            reference_bev = cv2.flip(reference_bev,1)
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']
        
        if self.transforms_bev is not None:
            reference_bev = self.transforms_bev(image=reference_bev)['image']
            reference_bev = reference_bev // 255
        # Rotate simultaneously query and reference
     
        if np.random.random() < self.prob_rotate:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2)) 
            reference_bev = torch.rot90(reference_bev,k=r,dims=(1,2))       
            # use roll for ground view if rotate sat view
            c, h, w = query_img.shape
            shifts = - w//4 * r 
            query_img = torch.roll(query_img, shifts=shifts, dims=2)  
        
            
        label = torch.tensor(idx, dtype=torch.long)  
        reference_bev = reference_bev.long()
        return query_img, reference_img,reference_bev, label
    
    def __len__(self):
        return len(self.samples)
        
        
            
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

            
       
class CVUSADatasetEval(Dataset):
    
    def __init__(self,
                 data_folder,
                 split,
                 img_type,
                 transforms=None,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.split = split
        self.img_type = img_type
        self.transforms = transforms
        
        if split == 'train':
            self.df = pd.read_csv(f'{data_folder}/splits/train-19zl.csv', header=None)
        else:
            self.df = pd.read_csv(f'{data_folder}/splits/val-19zl.csv', header=None)
        
        self.df = self.df.rename(columns={0:"sat", 1:"ground", 2:"ground_anno"})
        
        self.df["idx"] = self.df.sat.map(lambda x : int(x.split("/")[-1].split(".")[0]))

        self.idx2sat = dict(zip(self.df.idx, self.df.sat))
        self.idx2ground = dict(zip(self.df.idx, self.df.ground))
   
    
        if self.img_type == "reference":
            self.images = self.df.sat.values
            self.label = self.df.idx.values
            
        elif self.img_type == "query":
            self.images = self.df.ground.values
            self.label = self.df.idx.values 
        else:
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")
                

    def __getitem__(self, index):
        
        img = cv2.imread(f'{self.data_folder}/{self.images[index]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.images)

            





