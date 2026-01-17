import cv2
import numpy as np
from torch.utils.data import Dataset
import random
import copy
import torch
from tqdm import tqdm
import time
import scipy.io as sio
import os
from glob import glob

class CVACTDatasetTrain(Dataset):
    
    def __init__(self,
                 data_folder,
                 transforms_query=None,
                 transforms_reference=None,
                 transforms_depth= None,     # 对地面视角深度图的变换
                 transforms_bev =  None,  
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
        
        self.transforms_depth = transforms_depth
        self.transforms_bev  = transforms_bev
        
        anuData = sio.loadmat(os.path.join(self.data_folder,'CVACT_new','ACT_data.mat')) 
        
        ids = anuData['panoIds']

        train_ids = ids[anuData['trainSet'][0][0][1]-1]
        
        train_ids_list = []
        train_idsnum_list = []
        self.idx2numidx = dict()
        self.numidx2idx = dict()
        self.idx_ignor = set()
        i = 0

        for idx in train_ids.squeeze():
            
            idx = str(idx)
            grd_path = os.path.join(self.data_folder,'CVACT_new','streetview',f'{idx}_grdView.jpg')
            sat_path = os.path.join(self.data_folder,'CVACT_new','satview_polish',f'{idx}_satView_polish.jpg') 
            bev_path = os.path.join(self.data_folder,'CVACT_new','sat_BEV_label', f'{idx}_satView_polish.jpg')
            #grd_path = f'ANU_data_small/streetview/{idx}_grdView.jpg'
            #sat_path = f'ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
            #bev_path = f'ANU_data_small/sat_bevlabel/{idx}_satView_polish.jpg'
            if not os.path.exists(grd_path) or not os.path.exists(sat_path):
                self.idx_ignor.add(idx)
            else:
                self.idx2numidx[idx] = i
                self.numidx2idx[i] = idx
                train_ids_list.append(idx)
                train_idsnum_list.append(i)
                i+=1
        
        print("IDs not found in train images:", self.idx_ignor)
        
        self.train_ids = train_ids_list
        self.train_idsnum = train_idsnum_list
        self.samples = copy.deepcopy(self.train_idsnum)
            

    def __getitem__(self, index):
        
        idnum = self.samples[index]
        
        idx = self.numidx2idx[idnum]
        
        # load query -> ground image
        grd_path = os.path.join(self.data_folder,'CVACT_new','streetview',f'{idx}_grdView.jpg')
        query_img = cv2.imread(grd_path)
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
        
        # load reference -> satellite image
        sat_path = os.path.join(self.data_folder,'CVACT_new','satview_polish',f'{idx}_satView_polish.jpg')
        reference_img = cv2.imread(sat_path)
        reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)

        bev_path = os.path.join(self.data_folder,'CVACT_new','sat_BEV_label', f'{idx}_satView_polish.jpg')
        que_bevlab   = cv2.imread(bev_path,cv2.IMREAD_GRAYSCALE)
        #que_bevlab   =  cv2.imread(que_bevlab, cv2.IMREAD_GRAYSCALE)
        
        # Flip simultaneously query and reference
        if np.random.random() < self.prob_flip:
            query_img = cv2.flip(query_img, 1)
            reference_img = cv2.flip(reference_img, 1) 
            que_bevlab    = cv2.flip(que_bevlab,1)
        
        # image transforms
        if self.transforms_query is not None:
            query_img = self.transforms_query(image=query_img)['image']
            
        if self.transforms_reference is not None:
            reference_img = self.transforms_reference(image=reference_img)['image']
            
        if self.transforms_bev is not None:
            que_bevlab = self.transforms_bev(image=que_bevlab)['image']
            que_bevlab = que_bevlab // 255
        
        # Rotate simultaneously query and reference
        if np.random.random() < self.prob_rotate:
        
            r = np.random.choice([1,2,3])
            
            # rotate sat img 90 or 180 or 270
            reference_img = torch.rot90(reference_img, k=r, dims=(1, 2)) 
            que_bevlab = torch.rot90(que_bevlab, k=r, dims=(1, 2))
            # use roll for ground view if rotate sat view
            c, h, w = query_img.shape
            shifts = - w//4 * r
            query_img = torch.roll(query_img, shifts=shifts, dims=2)  
                   
            
        label = torch.tensor(idnum, dtype=torch.long)  
        que_bevlab = que_bevlab.long() 
                
        return query_img, reference_img,que_bevlab ,label
    
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

            
       
class CVACTDatasetEval(Dataset):
    
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
        
        anuData = sio.loadmat(os.path.join(self.data_folder,'CVACT_new','ACT_data.mat'))
        
        ids = anuData['panoIds']
        
        if split != "train" and split != "val":
            raise ValueError("Invalid 'split' parameter. 'split' must be 'train' or 'val'")  
            
        if img_type != 'query' and img_type != 'reference':
            raise ValueError("Invalid 'img_type' parameter. 'img_type' must be 'query' or 'reference'")


        ids = ids[anuData[f'{split}Set'][0][0][1]-1]
        
        ids_list = []
       
        self.idx2label = dict()
        self.idx_ignor = set()
        
        i = 0
        
        for idx in ids.squeeze():
            
            idx = str(idx)
            
            #grd_path = f'{self.data_folder}/ANU_data_small/streetview/{idx}_grdView.jpg'
            #sat_path = f'{self.data_folder}/ANU_data_small/satview_polish/{idx}_satView_polish.jpg'
            #bev_path = f'ANU_data_small/sat_bevlabel/{idx}_satView_polish.jpg'
            
            grd_path = os.path.join(self.data_folder,'CVACT_new','streetview',f'{idx}_grdView.jpg')
            sat_path = os.path.join(self.data_folder,'CVACT_new','satview_polish',f'{idx}_satView_polish.jpg') 
            bev_path = os.path.join(self.data_folder,'CVACT_new','sat_BEV_label', f'{idx}_satView_polish.jpg')
            
            if not os.path.exists(grd_path) or not os.path.exists(sat_path) or not os.path.exists(bev_path):
                self.idx_ignor.add(idx)
            else:
                self.idx2label[idx] = i
                ids_list.append(idx)
                i+=1
        
        #print(f"IDs not found in {split} images:", self.idx_ignor)

        self.samples = ids_list
       

    def __getitem__(self, index):
        
        idx = self.samples[index]
        
        if self.img_type == "reference":
            sat_path = os.path.join(self.data_folder,'CVACT_new','satview_polish',f'{idx}_satView_polish.jpg') 
            path = sat_path
        elif self.img_type == "query":
            grd_path = os.path.join(self.data_folder,'CVACT_new','streetview',f'{idx}_grdView.jpg')
            path = grd_path

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(self.idx2label[idx], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.samples)

            
class CVACTDatasetTest(Dataset):
    
    def __init__(self,
                 data_folder,
                 img_type,
                 transforms=None,
                 ):
        
        super().__init__()
 
        self.data_folder = data_folder
        self.img_type = img_type
        self.transforms = transforms
        
        files_sat = glob(f'{self.data_folder}/CVACT/satview_polish/*_satView_polish.jpg')
        files_ground = glob(f'{self.data_folder}/CVACT/streetview/*_grdView.jpg')
        
        sat_ids = []
        for path in files_sat:
        
            idx = path.split("/")[-1][:-19]
            sat_ids.append(idx)
        
        ground_ids = []
        for path in files_ground:
            idx = path.split("/")[-1][:-12]
            ground_ids.append(idx)  
            
        # only use intersection of sat and ground ids   
        test_ids = set(sat_ids).intersection(set(ground_ids))
        
        self.test_ids = list(test_ids)
        self.test_ids.sort()
        #self.images =  len(self.test_ids)
        self.idx2num_idx = dict()
        
        for i, idx in enumerate(self.test_ids):
            self.idx2num_idx[idx] = i


    def __getitem__(self, index):
        
        idx = self.test_ids[index]
        
        if self.img_type == "reference":
            path = f'{self.data_folder}/CVACT/satview_polish/{idx}_satView_polish.jpg' #CVACT 就是test分布， CVACT_new 则是train 和 val 
        else:
            path = f'{self.data_folder}/CVACT/streetview/{idx}_grdView.jpg'

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # image transforms
        if self.transforms is not None:
            img = self.transforms(image=img)['image']
            
        label = torch.tensor(self.idx2num_idx[idx], dtype=torch.long)

        return img, label

    def __len__(self):
        return len(self.test_ids)




