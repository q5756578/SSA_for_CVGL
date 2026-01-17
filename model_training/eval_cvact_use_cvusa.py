"""
This script evaluates the VIGOR (Visual Geo-localization) model on the CVACT dataset.
It performs evaluation on both validation and test sets, computing recall@k metrics
and optionally saving visualization results of top matches.
"""

import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch.serialization
from numpy.core.multiarray import scalar
import pickle
import einops
import numpy as np
import cv2
from sample4geo.dataset.cvact_v2 import CVACTDatasetTrain, CVACTDatasetEval, CVACTDatasetTest
from sample4geo.transforms import get_transforms_val
from sample4geo.evaluate.cvusa_and_cvact import evaluate, save_top_matches_cvact
from sample4geo.model import TimmModel, TimmModel_v5 

# Add safe global variables for model loading
torch.serialization.add_safe_globals([scalar])

'''
This file is intended to perform training on the CVACT dataset after training on the CVUSA dataset.  

'''



@dataclass
class Configuration:
    """
    Configuration class for model evaluation.
    
    This class defines all parameters needed for model evaluation, including:
    - Model architecture and pretrained weights
    - Evaluation settings (batch size, GPU usage)
    - Dataset paths and parameters
    - Checkpoint loading settings
    """
    
    # Model configuration
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'  # Base model architecture
    img_size: int = 384  # Input image size
    
    # Evaluation settings
    batch_size: int = 128  # Batch size for evaluation
    verbose: bool = True  # Whether to print detailed progress
    gpu_ids: tuple = (0,)  # GPU IDs to use for evaluation
    normalize_features: bool = True  # Whether to normalize features
    
    # Dataset configuration
    data_folder = "/ssd-data/jshen-data"  # Root directory for dataset
    
    # Checkpoint configuration
    checkpoint_start = '/home/jshen/Sample4Geo-my_DP/cvusa/convnext_base.fb_in22k_ft_in1k_384/0508_141030/weights_e96_98.5367.pth'  # Path to pretrained weights
    
    # System configuration
    num_workers: int = 0 if os.name == 'nt' else 4  # Number of data loading workers (0 for Windows)
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use for evaluation

# Initialize configuration
config = Configuration()

if __name__ == '__main__':
    #-----------------------------------------------------------------------------#
    # Model Setup                                                                 #
    #-----------------------------------------------------------------------------#
    
    print("\nModel: {}".format(config.model))

    # Initialize model with specified architecture
    model = TimmModel_v5(config.model,
                        pretrained=True,
                        img_size=config.img_size,
                        device=config.device,
                        classes=2)  # Binary classification
    
    # Get model configuration for image preprocessing
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    
    # Set image sizes for satellite and ground images
    image_size_sat = (img_size, img_size)
    new_width = config.img_size * 2    
    new_hight = round((224 / 1232) * new_width)
    img_size_ground = (new_hight, new_width)
     
    # Load pretrained weights
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        try:
            # First attempt: load without weights_only flag
            model_state_dict = torch.load(config.checkpoint_start, weights_only=False)
        except Exception as e:
            print(f"First attempt failed: {e}")
            try:
                # Second attempt: use safe_globals context
                with torch.serialization.safe_globals([scalar]):
                    model_state_dict = torch.load(config.checkpoint_start)
            except Exception as e:
                print(f"Second attempt failed: {e}")
                # Final attempt: use pickle_module
                model_state_dict = torch.load(config.checkpoint_start, pickle_module=pickle)
        
        model.load_state_dict(model_state_dict, strict=False)     

    # Setup data parallel if multiple GPUs available
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Move model to specified device
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 

    #-----------------------------------------------------------------------------#
    # Data Transforms                                                             #
    #-----------------------------------------------------------------------------#

    # Get validation transforms for satellite and ground images
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std)

    #-----------------------------------------------------------------------------#
    # Validation Dataset Setup                                                    #
    #-----------------------------------------------------------------------------#

    # Setup reference (satellite) dataset for validation
    reference_dataset_val = CVACTDatasetEval(data_folder=config.data_folder,
                                            split="val",
                                            img_type="reference",
                                            transforms=sat_transforms_val)
    
    reference_dataloader_val = DataLoader(reference_dataset_val,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)
    
    # Setup query (ground) dataset for validation
    query_dataset_val = CVACTDatasetEval(data_folder=config.data_folder,
                                        split="val",
                                        img_type="query",    
                                        transforms=ground_transforms_val)
    
    query_dataloader_val = DataLoader(query_dataset_val,
                                     batch_size=config.batch_size,
                                     num_workers=config.num_workers,
                                     shuffle=False,
                                     pin_memory=True)
    
    print("Reference Images Val:", len(reference_dataset_val))
    print("Query Images Val:", len(query_dataset_val))
    
    #-----------------------------------------------------------------------------#
    # Validation Evaluation                                                       #
    #-----------------------------------------------------------------------------#
    
    print("\n{}[{}]{}".format(30*"-", "CVACT_VAL", 30*"-"))  
    
    # Evaluate on validation set
    r1_test = evaluate(config=config,
                      model=model,
                      reference_dataloader=reference_dataloader_val,
                      query_dataloader=query_dataloader_val, 
                      ranks=[1, 5, 10],
                      step_size=1000,
                      cleanup=True)
        
    #-----------------------------------------------------------------------------#
    # Test Dataset Setup                                                          #
    #-----------------------------------------------------------------------------#
    
    # Setup reference (satellite) dataset for test
    reference_dataset_test = CVACTDatasetTest(data_folder=config.data_folder,
                                             img_type="reference",
                                             transforms=sat_transforms_val)
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                          batch_size=config.batch_size,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True)
    
    # Setup query (ground) dataset for test
    query_dataset_test = CVACTDatasetTest(data_folder=config.data_folder,
                                         img_type="query",    
                                         transforms=ground_transforms_val)
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      shuffle=False,
                                      pin_memory=True)
    
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))          

    #-----------------------------------------------------------------------------#
    # Test Evaluation                                                             #
    #-----------------------------------------------------------------------------#
    
    print("\n{}[{}]{}".format(30*"-", "CVACT_TEST", 30*"-"))  

    # Save visualization results for test set
    save_top_matches_cvact(
        config=config,
        model=model,
        reference_dataloader=reference_dataloader_test,
        query_dataloader=query_dataloader_test,
        num_queries=100,
        top_n=5,
        output_dir="cvact_top_matches_test",
        find_matching_images=True,
        matching_source_folder=['sat_BEV_label']
    )
    
    # Evaluate on test set
    r1_test = evaluate(config=config,
                      model=model,
                      reference_dataloader=reference_dataloader_test,
                      query_dataloader=query_dataloader_test, 
                      ranks=[1, 5, 10],
                      step_size=1000,
                      cleanup=True)
