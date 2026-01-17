"""
This script evaluates the VIGOR (Visual Geo-localization) model on the VIGOR dataset
in same-area mode. It performs evaluation on the test set, computing recall@k metrics
and saving visualization results of top matches between ground and satellite images.
The script also supports depth maps and bird's eye view (BEV) outputs.
"""

import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
import torch.serialization
from numpy.core.multiarray import scalar
import pickle
from sample4geo.dataset.vigor_npy_v2 import VigorDatasetEval, VigorDatasetTrain
from sample4geo.transforms import get_transforms_train, get_transforms_val, get_transforms_depth, get_transforms_bev, get_transforms_bev_depth
#from sample4geo.evaluate.vigor import evaluate
from sample4geo.evaluate.vigor_tensor_v2 import evaluate, save_top_matches
from sample4geo.model import TimmModel, TimmModel_v5

'''
This file is used to evaluate the VIGOR model on the VIGOR dataset. 

'''

# Add safe global variables for model loading
torch.serialization.add_safe_globals([scalar])

@dataclass
class Configuration:
    """
    Configuration class for model evaluation.
    
    This class defines all parameters needed for model evaluation, including:
    - Model architecture and pretrained weights
    - Evaluation settings (batch size, GPU usage)
    - Dataset paths and parameters
    - Checkpoint loading settings
    - VIGOR-specific settings (same-area mode, ground cutting)
    """
    
    # Model configuration
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'  # Base model architecture
    img_size: int = 384  # Input image size
    
    # Evaluation settings
    batch_size: int = 128  # Batch size for evaluation
    verbose: bool = True  # Whether to print detailed progress
    gpu_ids: tuple = (0,1,2,3)  # GPU IDs to use for evaluation
    normalize_features: bool = True  # Whether to normalize features
    
    # Dataset configuration
    data_folder = "/ssd-data/jshen-data/VIGOR"  # Root directory for dataset
    same_area: bool = True  # Whether to use same-area mode (True) or cross-area mode (False)
    ground_cutting = 0  # Number of pixels to remove from top and bottom of ground images
    source_data_path = '/home/jshen/data/VIGOR'  # Path to source data for matching images
    # Checkpoint to start from
    #checkpoint_start = 'pretrained/vigor_same/convnext_base.fb_in22k_ft_in1k_384/weights_e40_0.7786.pth' 
    checkpoint_start = '/home/jshen/Sample4Geo-my_DP/vigor_same/convnext_base.fb_in22k_ft_in1k_384/05060952/weights_e32_26.1482.pth'
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4  # Number of data loading workers (0 for Windows)
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use for evaluation


#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 


if __name__ == '__main__':

    #-----------------------------------------------------------------------------#
    # Model                                                                       #
    #-----------------------------------------------------------------------------#
        
    print("\nModel: {}".format(config.model))

    model = TimmModel_v5(config.model,
                          pretrained=True,
                          img_size=config.img_size,
                          device=config.device,
                          classes=2)
    
    data_config = model.get_config()
    print(data_config)
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = config.img_size
    
    image_size_sat = (img_size, img_size)

    new_width = img_size*2    
    new_hight = int(((1024 - 2 * config.ground_cutting) / 2048) * new_width)
    img_size_ground = (new_hight, new_width)
     
    # Load pretrained Checkpoint    
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        try:
            # 首先尝试使用 weights_only=False
            model_state_dict = torch.load(config.checkpoint_start, weights_only=False)
        except Exception as e:
            print(f"First attempt failed: {e}")
            try:
                # 如果失败，尝试使用 safe_globals 上下文管理器
                with torch.serialization.safe_globals([scalar]):
                    model_state_dict = torch.load(config.checkpoint_start)
            except Exception as e:
                print(f"Second attempt failed: {e}")
                # 最后尝试使用 pickle_module
                model_state_dict = torch.load(config.checkpoint_start, pickle_module=pickle)
        
        model.load_state_dict(model_state_dict, strict=False)

    # Data parallel
    print("GPUs available:", torch.cuda.device_count())  
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids)
            
    # Model to device   
    model = model.to(config.device)

    print("\nImage Size Sat:", image_size_sat)
    print("Image Size Ground:", img_size_ground)
    print("Mean: {}".format(mean))
    print("Std:  {}\n".format(std)) 


    #-----------------------------------------------------------------------------#
    # DataLoader                                                                  #
    #-----------------------------------------------------------------------------#

    # Eval
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std,
                                                                   ground_cutting=config.ground_cutting)


    # Reference Satellite Images Test
    
    ground_transforms_depth  = get_transforms_depth(
                                    img_size_ground,
                                    ground_cutting=0)

    sat_transforms_bev   = get_transforms_bev(image_size_sat)
    sat_transforms_bev_depth = get_transforms_bev_depth(image_size_sat)
    # Reference Satellite Images Test
    reference_dataset_test = VigorDatasetEval(data_folder=config.data_folder,
                                            split="test",
                                            img_type="reference",
                                            same_area=config.same_area,  
                                            transforms=sat_transforms_val,
                                            transforms_depth=sat_transforms_bev_depth)

  

    reference_dataloader_test = DataLoader(reference_dataset_test,
                                            batch_size=config.batch_size,
                                            num_workers=config.num_workers,
                                            shuffle=False,
                                            pin_memory=True,
                                            persistent_workers=True)






    # Query Ground Images Test
    query_dataset_test = VigorDatasetEval(data_folder=config.data_folder,
                                            split="test",
                                            img_type="query",
                                            same_area=config.same_area,      
                                            transforms=ground_transforms_val,
                                            transforms_depth=ground_transforms_depth)



    query_dataloader_test = DataLoader(query_dataset_test,
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers,
                                        shuffle=False,
                                        pin_memory=True,
                                        persistent_workers=True)

  
    print("Query Images Test:", len(query_dataset_test))
    print("Reference Images Test:", len(reference_dataset_test))
    

    #-----------------------------------------------------------------------------#
    # Evaluate                                                                    #
    #-----------------------------------------------------------------------------#

    print("\n{}[{}]{}".format(30*"-", "VIGOR Same", 30*"-"))  
    '''
    r1_test = evaluate(config=config,
                       model=model,
                       reference_dataloader=reference_dataloader_test,
                       query_dataloader=query_dataloader_test, 
                       ranks=[1, 5, 10],
                       step_size=1000,
                       cleanup=True)
    '''
    source_dirs = [
    'sat_artifical_building',
    'sat_structre',
    'sat_trees',
    'sat_BEV_label01'
    ]
    
    save_top_matches(
    config=config,
    model=model,
    reference_dataloader=reference_dataloader_test,
    query_dataloader=query_dataloader_test,
    num_queries=50,
    top_n=5,
    output_dir="top_matches",
    source_dirs=source_dirs,  # 指定源文件夹列表
    ensure_gt_in_top = True,
    find_matching_images=True,
    matching_source_folder=['sat_BEV_label01','map']
    )