"""
This script implements the training pipeline for the VIGOR (Visual Geo-localization) model
on the CVACT dataset. It includes training with various sampling strategies (GPS-based,
similarity-based), mixed precision training, gradient checkpointing, and comprehensive
evaluation metrics. The script also supports TensorBoard logging and model checkpointing.
"""

import os
import time
import math
import shutil
import sys
import torch
import pickle
from dataclasses import dataclass
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from transformers import get_constant_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_cosine_schedule_with_warmup

from sample4geo.dataset.cvact_v2 import CVACTDatasetTrain, CVACTDatasetEval, CVACTDatasetTest
from sample4geo.transforms import get_transforms_train, get_transforms_val, get_transforms_depth, get_transforms_bev, get_transforms_bev_depth
from sample4geo.utils import setup_system, Logger
from sample4geo.trainer import train
from sample4geo.evaluate.cvusa_and_cvact import evaluate, calc_sim
from sample4geo.loss import InfoNCE, MultiClassDiceLoss
from sample4geo.model import TimmModel, TimmModel_v5
import datetime
from torch.utils.tensorboard import SummaryWriter 

@dataclass
class Configuration:
    """
    Configuration class for model training.
    
    This class defines all parameters needed for training, including:
    - Model architecture and pretrained weights
    - Training hyperparameters (batch size, epochs, learning rate)
    - Sampling strategies (GPS-based, similarity-based)
    - Evaluation settings
    - Optimizer and scheduler settings
    - Dataset and augmentation parameters
    """
    
    # Model configuration
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'  # Base model architecture
    img_size: int = 384  # Input image size
    
    # Training settings
    mixed_precision: bool = True  # Whether to use mixed precision training
    seed = 1  # Random seed for reproducibility
    epochs: int = 100  # Number of training epochs
    batch_size: int = 32  # Batch size (note: real batch size = 2 * batch_size)
    verbose: bool = True  # Whether to print detailed progress
    gpu_ids: tuple = (0,1,2,3)  # GPU IDs for training
    
    # Sampling strategy configuration
    custom_sampling: bool = True  # Whether to use custom sampling instead of random
    gps_sample: bool = False  # Whether to use GPS-based sampling
    sim_sample: bool = True  # Whether to use similarity-based sampling
    neighbour_select: int = 32  # Maximum selection size from pool
    neighbour_range: int = 64  # Pool size for selection
    gps_dict_path: str = "./data/CVACT/gps_dict.pkl"  # Path to pre-computed GPS distances
    
    # Evaluation settings
    batch_size_eval: int = 128  # Batch size for evaluation
    eval_every_n_epoch: int = 4  # Evaluate every N epochs
    normalize_features: bool = True  # Whether to normalize features
    
    # Optimizer settings
    clip_grad = 100.  # Gradient clipping value (None or float)
    decay_exclue_bias: bool = False  # Whether to exclude bias from weight decay
    grad_checkpointing: bool = False  # Whether to use gradient checkpointing
    
    # Loss settings
    label_smoothing: float = 0.1  # Label smoothing factor
    
    # Learning rate settings
    lr: float = 0.0001  # Learning rate (1e-4 for ViT, 1e-3 for CNN)
    scheduler: str = "cosine"  # Learning rate scheduler type
    warmup_epochs: int = 1  # Number of warmup epochs
    lr_end: float = 0.00001  # Final learning rate (for polynomial scheduler)
    
    # Dataset settings
    data_folder = "/ssd-data"  # Root directory for dataset
    
    # Image augmentation settings
    prob_rotate: float = 0.75  # Probability of rotating images
    prob_flip: float = 0.5  # Probability of flipping images
    
    # Model checkpoint settings
    model_path: str = "./cvact"  # Directory to save model checkpoints
    
    # Evaluation settings
    zero_shot: bool = True  # Whether to evaluate before training
    checkpoint_start = None  # Path to pretrained weights
    
    # System settings
    num_workers: int = 0 if os.name == 'nt' else 4  # Number of data loading workers
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to use
    cudnn_benchmark: bool = True  # Whether to use cuDNN benchmarking
    cudnn_deterministic: bool = False  # Whether to make cuDNN deterministic

# Initialize configuration
config = Configuration()

if __name__ == '__main__':
    # Create timestamp for model path
    start_time = datetime.datetime.now().strftime('%m%d%H%M')
    config = Configuration()
    
    # Create model directory and setup logging
    model_path = "{}/{}/{}".format(config.model_path,
                                  config.model,
                                  start_time)
    
    writer = None
    sim_dict = None
    
    print('---------------------model path is creating')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shutil.copyfile(os.path.basename(__file__), "{}/train.py".format(model_path))
    
    # Setup logging
    log_file = os.path.join(model_path, f'train_{start_time}.log')
    sys.stdout = Logger(log_file, auto_flush=True)
    
    # Setup TensorBoard
    tensorboard_log_dir = os.path.join(model_path, 'tensorboard')
    writer = SummaryWriter(tensorboard_log_dir)
    print(f"Training started at: {start_time}")
    print(f"Model path: {model_path}")
    print(f"Log file: {log_file}")
    print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

    # Initialize model
    model = TimmModel_v5(
        config.model,
        pretrained=True,
        img_size=config.img_size,
        device=config.device
    )
    
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
    
    # Enable gradient checkpointing if specified
    if config.grad_checkpointing:
        model.set_grad_checkpointing(True)
     
    # Load pretrained weights if specified
    if config.checkpoint_start is not None:  
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)  
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
    # DataLoader Setup                                                            #
    #-----------------------------------------------------------------------------#

    # Get transforms for training and validation
    sat_transforms_train, ground_transforms_train = get_transforms_train(image_size_sat,
                                                                   img_size_ground,
                                                                   mean=mean,
                                                                   std=std)
                                                                   
    ground_transforms_depth = get_transforms_depth(img_size_ground,
                                                 ground_cutting=0)
    sat_transforms_bev = get_transforms_bev(image_size_sat)
    sat_transforms_bev_depth = get_transforms_bev_depth(image_size_sat) 
                                                                   
    # Setup training dataset and dataloader
    train_dataset = CVACTDatasetTrain(data_folder=config.data_folder,
                                     transforms_query=ground_transforms_train,
                                     transforms_reference=sat_transforms_train,
                                     transforms_bev=sat_transforms_bev,
                                     transforms_depth=ground_transforms_depth,
                                     prob_flip=config.prob_flip,
                                     prob_rotate=config.prob_rotate,
                                     shuffle_batch_size=config.batch_size)
    
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=config.batch_size,
                                 num_workers=config.num_workers,
                                 shuffle=not config.custom_sampling,
                                 pin_memory=True)
    
    # Setup validation transforms
    sat_transforms_val, ground_transforms_val = get_transforms_val(image_size_sat,
                                                               img_size_ground,
                                                               mean=mean,
                                                               std=std)

    # Setup validation datasets and dataloaders
    reference_dataset_val = CVACTDatasetEval(data_folder=config.data_folder,
                                           split="val",
                                           img_type="reference",
                                           transforms=sat_transforms_val)
    
    reference_dataloader_val = DataLoader(reference_dataset_val,
                                        batch_size=config.batch_size_eval,
                                        num_workers=config.num_workers,
                                        shuffle=False,
                                        pin_memory=True)
    
    query_dataset_val = CVACTDatasetEval(data_folder=config.data_folder,
                                       split="val",
                                       img_type="query",    
                                       transforms=ground_transforms_val)
    
    query_dataloader_val = DataLoader(query_dataset_val,
                                    batch_size=config.batch_size_eval,
                                    num_workers=config.num_workers,
                                    shuffle=False,
                                    pin_memory=True)
    
    print("Reference Images Val:", len(reference_dataset_val))
    print("Query Images Val:", len(query_dataset_val))
    
    #-----------------------------------------------------------------------------#
    # GPS Sampling Setup                                                          #
    #-----------------------------------------------------------------------------#
    if config.gps_sample:
        with open(config.gps_dict_path, "rb") as f:
            sim_dict = pickle.load(f)
    else:
        sim_dict = None

    #-----------------------------------------------------------------------------#
    # Similarity Sampling Setup                                                   #
    #-----------------------------------------------------------------------------#
    if config.sim_sample:
        # Setup training datasets for similarity sampling
        query_dataset_train = CVACTDatasetEval(data_folder=config.data_folder,
                                             split="train",
                                             img_type="query",   
                                             transforms=ground_transforms_val)
            
        query_dataloader_train = DataLoader(query_dataset_train,
                                          batch_size=config.batch_size_eval,
                                          num_workers=config.num_workers,
                                          shuffle=False,
                                          pin_memory=True)
        
        reference_dataset_train = CVACTDatasetEval(data_folder=config.data_folder,
                                                 split="train",
                                                 img_type="reference", 
                                                 transforms=sat_transforms_val)
        
        reference_dataloader_train = DataLoader(reference_dataset_train,
                                              batch_size=config.batch_size_eval,
                                              num_workers=config.num_workers,
                                              shuffle=False,
                                              pin_memory=True)

        print("\nReference Images Train:", len(reference_dataset_train))
        print("Query Images Train:", len(query_dataset_train))        

    #-----------------------------------------------------------------------------#
    # Loss Function Setup                                                          #
    #-----------------------------------------------------------------------------#
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    loss_function = InfoNCE(loss_function=loss_fn,
                           device=config.device)

    # Setup mixed precision training if enabled
    if config.mixed_precision:
        scaler = GradScaler(init_scale=2.**10)
    else:
        scaler = None
        
    #-----------------------------------------------------------------------------#
    # Optimizer Setup                                                             #
    #-----------------------------------------------------------------------------#
    if config.decay_exclue_bias:
        # Setup optimizer with different weight decay for bias and non-bias parameters
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=config.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    #-----------------------------------------------------------------------------#
    # Learning Rate Scheduler Setup                                               #
    #-----------------------------------------------------------------------------#
    train_steps = len(train_dataloader) * config.epochs
    warmup_steps = len(train_dataloader) * config.warmup_epochs
       
    if config.scheduler == "polynomial":
        print("\nScheduler: polynomial - max LR: {} - end LR: {}".format(config.lr, config.lr_end))  
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                           num_training_steps=train_steps,
                                                           lr_end=config.lr_end,
                                                           power=1.5,
                                                           num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "cosine":
        print("\nScheduler: cosine - max LR: {}".format(config.lr))   
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                  num_training_steps=train_steps,
                                                  num_warmup_steps=warmup_steps)
        
    elif config.scheduler == "constant":
        print("\nScheduler: constant - max LR: {}".format(config.lr))   
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps)
           
    else:
        scheduler = None
        
    print("Warmup Epochs: {} - Warmup Steps: {}".format(str(config.warmup_epochs).ljust(2), warmup_steps))
    print("Train Epochs:  {} - Train Steps:  {}".format(config.epochs, train_steps))
        
    #-----------------------------------------------------------------------------#
    # Zero-shot Evaluation                                                        #
    #-----------------------------------------------------------------------------#
    if config.zero_shot:
        print("\n{}[{}]{}".format(30*"-", "Zero Shot", 30*"-"))  

        # Evaluate on validation set
        r1_test = evaluate(config=config,
                          model=model,
                          reference_dataloader=reference_dataloader_val,
                          query_dataloader=query_dataloader_val, 
                          ranks=[1, 5, 10],
                          step_size=1000,
                          cleanup=True)
        
        # Calculate similarities for training set if using similarity sampling
        if config.sim_sample:
            r1_train, sim_dict = calc_sim(config=config,
                                        model=model,
                                        reference_dataloader=reference_dataloader_train,
                                        query_dataloader=query_dataloader_train, 
                                        ranks=[1, 5, 10],
                                        step_size=1000,
                                        cleanup=True)
                
    #-----------------------------------------------------------------------------#
    # Custom Sampling Setup                                                       #
    #-----------------------------------------------------------------------------#            
    if config.custom_sampling:
        train_dataloader.dataset.shuffle(sim_dict,
                                       neighbour_select=config.neighbour_select,
                                       neighbour_range=config.neighbour_range)
            
    #-----------------------------------------------------------------------------#
    # Training Loop                                                               #
    #-----------------------------------------------------------------------------#
    start_epoch = 0   
    best_score = 0
    
    for epoch in range(1, config.epochs+1):
        print("\n{}[Epoch: {}]{}".format(30*"-", epoch, 30*"-"))
        
        # Training step
        train_loss = train(config,
                          model,
                          dataloader=train_dataloader,
                          loss_function=loss_function,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          scaler=scaler)
        
        print("Epoch: {}, Train Loss = {:.3f}, Lr = {:.6f}".format(epoch,
                                                                  train_loss,
                                                                  optimizer.param_groups[0]['lr']))
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Evaluation step
        if (epoch % config.eval_every_n_epoch == 0 and epoch != 0) or epoch == config.epochs:
            print("\n{}[{}]{}".format(30*"-", "Evaluate", 30*"-"))
        
            # Evaluate on validation set
            results = evaluate(config=config,
                             model=model,
                             reference_dataloader=reference_dataloader_val,
                             query_dataloader=query_dataloader_val, 
                             ranks=[1, 5, 10],
                             step_size=1000,
                             cleanup=True)
            
            r1, r5, r10, r_1 = results
            
            # Log recall metrics to TensorBoard
            writer.add_scalar('Recall/R@1', r1, epoch)
            writer.add_scalar('Recall/R@5', r5, epoch)
            writer.add_scalar('Recall/R@10', r10, epoch)
            writer.add_scalar('Recall/R@Top1', r_1, epoch)
            
            # Update similarity dictionary if using similarity sampling
            if config.sim_sample:
                r1_train, sim_dict = calc_sim(config=config,
                                            model=model,
                                            reference_dataloader=reference_dataloader_train,
                                            query_dataloader=query_dataloader_train, 
                                            ranks=[1, 5, 10],
                                            step_size=1000,
                                            cleanup=True)
                
            # Save best model
            if r1 > best_score:
                best_score = r1
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score,
                }
                if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
                    torch.save(model.module.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1))
                else:
                    torch.save(model.state_dict(), '{}/weights_e{}_{:.4f}.pth'.format(model_path, epoch, r1))

        # Update sampling if using custom sampling
        if config.custom_sampling:
            train_dataloader.dataset.shuffle(sim_dict,
                                           neighbour_select=config.neighbour_select,
                                           neighbour_range=config.neighbour_range)
                
    # Save final model
    if torch.cuda.device_count() > 1 and len(config.gpu_ids) > 1:
        torch.save(model.module.state_dict(), '{}/weights_end.pth'.format(model_path))
    else:
        torch.save(model.state_dict(), '{}/weights_end.pth'.format(model_path))  

    # Close TensorBoard writer
    if writer is not None:
        writer.close()

    #-----------------------------------------------------------------------------#
    # Final Test Evaluation                                                       #
    #-----------------------------------------------------------------------------#
    # Setup test datasets and dataloaders
    reference_dataset_test = CVACTDatasetTest(data_folder=config.data_folder,
                                            img_type="reference",
                                            transforms=sat_transforms_val)
    
    reference_dataloader_test = DataLoader(reference_dataset_test,
                                         batch_size=config.batch_size_eval,
                                         num_workers=config.num_workers,
                                         shuffle=False,
                                         pin_memory=True)
    
    query_dataset_test = CVACTDatasetTest(data_folder=config.data_folder,
                                        img_type="query",    
                                        transforms=ground_transforms_val)
    
    query_dataloader_test = DataLoader(query_dataset_test,
                                     batch_size=config.batch_size_eval,
                                     num_workers=config.num_workers,
                                     shuffle=False,
                                     pin_memory=True)
    
    print("Reference Images Test:", len(reference_dataset_test))
    print("Query Images Test:", len(query_dataset_test))          

    print("\n{}[{}]{}".format(30*"-", "Test", 30*"-"))  

    # Final evaluation on test set
    r1_test = evaluate(config=config,
                      model=model,
                      reference_dataloader=reference_dataloader_test,
                      query_dataloader=query_dataloader_test, 
                      ranks=[1, 5, 10],
                      step_size=1000,
                      cleanup=True)
