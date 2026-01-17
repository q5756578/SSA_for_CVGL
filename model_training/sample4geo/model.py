import torch
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn as nn
from .convnext_base import Conv2dNormActivation 
from .BEV_Layout import  BEV_Layout_Estimation,BEV_projection,MLSP,ConvNextBase_4first,ConvNextBase
from functools import partial 
from .convnext_base import LayerNorm2d 
import math
import torch.nn.functional as F

def modify_model(model):
    """
    Modify the model:
    1. Remove the last two layers of the model.
    2. Remove the layer at index (3) in the second nn.Sequential (if it exists).
    
    Args:
        model (nn.Module): PyTorch model to be modified.
    
    Returns:
        nn.Sequential: Modified model.
    """
    # Remove last two layers
    layers = list(model.children())[:-2]

    # Count occurrences of nn.Sequential
    seq_count = 0
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Sequential):
            seq_count += 1
            if seq_count == 2:  # Process second nn.Sequential
                layer = list(layer.children())  # Convert to list
                if len(layer) > 3:  # Ensure index (3) exists
                    del layer[3]  # Delete layer at index (3)
                layers[i] = nn.Sequential(*layer)  # Convert back to nn.Sequential
                break  # Only modify once, exit loop

    # Reconstruct model
    new_model = nn.Sequential(*layers)
    return new_model

def modify_model_v2(model):
    """
    Split model_ceshi and make the following modifications:
    1. Remove the last two layers of the model.
    2. Remove the first nn.Sequential.
    3. Delete the layer at index (3) in the second nn.Sequential (if it exists).
    
    Args:
        model (nn.Module): Original model.

    Returns:
        nn.Sequential: Modified model.
    """
    # Split model, remove last two layers
    layers = list(model.children())[:-2]

    # Extract all Sequential layers
    sequentials = [layer for layer in layers if isinstance(layer, nn.Sequential)]

    # Delete first Sequential
    if sequentials:
        sequentials.pop(0)

    # Modify second Sequential, delete index (3)
    if len(sequentials) > 0 and len(sequentials[0]) > 3:
        del sequentials[0][3]

    # Recombine model
    new_layers = [layer for layer in layers if not isinstance(layer, nn.Sequential)] + sequentials
    return nn.Sequential(*new_layers)  

def extract_custom_layers(model):
    """
    This function is used to extract the output head for contrastive learning to compress information
    """
    # 2. Get Sequential structure
    sequential_layers = [m for m in model.children() if isinstance(m, torch.nn.Sequential)]
    
    if len(sequential_layers) < 2:
        raise ValueError("At least two Sequential modules not found in model structure")
    
    second_sequential = sequential_layers[1]  # Get second Sequential
    if len(second_sequential) <= 3:
        raise ValueError("Layer (3) not found in second Sequential structure")

    layer_3 = second_sequential[3]  # Extract layer at index (3)

    # 3. Get last two layers of model
    model_children = list(model.children())
    last_two_layers = model_children[-2:]

    # 4. Combine new model
    new_model = torch.nn.Sequential(
        layer_3,         # Layer (3) from second Sequential
        *last_two_layers  # Last two layers
    )

    return new_model

class TimmModel(nn.Module):

    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383):
                 
        super(TimmModel, self).__init__()
        
        self.img_size = img_size
        
        if "vit" in model_name:
            # automatically change interpolate pos-encoding to img_size
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size) 
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img1, img2=None):
        
        if img2 is not None:
       
            image_features1 = self.model(img1)     
            image_features2 = self.model(img2)
            
            return image_features1, image_features2            
              
        else:
            image_features = self.model(img1)
             
            return image_features

class TimmModel_v2(nn.Module):
    #
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 backbone_channels=512, 
                 depth_channels=96, 
                 bev_projection_size=48,
                 device= 'cpu',
                 classes = 2
                    ):
                 
        super(TimmModel_v2, self).__init__()
        
        self.img_size = img_size
        
        
        '''
        # 实例化一个convnextbase ，去除后几层 ，使得输出的通道数为512
        '''
        model_sat= timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.model_sat = modify_model(model_sat)
        '''
        #实例化一个convnextbase去掉 第一层 和 后两层 
        地面这个编码器,需要增加一个新的头
        
        '''
        model_ground = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
      
        self.model_ground = modify_model_v2(model_ground)
        self.model_ground_head = nn.Sequential(
            nn.Conv2d(4,128,kernel_size=4,stride=4,padding=0,bias=True),
            LayerNorm2d(128)
            
        )
       
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.BEV_Estimation = BEV_Layout_Estimation(
            backbone_channels=backbone_channels,
            depth_channels=depth_channels,
            bev_projection_size=bev_projection_size,
            classes=classes,
            device=device,
            return_latent=False
            )
        self.contrac_outhead = extract_custom_layers(
                                                     timm.create_model(model_name, 
                                                                       pretrained=pretrained,
                                                                       num_classes=0)
                                                    )
        self.soft =  nn.Softmax(dim=1)
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model_sat)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img2_ground, img1_sat,img_type= None,multi_task=True):
        # img_sat : 384x384
        # img_ground : 1X4X384X768
    
        if img_type == "reference":
            emed_image_sat = self.model_sat(img1_sat) # 1X512X24x24 
            
            contrac_sat = self.contrac_outhead(emed_image_sat) 
            
            return contrac_sat
        elif img_type == "query":
            
            emed_image_ground  = self.model_ground(x)
            contrac_ground = self.contrac_outhead(emed_image_ground) 
            
            return contrac_ground
        
        else:
            if multi_task :
        
                emed_image_sat = self.model_sat(img1_sat) # 1X512X24x24 
    
                x  = self.model_ground_head(img2_ground)
                emed_image_ground  = self.model_ground(x) # 1X512X24X48
        
                bev_layout_feat  = self.BEV_Estimation(emed_image_ground)
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                bev_layout_feat = self.soft(bev_layout_feat)
                return contrac_sat,contrac_ground,bev_layout_feat             
                
            else:

                image_features1 = self.model_sat(img1_sat)   
                image_feat_bev_emb = self.model_ground(img2_ground) #cx32x32 
                
                return image_features1,image_feat_bev_emb


class TimmModel_v3(nn.Module):
    #
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 backbone_channels=512, 
                 depth_channels=96, 
                 bev_projection_size=48,
                 device= 'cpu',
                 classes = 2
                    ):
                 
        super(TimmModel_v3, self).__init__()
        
        self.img_size = img_size
        
        
        '''
        # 实例化一个convnextbase ，去除后几层 ，使得输出的通道数为512
        
        #实例化一个convnextbase去掉 第一层 和 后两层 
        地面这个编码器,需要增加一个新的头
        卫星同时也保留一个编码头
        '''
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
      
        self.model = modify_model_v2(model)
        self.model_sat_head  =  self.model_ground_head = nn.Sequential(
            nn.Conv2d(3,128,kernel_size=4,stride=4,padding=0,bias=True),
            LayerNorm2d(128)
            
        )
       
        self.model_ground_head = nn.Sequential(
            nn.Conv2d(4,128,kernel_size=4,stride=4,padding=0,bias=True),
            LayerNorm2d(128)
            
        )
       
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.BEV_Estimation = BEV_Layout_Estimation(
            backbone_channels=backbone_channels,
            depth_channels=depth_channels,
            bev_projection_size=bev_projection_size,
            classes=classes,
            device=device,
            return_latent=False
            )
        self.contrac_outhead = extract_custom_layers(
                                                     timm.create_model(model_name, 
                                                                       pretrained=pretrained,
                                                                       num_classes=0)
                                                    )
        self.soft =  nn.Softmax(dim=1)
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img2_ground, img1_sat,img_type= None,multi_task=True):
        # img_sat : 384x384
        # img_ground : 1X4X384X768
    
        if img_type == "reference":
            emed_image_sat  = self.model_sat_head(img1_sat)
            emed_image_sat = self.model(emed_image_sat) # 1X512X24x24 
            
            contrac_sat = self.contrac_outhead(emed_image_sat) 
            
            return contrac_sat
        elif img_type == "query":
            
            emed_image_ground  = self.model_ground_head(img2_ground)
            emed_image_ground  = self.model(emed_image_ground)
            contrac_ground = self.contrac_outhead(emed_image_ground) 
            
            return contrac_ground
        
        else:
            if multi_task :
        
                emed_image_sat = self.model_sat_head(img1_sat) # 1X512X24x24 
                emed_image_sat = self.model(emed_image_sat)
                
                emed_image_ground  = self.model_ground_head(img2_ground)
                emed_image_ground  = self.model(emed_image_ground) # 1X512X24X48
        
                bev_layout_feat  = self.BEV_Estimation(emed_image_ground)
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                bev_layout_feat = self.soft(bev_layout_feat)
                return contrac_sat,contrac_ground,bev_layout_feat             
                
            else:

                emed_image_sat = self.model_sat_head(img1_sat) # 1X512X24x24 
                emed_image_sat = self.model(emed_image_sat)
                
                emed_image_ground  = self.model_ground_head(img2_ground)
                emed_image_ground  = self.model(emed_image_ground) # 1X512X24X48
        
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                
                return contrac_sat,contrac_ground

class TimmModel_v4(nn.Module):
    #
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 backbone_channels=512, 
                 depth_channels=96, 
                 bev_projection_size=48,
                 device= 'cpu',
                 classes = 2
                    ):
                 
        super(TimmModel_v4, self).__init__()
        
        self.img_size = img_size
        
        
        '''
        # 实例化一个convnextbase ，去除后几层 ，使得输出的通道数为512
        
        #实例化一个convnextbase去掉 第一层 和 后两层 
        地面这个编码器,需要增加一个新的头
        卫星同时也保留一个编码头
        '''
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
      
        self.model = modify_model_v2(model)
        self.model_sat_head  =  self.model_ground_head = nn.Sequential(
            nn.Conv2d(3,128,kernel_size=4,stride=4,padding=0,bias=True),
            LayerNorm2d(128)
            
        )
       
        self.model_ground_head = nn.Sequential(
            nn.Conv2d(3,128,kernel_size=4,stride=4,padding=0,bias=True),
            LayerNorm2d(128)
            
        )
       
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.BEV_Estimation = BEV_Layout_Estimation(
            backbone_channels=backbone_channels,
            depth_channels=depth_channels,
            bev_projection_size=bev_projection_size,
            classes=classes,
            device=device,
            return_latent=False
            )
        self.contrac_outhead = extract_custom_layers(
                                                     timm.create_model(model_name, 
                                                                       pretrained=pretrained,
                                                                       num_classes=0)
                                                    )
        self.soft =  nn.Softmax(dim=1)
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img2_ground, img1_sat,img_type= None,multi_task=True):
        # img_sat : 384x384
        # img_ground : 1X4X384X768
    
        if img_type == "reference":
            emed_image_sat  = self.model_sat_head(img1_sat)
            emed_image_sat = self.model(emed_image_sat) # 1X512X24x24 
            
            contrac_sat = self.contrac_outhead(emed_image_sat) 
            
            return contrac_sat
        elif img_type == "query":
            
            emed_image_ground  = self.model_ground_head(img2_ground)
            emed_image_ground  = self.model(emed_image_ground)
            contrac_ground = self.contrac_outhead(emed_image_ground) 
            
            return contrac_ground
        
        else:
            if multi_task :
        
                emed_image_sat = self.model_sat_head(img1_sat) # 1X512X24x24 
                emed_image_sat = self.model(emed_image_sat)
                
                emed_image_ground  = self.model_ground_head(img2_ground)
                emed_image_ground  = self.model(emed_image_ground) # 1X512X24X48
        
                bev_layout_feat  = self.BEV_Estimation(emed_image_ground)
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                bev_layout_feat = self.soft(bev_layout_feat)
                return contrac_sat,contrac_ground,bev_layout_feat             
                
            else:

                emed_image_sat = self.model_sat_head(img1_sat) # 1X512X24x24 
                emed_image_sat = self.model(emed_image_sat)
                
                emed_image_ground  = self.model_ground_head(img2_ground)
                emed_image_ground  = self.model(emed_image_ground) # 1X512X24X48
        
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                
                return contrac_sat,contrac_ground

class TimmModel_v5(nn.Module):
    #
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 backbone_channels=512, 
                 depth_channels=96, 
                 bev_projection_size=48,
                 device= 'cpu',
                 classes = 2
                    ):
                 
        super(TimmModel_v5, self).__init__()
        
        self.img_size = img_size
        self.classes = classes
        
        '''
        # 实例化一个convnextbase ，去除后几层 ，使得输出的通道数为512
        
        #实例化一个convnextbase去掉 第一层 和 后两层 
        地面这个编码器,需要增加一个新的头
        卫星同时也保留一个编码头
        '''
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
      
        self.model = modify_model(model)
     
     
       
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.BEV_Estimation = BEV_Layout_Estimation(
            backbone_channels=backbone_channels,
            depth_channels=depth_channels,
            bev_projection_size=bev_projection_size,
            classes=classes,
            device=device,
            return_latent=False
            )
        self.contrac_outhead = extract_custom_layers(
                                                     timm.create_model(model_name, 
                                                                       pretrained=pretrained,
                                                                       num_classes=0)
                                                    )
        #self.soft =  nn.Softmax(dim=1)
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img2_ground, img1_sat,img_type= None,multi_task=True):
        # img_sat : 384x384
        # img_ground : 1X4X384X768
    
        if img_type == "reference":
            #emed_image_sat  = self.model_sat_head(img1_sat)
            emed_image_sat = self.model(img1_sat) # 1X512X24x24 
            
            contrac_sat = self.contrac_outhead(emed_image_sat) 
            
            return contrac_sat
        elif img_type == "query":
            
            emed_image_ground  = self.model(img2_ground)
            contrac_ground = self.contrac_outhead(emed_image_ground) 
            
            return contrac_ground
        
        else:
            if multi_task :
        
               # 1X512X24x24 
                emed_image_sat = self.model(img1_sat)
                
                emed_image_ground  = self.model(img2_ground) # 1X512X24X48
        
                bev_layout_feat  = self.BEV_Estimation(emed_image_ground)
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                #bev_layout_feat = self.soft(bev_layout_feat)
                return contrac_sat,contrac_ground,bev_layout_feat             
                
            else:

               
                emed_image_sat = self.model(img1_sat)
       
                emed_image_ground  = self.model(img2_ground) # 1X512X24X48
        
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                
                return contrac_sat,contrac_ground

class TimmModel_v6(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 backbone_channels=512, 
                 depth_channels=96, 
                 bev_projection_size=48,
                 device='cpu',
                 classes=2,
                 freeze_backbone=True  # Add freeze parameter
                ):
                 
        super(TimmModel_v6, self).__init__()
        
        self.img_size = img_size
        
        # Instantiate base model
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
        self.model = modify_model(model)
        self.ground_model = modify_model(timm.create_model(model_name, pretrained=pretrained, num_classes=0))
        # Freeze backbone weights
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            print("Backbone weights are frozen")
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # These layers remain trainable
        self.BEV_Estimation = BEV_Layout_Estimation(
            backbone_channels=backbone_channels,
            depth_channels=depth_channels,
            bev_projection_size=bev_projection_size,
            classes=classes,
            device=device,
            return_latent=False
        )
        
        self.contrac_outhead = extract_custom_layers(
            timm.create_model(model_name, 
                            pretrained=pretrained,
                            num_classes=0)
        )
        self.soft = nn.Softmax(dim=1)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("Backbone weights are unfrozen")
    
    def freeze_backbone(self):
        """Freeze backbone"""
        for param in self.model.parameters():
            param.requires_grad = False
        print("Backbone weights are frozen")
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        return trainable_params
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img2_ground, img1_sat,img_type= None,multi_task=True):
        # img_sat : 384x384
        # img_ground : 1X4X384X768
    
        if img_type == "reference":
            #emed_image_sat  = self.model_sat_head(img1_sat)
            emed_image_sat = self.model(img1_sat) # 1X512X24x24 
            
            contrac_sat = self.contrac_outhead(emed_image_sat) 
            
            return contrac_sat
        elif img_type == "query":
            
            emed_image_ground  = self.ground_model(img2_ground)
            contrac_ground = self.contrac_outhead(emed_image_ground) 
            
            return contrac_ground
        
        else:
            if multi_task :
        
               # 1X512X24x24 
                emed_image_sat = self.model(img1_sat)
                
                emed_image_ground  = self.ground_model(img2_ground) # 1X512X24X48
        
                bev_layout_feat  = self.BEV_Estimation(emed_image_ground)
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                bev_layout_feat = self.soft(bev_layout_feat)
                return contrac_sat,contrac_ground,bev_layout_feat             
                
            else:

               
                emed_image_sat = self.model(img1_sat)
       
                emed_image_ground  = self.ground_model(img2_ground) # 1X512X24X48
        
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                
                return contrac_sat,contrac_ground

class TimmModel_v7(nn.Module):
    def __init__(self, 
                 model_name,
                 pretrained=True,
                 img_size=383,
                 backbone_channels=512, 
                 depth_channels=96, 
                 bev_projection_size=48,
                 device='cpu',
                 classes=2,
                 freeze_backbone=True  # Add freeze parameter
                ):
                 
        super(TimmModel_v7, self).__init__()
        
        self.img_size = img_size
        
        # Instantiate base model
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0) 
        self.model =  modify_model_v2(model)
        self.model_head = nn.Sequential(
            nn.Conv2d(4,128,kernel_size=4,stride=4,padding=0,bias=True),
            LayerNorm2d(128)
            
        )
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        # These layers remain trainable
        self.BEV_Estimation = BEV_Layout_Estimation(
            backbone_channels=backbone_channels,
            depth_channels=depth_channels,
            bev_projection_size=bev_projection_size,
            classes=classes,
            device=device,
            return_latent=False
        )
        
        self.contrac_outhead = extract_custom_layers(
            timm.create_model(model_name, 
                            pretrained=pretrained,
                            num_classes=0)
        )
        self.soft = nn.Softmax(dim=1)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("Backbone weights are unfrozen")
    
    def freeze_backbone(self):
        """Freeze backbone"""
        for param in self.model.parameters():
            param.requires_grad = False
        print("Backbone weights are frozen")
    
    def get_trainable_params(self):
        """Get number of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
        return trainable_params
        
    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.model)
        return data_config
    
    
    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

        
    def forward(self, img2_ground, img1_sat,img_type= None,multi_task=True):
        # img_sat : 384x384
        # img_ground : 1X4X384X768
    
        if img_type == "reference":
            #emed_image_sat  = self.model_sat_head(img1_sat)
            emed_image_sat = self.model_head(img1_sat) # 1X512X24x24 
            emed_image_sat = self.model(emed_image_sat)
            contrac_sat = self.contrac_outhead(emed_image_sat) 
            
            return contrac_sat
        elif img_type == "query":
            
            emed_image_ground  = self.model_head(img2_ground)
            emed_image_ground  = self.model(emed_image_ground)
            contrac_ground = self.contrac_outhead(emed_image_ground) 
            
            return contrac_ground
        
        else:
            if multi_task :
        
               # 1X512X24x24 
                emed_image_sat = self.model_head(img1_sat)
                emed_image_sat  = self.model(emed_image_sat)
                
                emed_image_ground  = self.model_head(img2_ground) # 1X512X24X48
                emed_image_ground  = self.model(emed_image_ground)
                
                bev_layout_feat  = self.BEV_Estimation(emed_image_ground)
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                bev_layout_feat = self.soft(bev_layout_feat)
                return contrac_sat,contrac_ground,bev_layout_feat             
                
            else:

               
                emed_image_sat = self.model_head(img1_sat)
                emed_image_ground = self.model_head(img2_ground)
                
                
                emed_image_ground  = self.model(emed_image_ground) # 1X512X24X48
                emed_image_sat     = self.model(emed_image_sat)
        
                
                contrac_sat = self.contrac_outhead(emed_image_sat)
                contrac_ground = self.contrac_outhead(emed_image_ground)
                
                
                return contrac_sat,contrac_ground

class TimmModel_v8(nn.Module):
    def __init__(self, 
                 model_name='convnext_small',
                 pretrained=True,
                 img_size=384,
                 fusion_dim=512,
                 num_fusion_layers=2,
                 output_dim=1024,
                 num_heads=8,
                 mlp_ratio=4.,
                 drop_rate=0.1,
                 attn_drop_rate=0.1,
                 drop_path_rate=0.1):
        super().__init__()
        
        # RGB Encoder (ConvNeXt-small first two stages)
        rgb_model = timm.create_model(model_name, pretrained=pretrained)
        self.rgb_encoder = nn.Sequential(*list(rgb_model.children())[:2])
        
        # Depth Encoder (ConvNeXt-small first two stages)
        depth_model = timm.create_model(model_name, pretrained=False)
        # Modify first convolution layer to accept single channel input
        first_conv = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        # Get first layer of model and replace
        for name, module in depth_model.named_children():
            if isinstance(module, nn.Sequential):
                if isinstance(module[0], nn.Conv2d):
                    module[0] = first_conv
                    break
        self.depth_encoder = nn.Sequential(*list(depth_model.children())[:2])
        
        # Apply He initialization to depth branch
        def _init_depth_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Apply He initialization to depth encoder
        self.depth_encoder.apply(_init_depth_weights)
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, img_size, img_size)
            feat_dim = self.rgb_encoder(dummy_input).shape[1]
        
        # Other parts remain unchanged
        self.feature_dim = fusion_dim
        
        # Projection layers
        self.rgb_proj = nn.Linear(feat_dim, fusion_dim)
        self.depth_proj = nn.Linear(feat_dim, fusion_dim)
        
        # Positional embeddings
        self.rgb_pos_embed = nn.Parameter(torch.zeros(1, (img_size//16)**2, fusion_dim))
        self.depth_pos_embed = nn.Parameter(torch.zeros(1, (img_size//16)**2, fusion_dim))
        
        # Fusion transformer layers
        self.fusion_layers = nn.ModuleList([
            FusionTransformerV2(
                dim=fusion_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate
            ) for _ in range(num_fusion_layers)
        ])
        
        # Final MLP head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(fusion_dim * 4, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize non-depth parts
        self._init_other_weights()

    def _init_other_weights(self):
        """Initialize weights for non-depth parts"""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        # Apply initialization to all parts except depth encoder
        self.rgb_proj.apply(_init_weights)
        self.depth_proj.apply(_init_weights)
        self.fusion_layers.apply(_init_weights)
        self.mlp_head.apply(_init_weights)

    def forward(self, img2_ground, img1_sat, img_type=None, multi_task=False):
        """
        Args:
            img2_ground: Ground view input (B, 4, H, W) - 3 RGB channels + 1 depth channel
            img1_sat: Satellite view input (B, 4, H, W) - 3 RGB channels + 1 depth channel
            img_type: 'reference' or 'query' or None
            multi_task: Whether to enable multi-task mode
        """
        if img_type == "reference":
            # Process satellite image
            sat_rgb = img1_sat[:, :3, :, :]  # RGB channels
            sat_depth = img1_sat[:, 3:, :, :]  # Depth channel
            
            # Encode features
            sat_rgb_feat = self.rgb_encoder(sat_rgb)
            sat_depth_feat = self.depth_encoder(sat_depth)
            
            # Reshape and project features
            sat_rgb_feat = sat_rgb_feat.flatten(2).transpose(1, 2)
            sat_depth_feat = sat_depth_feat.flatten(2).transpose(1, 2)
            
            sat_rgb_feat = self.rgb_proj(sat_rgb_feat) + self.rgb_pos_embed
            sat_depth_feat = self.depth_proj(sat_depth_feat) + self.depth_pos_embed
            
            # Pass through fusion layers
            for layer in self.fusion_layers:
                fused_feat = layer(sat_rgb_feat, sat_depth_feat)
                sat_rgb_feat = fused_feat[..., :self.feature_dim]
                sat_depth_feat = fused_feat[..., self.feature_dim:]
            
            # Output fused features
            sat_output = self.mlp_head(torch.mean(sat_rgb_feat, dim=-1))
            return sat_output

        elif img_type == "query":
            # Process ground image
            ground_rgb = img2_ground[:, :3, :, :]  # RGB channels
            ground_depth = img2_ground[:, 3:, :, :]  # Depth channel
            
            # Encode features
            ground_rgb_feat = self.rgb_encoder(ground_rgb)
            ground_depth_feat = self.depth_encoder(ground_depth)
            
            # Reshape and project features
            ground_rgb_feat = ground_rgb_feat.flatten(2).transpose(1, 2)
            ground_depth_feat = ground_depth_feat.flatten(2).transpose(1, 2)
            
            ground_rgb_feat = self.rgb_proj(ground_rgb_feat) + self.rgb_pos_embed
            ground_depth_feat = self.depth_proj(ground_depth_feat) + self.depth_pos_embed
            
            # Pass through fusion layers
            for layer in self.fusion_layers:
                fused_feat = layer(ground_rgb_feat, ground_depth_feat)
                ground_rgb_feat = fused_feat[..., :self.feature_dim]
                ground_depth_feat = fused_feat[..., self.feature_dim:]
            
            # Output fused features
            ground_output = self.mlp_head(torch.mean(ground_rgb_feat, dim=1))
            return ground_output

        else:
            # Process both ground and satellite images
            # Process ground image
            ground_rgb = img2_ground[:, :3, :, :]
            ground_depth = img2_ground[:, 3:, :, :]
            
            # Process satellite image
            sat_rgb = img1_sat[:, :3, :, :]
            sat_depth = img1_sat[:, 3:, :, :]
            
            # Encode features and immediately clean up intermediate results
            ground_rgb_feats = self.rgb_encoder(ground_rgb)
            ground_rgb_feat = ground_rgb_feats[-1]  # Take features from last stage
            del ground_rgb_feats
            
            ground_depth_feats = self.depth_encoder(ground_depth)
            ground_depth_feat = ground_depth_feats[-1]
            del ground_depth_feats
            
            sat_rgb_feats = self.rgb_encoder(sat_rgb)
            sat_rgb_feat = sat_rgb_feats[-1]  # Take features from last stage
            del sat_rgb_feats
            
            sat_depth_feats = self.depth_encoder(sat_depth)
            sat_depth_feat = sat_depth_feats[-1]  # Take features from last stage
            del sat_depth_feats
            
            # Reshape and project features
            B, C, H, W = ground_rgb_feat.shape
            ground_rgb_feat = ground_rgb_feat.flatten(2).transpose(1, 2)
            ground_depth_feat = ground_depth_feat.flatten(2).transpose(1, 2)
            
            B_sat, C_sat, H_sat, W_sat = sat_rgb_feat.shape
            sat_rgb_feat = sat_rgb_feat.flatten(2).transpose(1, 2)
            sat_depth_feat = sat_depth_feat.flatten(2).transpose(1, 2)
            
            # Adjust positional embedding size to match feature map
            if self.rgb_pos_embed.shape[1] != H * W:
                ground_rgb_pos_embed = F.interpolate(
                    self.rgb_pos_embed.reshape(1, int(math.sqrt(self.rgb_pos_embed.shape[1])), 
                    int(math.sqrt(self.rgb_pos_embed.shape[1])), self.fusion_dim).permute(0, 3, 1, 2),
                    size=(H, W), mode='bilinear'
                ).permute(0, 2, 3, 1).reshape(1, H * W, self.fusion_dim)
                
                ground_depth_pos_embed = F.interpolate(
                    self.depth_pos_embed.reshape(1, int(math.sqrt(self.depth_pos_embed.shape[1])), 
                    int(math.sqrt(self.depth_pos_embed.shape[1])), self.fusion_dim).permute(0, 3, 1, 2),
                    size=(H, W), mode='bilinear'
                ).permute(0, 2, 3, 1).reshape(1, H * W, self.fusion_dim)
                
                sat_rgb_pos_embed = F.interpolate(
                    self.rgb_pos_embed.reshape(1, int(math.sqrt(self.rgb_pos_embed.shape[1])), 
                    int(math.sqrt(self.rgb_pos_embed.shape[1])), self.fusion_dim).permute(0, 3, 1, 2),
                    size=(H_sat, W_sat), mode='bilinear'
                ).permute(0, 2, 3, 1).reshape(1, H_sat * W_sat, self.fusion_dim)
                
                sat_depth_pos_embed = F.interpolate(
                    self.depth_pos_embed.reshape(1, int(math.sqrt(self.depth_pos_embed.shape[1])), 
                    int(math.sqrt(self.depth_pos_embed.shape[1])), self.fusion_dim).permute(0, 3, 1, 2),
                    size=(H_sat, W_sat), mode='bilinear'
                ).permute(0, 2, 3, 1).reshape(1, H_sat * W_sat, self.fusion_dim)
            else:
                ground_rgb_pos_embed = sat_rgb_pos_embed = self.rgb_pos_embed
                ground_depth_pos_embed = sat_depth_pos_embed = self.depth_pos_embed
            
            # Project and add positional embeddings
            ground_rgb_feat = self.rgb_proj(ground_rgb_feat) + ground_rgb_pos_embed
            ground_depth_feat = self.depth_proj(ground_depth_feat) + ground_depth_pos_embed
            sat_rgb_feat = self.rgb_proj(sat_rgb_feat) + sat_rgb_pos_embed
            sat_depth_feat = self.depth_proj(sat_depth_feat) + sat_depth_pos_embed
            
            # Fuse ground and satellite features separately
            for layer in self.fusion_layers:
                ground_rgb_feat, ground_depth_feat = layer(ground_rgb_feat, ground_depth_feat)
                sat_rgb_feat, sat_depth_feat = layer(sat_rgb_feat, sat_depth_feat)
            
            # Generate final features
            ground_output = self.mlp_head(torch.mean(torch.cat([ground_rgb_feat, ground_depth_feat], dim=-1), dim=1))
            sat_output = self.mlp_head(torch.mean(torch.cat([sat_rgb_feat, sat_depth_feat], dim=-1), dim=1))
            
            return sat_output, ground_output

    def get_config(self):
        return {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }

if __name__ == '__main__':
    model = TimmModel(model_name='convnext_base.fb_in22k_ft_in1k_384',img_size=384,pretrained=True)
    print(model)
    image = torch.randn((1,3,384,384))
    out = model(image)
    print(out.shape)