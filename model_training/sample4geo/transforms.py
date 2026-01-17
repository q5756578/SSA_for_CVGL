"""
This module provides image transformation pipelines for the VIGOR (Visual Geo-localization) model.
It includes custom transforms and predefined transformation pipelines for satellite images, ground images,
depth maps, and bird's eye view (BEV) outputs.
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import os 
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

class Cut(ImageOnlyTransform):
    """
    Custom transform to cut/trim images from top and bottom.
    
    This transform removes a specified number of pixels from both the top and bottom
    of the input image. It's particularly useful for ground images where the sky and
    ground regions may need to be removed.
    
    Args:
        cutting (int): Number of pixels to remove from top and bottom
        always_apply (bool): Whether to always apply this transform
        p (float): Probability of applying this transform
    """
    def __init__(self, 
                 cutting=None,
                 always_apply=False,
                 p=1.0):
        
        super(Cut, self).__init__(always_apply, p)
        self.cutting = cutting
    
    def apply(self, image, **params):
        """
        Apply the cutting transform to the image.
        
        Args:
            image (numpy.ndarray): Input image to transform
            **params: Additional parameters
            
        Returns:
            numpy.ndarray: Transformed image with top and bottom regions removed
        """
        if self.cutting:
            image = image[self.cutting:-self.cutting,:,:]
            
        return image
            
    def get_transform_init_args_names(self):
        """Return the names of arguments used to initialize the transform."""
        return ("size", "cutting")     




def get_transforms_train(image_size_sat,
                         img_size_ground,
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         ground_cutting=0):
    """
    Get transformation pipelines for training satellite and ground images.
    
    The satellite transform pipeline includes:
    - Image compression
    - Resizing
    - Color jittering
    - Random blur/sharpen
    - Random dropout
    - Normalization
    - Conversion to tensor
    
    The ground transform pipeline includes:
    - Top/bottom cutting
    - Image compression
    - Resizing
    - Color jittering
    - Random blur/sharpen
    - Random dropout
    - Normalization
    - Conversion to tensor
    
    Args:
        image_size_sat (tuple): Target size for satellite images (height, width)
        img_size_ground (tuple): Target size for ground images (height, width)
        mean (list): Mean values for normalization [R, G, B]
        std (list): Standard deviation values for normalization [R, G, B]
        ground_cutting (int): Number of pixels to remove from top and bottom of ground images
        
    Returns:
        tuple: (satellite_transforms, ground_transforms) - Two albumentations Compose objects
    """
    
    satellite_transforms = A.Compose([
                                      A.ImageCompression(quality_range=(90,100), p=0.5),
                                      A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
                                      A.OneOf([
                                               A.AdvancedBlur(p=1.0),
                                               A.Sharpen(p=1.0),
                                              ], p=0.3),
                                      A.OneOf([
                                               A.GridDropout(ratio=0.4, p=1.0),
                                               A.CoarseDropout(
                                                               num_holes_range = (10,25),
                                                              hole_height_range = (int(0.1*image_size_sat[0]),int(0.2*image_size_sat[0])),
                                                              hole_width_range  = (int(0.1*image_size_sat[0]),int(0.2*image_size_sat[0])),
                                                               p=1.0),
                                              ], p=0.3),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    

    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.ImageCompression(quality_range=(90,100), p=0.5),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15, p=0.5),
                                   A.OneOf([
                                            A.AdvancedBlur(p=1.0),
                                            A.Sharpen(p=1.0),
                                           ], p=0.3),
                                   A.OneOf([
                                            A.GridDropout(ratio=0.5, p=1.0),
                                            A.CoarseDropout(
                                                               num_holes_range = (10,25),
                                                              hole_height_range = (int(0.1*image_size_sat[0]),int(0.2*image_size_sat[0])),
                                                              hole_width_range  = (int(0.1*image_size_sat[0]),int(0.2*image_size_sat[0])),
                                                               p=1.0),
                                           ], p=0.3),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                   ])
    
               
    return satellite_transforms, ground_transforms


def get_transforms_depth(img_size_ground, ground_cutting=0):
    """
    Get transformation pipeline for depth maps.
    
    The depth transform pipeline includes:
    - Top/bottom cutting
    - Resizing
    - Conversion to tensor
    
    Args:
        img_size_ground (tuple): Target size for depth maps (height, width)
        ground_cutting (int): Number of pixels to remove from top and bottom
        
    Returns:
        albumentations.Compose: Transformation pipeline for depth maps
    """
    
    depth_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   ToTensorV2(),
                                   ])
            
    return depth_transforms 


def get_transforms_bev(image_size_bev):
    """
    Get transformation pipeline for bird's eye view (BEV) outputs.
    
    The BEV transform pipeline includes:
    - Resizing using nearest neighbor interpolation
    - Conversion to tensor
    
    Args:
        image_size_bev (tuple): Target size for BEV outputs (height, width)
        
    Returns:
        albumentations.Compose: Transformation pipeline for BEV outputs
    """
    
    depth_transforms = A.Compose([
                                   A.Resize(image_size_bev[0], image_size_bev[1], interpolation=cv2.INTER_NEAREST, p=1.0),
                                   ToTensorV2(),
                                   ])
            
    return depth_transforms 

def get_transforms_bev_depth(image_size_bev):
    """
    Get transformation pipeline for BEV depth maps.
    
    The BEV depth transform pipeline includes:
    - Resizing using linear interpolation
    - Conversion to tensor
    
    Args:
        image_size_bev (tuple): Target size for BEV depth maps (height, width)
        
    Returns:
        albumentations.Compose: Transformation pipeline for BEV depth maps
    """
    
    depth_transforms = A.Compose([
                                   A.Resize(image_size_bev[0], image_size_bev[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   ToTensorV2(),
                                   ])
            
    return depth_transforms 


def get_transforms_val(image_size_sat,
                       img_size_ground,
                       mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225],
                       ground_cutting=0):
    """
    Get transformation pipelines for validation satellite and ground images.
    
    The validation pipelines are simpler than training pipelines, including only:
    - Resizing
    - Normalization
    - Conversion to tensor
    
    For ground images, also includes:
    - Top/bottom cutting
    
    Args:
        image_size_sat (tuple): Target size for satellite images (height, width)
        img_size_ground (tuple): Target size for ground images (height, width)
        mean (list): Mean values for normalization [R, G, B]
        std (list): Standard deviation values for normalization [R, G, B]
        ground_cutting (int): Number of pixels to remove from top and bottom of ground images
        
    Returns:
        tuple: (satellite_transforms, ground_transforms) - Two albumentations Compose objects
    """
    
    satellite_transforms = A.Compose([
                                    A.Resize(image_size_sat[0], image_size_sat[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                      A.Normalize(mean, std),
                                      ToTensorV2(),
                                     ])
            
    ground_transforms = A.Compose([Cut(cutting=ground_cutting, p=1.0),
                                   A.Resize(img_size_ground[0], img_size_ground[1], interpolation=cv2.INTER_LINEAR_EXACT, p=1.0),
                                   A.Normalize(mean, std),
                                   ToTensorV2(),
                                  ])
            
    return satellite_transforms, ground_transforms