import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn

class InfoNCE(nn.Module):

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        
        self.loss_function = loss_function
        self.device = device

    def forward(self, image_features1, image_features2, logit_scale):
        image_features1 = F.normalize(image_features1, dim=-1)
        image_features2 = F.normalize(image_features2, dim=-1)
        
        logits_per_image1 = logit_scale * image_features1 @ image_features2.T
        
        logits_per_image2 = logits_per_image1.T
        
        labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
        
        loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels))/2

        return loss  
 

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1e-6):
        """
        Multi-class Dice Loss
        :param num_classes: Number of classes
        :param smooth: Smoothing term to prevent division by zero
        """
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Calculate multi-class Dice Loss
        :param pred: (B, C, H, W) - Must be **probability distribution** (output after Softmax)
        :param target: (B, H, W) - Class index for each pixel (0~C-1)
        :return: Dice Loss
        """
        B, C, H, W = pred.shape
        
        # Convert target to one-hot encoding (B, C, H, W)
        target_one_hot = torch.zeros_like(pred)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        # Calculate Dice coefficient
        intersection = (pred * target_one_hot).sum(dim=(2, 3))  # (B, C)
        union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))  # (B, C)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        dice_loss = 1 - dice.mean()  # Average across all classes

        return dice_loss