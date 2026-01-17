# ConvNext code modified based on official torch code
# Model download URLs:
#           tiny --- https://download.pytorch.org/models/convnext_tiny-983f1562.pth
#          small --- https://download.pytorch.org/models/convnext_small-0c510722.pth
#           base --- https://download.pytorch.org/models/convnext_base-6075fbad.pth
#          large --- https://download.pytorch.org/models/convnext_large-ea097f82.pth
 
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Any, Callable, List, Optional, Sequence
from functools import partial
 
 
# Define a convolution + normalization + activation layer, this class is only used once in the convnext stem layer
# torch.nn.Sequential is equivalent to tf2.0's keras.Sequential(), it's the simplest way to build a sequential model
# No need to write forward() function, just pass each submodule as a list, or use OrderedDict() or add_module() to add submodules
# https://blog.csdn.net/weixin_42486623/article/details/122822580
class Conv2dNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None
    ) -> None:
 
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
 
        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]
 
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
 
        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
 
        super().__init__(*layers)  # Add submodules to torch.nn.Sequential directly as a list
 
 

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)      # Move channel dimension to last position, normalize across all dimensions for each pixel, not across all pixels
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps) 
        x = x.permute(0, 3, 1, 2)      # Move channel dimension back to second position, since only one parameter is passed later, normalization is performed on the last dimension
        return x


##############################################################################################################################

