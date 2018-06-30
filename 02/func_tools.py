from torch import DoubleTensor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def convolution(image, kernel_size=3) -> DoubleTensor:
    """"Function for convolution of image with random kernels
    Args:
        image (DoubleTensor): Tensor representing image with size (height, width, channels_number)
        kernel_size (int): size of square random kernel; this size is the same for every channel in image
    
    Returns:
        DoubleTensor: new convoluted image with shape (height - kernel_size, height - kernel_size, in_channels)
    """
    
    in_channels = image.size()[2] # channels_number
    
    # create random kernel for every channel in image from normal distribution with params (0, 1)
    frames = DoubleTensor(kernel_size, kernel_size, in_channels).normal_(mean=0, std=1)
    
    # create new DoubleTensor for result of applied convolutions
    out_tensor = DoubleTensor(image.size(0) - kernel_size, image.size(1) - kernel_size, in_channels)    

    # iterate through Tensor and multiply it part with kernels pointwise
    for i in range(image.size(0) - kernel_size):
        for j in range(image.size(1) - kernel_size):
            convolution_res = (image[i:i + kernel_size, j:j + kernel_size, :] * frames).sum(dim=0).sum(dim=0)
            out_tensor[i, j, :] = convolution_res
    
    # values in tensor must be between 0 and 1, need to norm them
    out_tensor = (out_tensor / out_tensor.abs().max()) + 1
    
    return out_tensor

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
        
    
        
    
        
        

    
    
    