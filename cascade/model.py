import torch
import torch.nn as nn
from .layers import UnPool, PoolConv, DataConsistency, MaskLayer2D, MaskLayer1D, ConvBlockDC
from .binarize import *
import sys
sys.path.append('../mri_utils')
from mri_utils import *

class CascadeNet(nn.Module):

    def __init__(self, input_channel=4, image_size=256, num_blocks=3, num_layers=5, lambd=-1):
        '''
        >>> Build the Cascade of CNN module, as described in
            Schlemper, J. et al. " A Deep Cascade of Convolutional Neural Networks for 
            Dynamic MR Image Reconstruction". <https://arxiv.org/pdf/1704.02422.pdf>
        >>> image_size: int, the size of the images
        >>> num_blocks: the number of convolutional + DC blocks
        >>> num_layers: the number of convolutional layers in each of the convolutional blocks
        >>> lambd: parameter of the Data Consistency layer (Setting it to None yields a perfect data consistency)
        '''
        super(CascadeNet, self).__init__()

        self.image_size = image_size
        self.convblocklist = nn.ModuleList([])
        self.num_blocks = num_blocks
        self.convblocklist.append(ConvBlockDC(num_layers, input_channel, 64, 3, lambd=lambd).to('cuda'))
        for i in range(num_blocks-1):
            self.convblocklist.append(ConvBlockDC(num_layers, 2, 64,3,lambd =lambd).to('cuda'))
        
    def forward(self, x, mask):
        '''
        >>> feedforward process of CascadeNet

        >>> x should have the shape of [batch_size, 1, image_size, image_size, 2]
            the last dimension must be 2, representing the real and imaginary part.
        >>> mask: Tensor, the mask of shape [batchsize, image_size, image_size]
        >>> outputs
            >>> x: tensor of shape [batch_size, 2, image_size, image_size],
                the recovered image
        '''

        x_meas = x[:,:2]
        for i in range(self.num_blocks):
            x = self.convblocklist[i](x,x_meas, mask)
        return x
