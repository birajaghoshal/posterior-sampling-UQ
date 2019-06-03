import torch
import torch.nn as nn
import torch.nn.functional as F
from .binarize import *
import sys
sys.path.append('../mri_utils')
from mri_utils import *

class UnPool(nn.Module):
    '''
    >>> the inverse of pooling layer
    '''

    def __init__(self, kernel_size):
        '''
        >>> kernel_size: int, the size of the kernel 
        '''
        super(UnPool, self).__init__()

        self.kernel_size = kernel_size

    def forward(self, x):
        '''
        >>> x: torch.Tensor of shape [b, channel, d0, d1, ..., dn]
        >>> output: torch.Tensor of shape [b, channel, d0 * kernel_size, ..., dn * kernel_size]
        '''
        x_shape = x.size()
        dim = len(x_shape)
        for d in range(2, dim):
            x = torch.cat([x / self.kernel_size,] * self.kernel_size, dim = d)

        return x

class PoolConv(nn.Module):
    '''
    >>> the pooling-convolutional layer : a pooling layer followed by serveral convoluational layers
    '''

    def __init__(self, conv_channels, conv_kernels, pool_kernel, pool_type = 'max', nonlinearity = 'relu'):
        '''
        >>> conv_channels: list of int, the number of channels for convolutional layers
        >>> conv_kernels: list of int, the size of kernel size in the convolutional layers
        >>> pool_kernel: int, the kernel size of pooling, can be None
        >>> pool_type: str, the type of pooling, max-pooling, average-pooling or unpooling
        >>> nonlinearity: str, the nonlinearity
        '''
        super(PoolConv, self).__init__()
        assert len(conv_channels) == len(conv_kernels) + 1
        self.conv_channels = conv_channels
        self.conv_kernels = conv_kernels
        self.pool_kernel = pool_kernel
        self.pool_type = pool_type

        convlayer_list = []
        for idx, (in_channel, out_channel, conv_kernel) in enumerate(zip(self.conv_channels[:-1], self.conv_channels[1:], self.conv_kernels)):
            padding = (conv_kernel - 1) // 2
            convlayer_list.append(nn.Conv2d(in_channel, out_channel, kernel_size = conv_kernel, stride = 1, padding = padding))
            convlayer_list.append({'relu': nn.ReLU(), 'sigd': nn.Sigmoid(), 'tanh': nn.Tanh()}[nonlinearity])
        self.convlayer = nn.Sequential(*convlayer_list)

        poollayer_list = []
        if pool_kernel != None:
            poollayer_constructor = {'max': nn.MaxPool2d, 'average': nn.AvgPool2d, 'avg': nn.AvgPool2d, 'unpool': UnPool}[pool_type]
            poollayer_list.append(poollayer_constructor(self.pool_kernel))
        self.poollayer = nn.Sequential(*poollayer_list)

    def forward(self, x):
        '''
        >>> feedforward a conv-pool layer
        '''
        x = self.poollayer(x)
        x = self.convlayer(x)

        return x

class DataConsistency(nn.Module):
    '''
    >>> the data-consistency layer: as prescribed by Schlemper et al.
    >>> Enforces the consistency of the measured data, following:
        lambda_mat*f_recon + lambda/(1+lambda)*f_measured
        (lambda_mat)_jj = { 1 if j not in Omega, 
                            1/(1+lambd) if j in Omega}
    '''

    def __init__(self, lambd = -1.):
        '''

        >>> x_recon and x_measured should have dimension [batch_size, 1, image_size, image_size, 2]
        >>> mask: tensor, locations of the undersampled data
        >>> lambd: double, -1 for full data consistency, otherwise lambd >= 0
        '''
        super(DataConsistency, self).__init__()
        # assert( lambd  == None or lambd >= 0)
        self.lambd = lambd
       

    def forward(self, x_recon, x_measured, mask):
        '''
        >>> feedforward data-consistency layer
        >>> x_recon: tensor, the reconstructed data, from the previous neural network steps
        >>> x_measured: tensor, measured data, i.e. undersampled IFFT data
        '''
        #NORMALISATION IS CRUCIAL TO BE CONSISTENT WITH FORWARD_OP AND BACKWARD_OP
        f_recon = torch.fft(x_recon, signal_ndim = 2,normalized=True) 
        f_measured = forward_op_new(x_measured, mask)
        if self.lambd >= 0:
            f =  mask_mult_new(f_recon,1-mask) + 1/(1+self.lambd)*mask_mult_new(f_recon,self.mask)
            f += self.lambd/(1+self.lambd)*f_measured
        else:
            f = mask_mult_new(f_recon,1-mask) + f_measured
       

        return backward_op(f)

    
class ConvBlock(nn.Module):
    '''
    >>> A Convolutional Block, as described in 
        Schlemper, J. et al. " A Deep Cascade of Convolutional Neural Networks for 
        Dynamic MR Image Reconstruction". <https://arxiv.org/pdf/1704.02422.pdf>
    
    >>> Creates num_layers consecutive convolutional layers,
        moving from 2 channels (complex input) to conv_channel channels,
        the last layer gets back to 2 channels.
    >>> Each layer except the last one is followed by a ReLu
    '''

    def __init__(self, num_layers, input_channel, conv_channel, conv_kernel, nonlinearity = 'relu'):
        '''
        >>> num_layers: int, number of layers in the convolutional block
            (will have num_layers -1 convolutional units)
        >>> conv_channel: int, the number of channels for convolutional layers
        >>> conv_kernel: int, the size of kernel size in the convolutional layers
        >>> nonlinearity: str, the nonlinearity
        '''
        super(ConvBlock, self).__init__()
        assert num_layers >= 2
        self.conv_channels = conv_channel
        self.conv_kernels = conv_kernel

        convlayer_list = []
        padding = (conv_kernel - 1) // 2
        
        convlayer_list.append(nn.Conv2d(input_channel, conv_channel, kernel_size = conv_kernel,
                                        stride = 1, padding = padding))
        convlayer_list.append({'relu': nn.ReLU(), 'sigd': nn.Sigmoid(), 
                               'tanh': nn.Tanh()}[nonlinearity])

        
        for idx in range(1,num_layers-1) :
            
            convlayer_list.append(nn.Conv2d(conv_channel, conv_channel, 
                                            kernel_size = conv_kernel,
                                            stride = 1, padding = padding))
            convlayer_list.append({'relu': nn.ReLU(), 'sigd': nn.Sigmoid(),
                                   'tanh': nn.Tanh()}[nonlinearity])
        
        convlayer_list.append(nn.Conv2d(conv_channel, 2, kernel_size = conv_kernel,
                                        stride = 1, padding = padding))
        
        self.convlayer = nn.Sequential(*convlayer_list)
        
        
    def forward(self, x):
        '''
        >>> feedforward a conv-pool layer
        '''
        out = self.convlayer(x)
        return x[:,:2]+out

class ConvBlockDC(nn.Module):
    '''
    >>> A Convolutional Block followed by a Data Consistency step, as described in 
        Schlemper, J. et al. " A Deep Cascade of Convolutional Neural Networks for 
        Dynamic MR Image Reconstruction". <https://arxiv.org/pdf/1704.02422.pdf>
    
    >>> Creates a conv_block with num_layers layers and conv_channel channels
    >>> Follows it with a data-consistency layer
    '''

    def __init__(self, num_layers, input_channel, conv_channel, conv_kernel, nonlinearity = 'relu',lambd=None):
        '''
        >>> num_layers: int, number of layers in the convolutional block 
            (will have num_layers -1 convolutional units)
        >>> conv_channel: int, the number of channels for convolutional layers
        >>> conv_kernel: int, the size of kernel size in the convolutional layers
        >>> nonlinearity: str, the nonlinearity
        '''
        super(ConvBlockDC, self).__init__()
        assert num_layers >= 2
        self.conv_block = ConvBlock(num_layers, input_channel, conv_channel, conv_kernel, nonlinearity)
        self.dc_layer = DataConsistency(lambd)
        
    def forward(self, x,x_meas, mask):
        '''
        >>> feedforward a conv block layer
        >>> takes a 4-d tensor (batch_size, num_channel, im_x, im_y)
            and outputs the same dimension
        '''
        x = self.conv_block(x)
        x = self.dc_layer(to5d(x),to5d(x_meas),mask)
        return x    
    
class MaskLayer2D(nn.Module):
    '''
    >>> Undersamples the Fourier transform of the image given as input.
    '''

    def __init__(self, mask, image_size = 256):
        '''
        >>> mask: tensor, locations of the undersampled data
        >>> lambd: double, -1 for full data consistency, otherwise lambd >= 0
        '''
        super(MaskLayer2D, self).__init__()
        self.image_size = image_size
        assert(torch.numel(mask)==image_size**2)
        self.mask = torch.nn.Parameter(mask)
       
    def get_mask(self):
        return self.mask
    
    def forward(self, x):
        '''
        >>> feedforward Binarization and undersampling
        >>> x: tensor, the ground truth data
        '''
        x = x.view(-1, 1, self.image_size, self.image_size, 2)
        return backward_op(forward_op(x,Binarize.apply(self.mask,.5),image_size=self.image_size),image_size=self.image_size)        

class MaskLayer1D(nn.Module):
    '''
    >>> Undersamples the Fourier transform of the image given as input using a 1D undersampling mask.
    '''

    def __init__(self, mask, image_size = 256):
        '''
        >>> mask: 1D tensor, locations lines of the undersampled data
        >>> lambd: double, -1 for full data consistency, otherwise lambd >= 0
        '''
        super(MaskLayer1D, self).__init__()
        self.image_size = image_size
        assert(torch.numel(mask)==image_size)
        mask = mask.view(image_size,1) # (1, image_size) would make vertical sampling
        self.mask = torch.nn.Parameter(mask)
    def get_mask(self):
        return self.mask.repeat(1,self.image_size)
    def forward(self, x):
        '''
        >>> feedforward Binarization and undersampling
        >>> x: tensor, the ground truth data
        '''
        x = x.view(-1, 1, self.image_size, self.image_size, 2)
        return backward_op(forward_op(x,Binarize.apply(self.mask.repeat(1,self.image_size),.5),
                                      image_size=self.image_size),image_size = self.image_size)      
        
