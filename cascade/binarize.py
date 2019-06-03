import torch
import torch.nn as nn

class Binarize(torch.autograd.Function):
    '''
        >>> Binarizses the weight of the layer
        >>> The backward method simply returns the input as such, i.e. as
            if no thresholding was applied to the data.
        >>> x: torch.Tensor, the input to be binarized
        >>> th: float, the threshold considered
    '''
    @staticmethod
    def forward(ctx, x,th=0.5):
        x_copy = x.clone()
        x_copy[x>th] = 1
        x_copy[x<=th] = 0
        return x_copy
        #return (torch.sign(x1 - th) + 1.) / 2.
    @staticmethod
    def backward(ctx, x):
        return x, None
        #return (x - th + 1.) / 2., None 

