from cascade.model import CascadeNet
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from utils import *
from mri_utils import *

G = CascadeNet(input_channel=4).cuda()
idx =1999
G.load_state_dict(torch.load('../posterior_sampling_cascade/parameters/checkpoint%d/G.pt'%(idx)))
G = G.eval()

def uncertainty_1d(y, mask):
    '''
    >>> y: subsampled image(spatial domain) (1, 2, 256, 256)
    >>> mask: 2d mask(256, 256)
    return
    >>> uncertainty along the vertical axis (256, )
    '''
    fs = []
    for i in range(10):
        zv = torch.rand((1, 2, 256, 256)).cuda()
        observedv = y.cuda()
        inpv = torch.cat((observedv, zv), 1)
        fs.append(torch.fft(to5d(G(inpv, torch.unsqueeze(mask, 0).cuda()).cpu().detach()), signal_ndim = 2,normalized=True))

    fs = torch.cat(fs, 0)
    f_var = torch.var(fs, 0)
    f_var = f_var[0,:,:,0] +  f_var[0,:,:,1]
#     f_std = torch.sqrt(f_var)
    return torch.sum(f_var, 1)


def apply_mask(x, mask):
    '''
    >>> x : full scan MRI image(spatial domain) (1, 2, 256, 256)
    >>> mask: 2d mask (H, W)

    return
    >>> subsampled image(spatial domain) (1,2, 256, 256)
    '''
    return backward_op(forward_op(to5d(x), mask))

def adapt_mask(x):
    idx = [] # indices where a line will be put
    masks = []
    uncers = []
    mask = torch.zeros((256,256))
    rate = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    line_num = np.round(256*rate).astype('int')

    for i in range(128):
        y = apply_mask(x, mask)
        uncer = uncertainty_1d(y, mask)
        idx.append(torch.argmax(uncer).item())
        mask[idx, :] = 1
        if i+1 in line_num:
            new_mask = torch.zeros((256,256))
            new_mask[idx, :] = 1
            masks.append(new_mask)
            uncers.append(uncer.sum().item())
    return masks, uncers
