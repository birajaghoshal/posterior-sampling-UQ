from cascade.model import CascadeNet
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
from utils import *
from mri_utils import *
import os
import torchvision
from skimage.measure import compare_ssim as ssim

G = CascadeNet().cuda() # input channels:  Z(X) x Y -> output channel X
mri_dataset = MRIDataSet('../data/CORRECTED_COMBINED_bilkent_60_training.mat')

idx =1999
G.load_state_dict(torch.load('./parameters/checkpoint%d/G.pt'%(idx)))

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
        fs.append(torch.fft(to5d(G(inpv, torch.unsqueeze(mask, 0).cuda())), signal_ndim = 2,normalized=True).cpu().detach())

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
    return backward_op(forward_op(to5d(x),mask))

def mean_uncertainty(batch, omega_new):
    uncer = 0
    mask_new = torch.zeros((256,256))
    mask_new[omega_new] = 1
    for x in batch:
        x = x.unsqueeze(0)
        y = apply_mask(x, mask_new)
        uncer  += torch.sum(uncertainty_1d(y, mask_new))
    return uncer / batch.shape[0]

omega = [0, 255]
S = list(range(256))
S.remove(0)
S.remove(255)
batchsize = 5
uncers = []

for i in range(126):
    min_unc = 1e20
    new_idx = None
    S_stochastic = np.random.choice(S, (60,), False).tolist()
    batch = mri_dataset.random_get(batchsize)
    for s in S_stochastic:
        omega_new = omega + [s]
        unc = mean_uncertainty(batch, omega_new)
        if unc < min_unc:
            min_unc = unc
            new_idx = s
        print('idx %d unc %.2f min idx %d'%(s, unc, new_idx))
    S.remove(new_idx)
    omega.append(new_idx)
    uncers.append(min_unc)
    print('line %d/126 unc %f'%(i, min_unc))

torch.save(omega, '../data/uncer_greedy_mask.pt')
torch.save(uncers,  '../data/uncer_greedy_uncers.pt')
