from cascade.model import CascadeNet
import torch
import numpy as np
from torch.utils import data
from mri_utils import *

import sys
sys.path.append('../posterior_sampling_cascade')
from adaptive_mask import adapt_mask, apply_mask


uncers_list = [[], [], [], [], [], [], [], [], []]

rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
mri_test = MRIDataSet('../data/CORRECTED_COMBINED_bilkent_20_validation.mat')
dataloader_test = DataLoader(mri_test, batch_size=1)



for i, (gt, _, _) in enumerate(dataloader_test):
    _, uncers = adapt_mask(gt)

    for j, uncer in enumerate(uncers):
        uncers_list[j].append(uncer)
    print('%d/%d'%(i, len(dataloader_test)), end =' ')
    for k, un in enumerate(uncers_list):
        print('%.2f: %.2f'%(rates[k], np.mean(un)), end=' ')
    print()
