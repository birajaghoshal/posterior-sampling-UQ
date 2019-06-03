from cascade.model import CascadeNet
import torch
import numpy as np
from torch.utils import data
from utils import *
from mri_utils import *

G = CascadeNet().cuda() # input channels:  Z(X) x Y -> output channel X
G.eval()
idx=1999
ma = 8
G.load_state_dict(torch.load('./parameters/checkpoint%d/G.pt'%(idx)))

uncers = []
mri_test = MRIDataSet('../data/CORRECTED_COMBINED_bilkent_20_validation.mat', external_mask='../data/uncertainty_descent_2/mask_10.pt')
dataloader_test = DataLoader(mri_test, batch_size=1)

def psnr(img, r_img):
    return 10*torch.log10(torch.max(r_img)**2/torch.mean((img-r_img)**2))

for i, (x, y, mask) in enumerate(dataloader_test):
    outps = []
    for j in range(10):
        zv = torch.rand((1, 2, 256, 256)).cuda()
        observedv = y.reshape((1,2,256,256)).cuda()
        inpv = torch.cat((observedv, zv), 1)
        outps.append(G(inpv, mask.cuda()).cpu().detach().numpy())

    outps = np.concatenate(outps)
    var = np.var(outps,0)
    uncers.append(np.sum(var))
    print('epoch %d %d/%d'%(idx ,i, len(dataloader_test)),end=' ')
    print('uncer: %.4f'%(np.mean(uncers)))
