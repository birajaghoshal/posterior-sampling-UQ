from cascade.model import CascadeNet
import torch
import numpy as np
from torch.utils import data
from utils import *
from mri_utils import *
from skimage.measure import compare_ssim as ssim

G = CascadeNet().cuda() # input channels:  Z(X) x Y -> output channel X
G.eval()
idx=1999
ma = 8
G.load_state_dict(torch.load('./parameters/checkpoint%d/G.pt'%(idx)))

psnrs = []
ssmis = []
mri_test = MRIDataSet('../data/CORRECTED_COMBINED_bilkent_20_validation.mat', mask=ma)#,external_mask='../data/mask_50.pt')
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
    mean = np.mean(outps,0)
    mean = np.sqrt(mean[0]**2+mean[1]**2)
    x = torch.sqrt(x[0,0]**2+x[0,1]**2)
    psnrs.append(psnr(torch.from_numpy(mean),x).cpu().numpy())
    ssmis.append(ssim(x.numpy(), mean, data_range=mean.max()-mean.min()))
    print('epoch %d %d/%d'%(idx ,i, len(dataloader_test)),end=' ')
    print('PSNR: %.4f'%(np.mean(psnrs)), end=' ')
    print('SSIM: %.4f'%(np.mean(ssmis)))
