from discriminator.dis import Dis
from cascade.model import CascadeNet
import torch
import numpy as np
from torch.utils import data
from mri_utils import *
from utils import *
import os

D = Dis(6).cuda()# input channels X x Y -> output scalar R
G = CascadeNet().cuda() # input channels:  Z x Y -> output channel X

hyperp = {'batch_size': 10, 'epochs': 2000}

mri_train = MRIDataSet('../data/CORRECTED_COMBINED_bilkent_60_training.mat')
loader = DataLoader(mri_train, batch_size=hyperp['batch_size'],shuffle=True)
D_solver = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=0.0001)
G_solver = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=0.0001)

idx = -1
if idx!=-1:
    D.load_state_dict(torch.load('./parameters/checkpoint%d//D.pt'%(idx)))
    G.load_state_dict(torch.load('./parameters/checkpoint%d/G.pt'%(idx)))
    D_solver.load_state_dict(torch.load('./parameters/checkpoint%d//D_solver.pt'%(idx)))
    G_solver.load_state_dict(torch.load('./parameters/checkpoint%d/G_solver.pt'%(idx)))
    
G.train()
D.train()
for e in range(idx+1,hyperp['epochs']):
    for i, (gt, observed, mask) in enumerate(loader):
        z1 = torch.rand((observed.size(0), 2, 256, 256)).cuda()
        z2 = torch.rand((observed.size(0), 2, 256, 256)).cuda()
        gt = gt.cuda()
        observed = observed.cuda()
        mask = mask.cuda()
        x_posterior_1 = G(torch.cat((observed, z1), 1), mask)
        x_posterior_2 = G(torch.cat((observed, z2), 1), mask)

        ### channel shuffle
        rand_num = np.random.randint(0, 2, 1)
        if rand_num == 0:
            x_expect = torch.cat([gt, x_posterior_1], 1)
        elif rand_num == 1:
            x_expect = torch.cat([x_posterior_1, gt], 1)
        else:
            raise ValueError('sth. wrong with rand_num')
        d_true = D(torch.cat([x_expect, observed], 1))

        x_posterior_concat = torch.cat([x_posterior_1, x_posterior_2], 1)
        d_generated = D(torch.cat([x_posterior_concat, observed], 1))
        if (i+1) % 5 != 0:
            d_loss = torch.mean(d_true) - torch.mean(d_generated)
            d_drift = 0.001 * torch.mean(torch.pow(d_true, 2))
            d_grad = grad_penalty(D, x_expect, x_posterior_concat, observed)
            d_total_loss = d_loss + d_drift + d_grad
            D.zero_grad()
            d_total_loss.backward()
            D_solver.step()
        if (i+1) % 5 == 0:
            g_loss = torch.mean(d_generated)
            G.zero_grad()
            g_loss.backward()
            G_solver.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [drift loss: %f][grad loss: %f]"
                % (e, hyperp['epochs'], i, len(loader), d_loss.item(), g_loss.item(), d_drift.item(), d_grad.item())
            )
    if (e+1)%100==0:
        os.mkdir('./parameters/checkpoint%d'%(e))
        torch.save(D.state_dict(),'./parameters/checkpoint%d/D.pt'%(e))
        torch.save(G.state_dict(),'./parameters/checkpoint%d/G.pt'%(e))
        torch.save(D_solver.state_dict(),'./parameters/checkpoint%d/D_solver.pt'%(e))
        torch.save(G_solver.state_dict(),'./parameters/checkpoint%d/G_solver.pt'%(e))
