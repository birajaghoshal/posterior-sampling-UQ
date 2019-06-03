import torch
from torch import autograd

def grad_penalty(netD, x_expected, x_posterior_concat, observed):

    epsilon = torch.rand(x_expected.shape[0], 1, 1, 1).cuda()

    x_hat = epsilon * x_expected + (1 - epsilon) * x_posterior_concat

    x_hat = x_hat.cuda()
    x_hat.requires_grad_()

    d_hat = netD(torch.cat((x_hat, observed), 1))

    gradients = autograd.grad(outputs=d_hat, inputs=x_hat,
                              grad_outputs=torch.ones(d_hat.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    assert gradients.shape==(x_expected.shape[0], 4, 256, 256)

    gradients1 = gradients[:,:2].reshape((-1,256,256))
    assert gradients1.shape == (x_expected.shape[0]*2, 256, 256)

    gradients2 = gradients[:,2:].reshape((-1, 256, 256))
    assert gradients2.shape == (x_expected.shape[0]*2, 256, 256)

    gradients = torch.cat([gradients1, gradients2], 0)
    assert gradients.shape == (x_expected.shape[0]*4, 256, 256)

    ddx = torch.sqrt(torch.sum(gradients ** 2, dim=(1,2)))
    assert ddx.shape == (x_expected.shape[0]*4,)
    gradient_penalty = 10 * torch.mean((ddx - 1.0) ** 2)
    return gradient_penalty
