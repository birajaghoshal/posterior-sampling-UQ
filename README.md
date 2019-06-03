# posterior sampling for reconstruction and uncertainty quantification
Posterior sampling (from the paper "[Deep Bayesian Inversion](https://arxiv.org/abs/1811.05910)") for MRI uncertainty quantification and reconstruction

## Problem Formulation
The code here is to solve an inverse problem by conditional WGAN. The optimization problem can be defined below.

$$
\inf_{\theta \in \Theta}
\sup_{\phi \in \Phi}
\mathbb{E}_
{(\mathbf{x},\mathbf{y}) \thicksim \mu,
z \thicksim \eta}
[D_\phi(\mathbf{x},\mathbf{y})
- 
D_\phi(G_{\theta}\mathbf{(z,y)},\mathbf{y})] 
$$
$$s.t. \parallel \partial_\mathbf{x} D_{\phi}(\mathbf{x},\mathbf{y}) \parallel_2 \leq 1$$

Actually, this loss might cause mode collapse and the implementation is different from the loss function. For more details, please refer to [Deep Bayesian Inversion](https://arxiv.org/abs/1811.05910).

## Results
![pixel-wise standard deviation for masks with sampling rate 0.1 and 0.5]('./figures/std.pdf')
