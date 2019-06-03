import numpy as np
import random
import scipy.stats
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import hdf5storage



"""
    1. Mask generation from random distribution and Gaussian PDF
        a. randomMask
        b. gaussianPDF1D
        c. gaussianMask1D
"""


def randomMask(imSize,rate,direction='horizontal', pattern='1D'):
    # """
    #     Constructs a 1D undersampling mask
    # >>> imSize: 2D list of int, specifies size of mask to be obtained
    # >>> rate: double in [0,1], undersampling rate
    # >>> direction: str, horizontal or vertical
    # >>> pattern: str, '1D' or '2D' [2D: TODO]
    # """
    if np.size(imSize) > 2:
        raise TypeError('imSize must be 2 dimensional')
    
    if 0 > rate or rate > 1:
        raise ValueError('Rate must be between 0 and 1')
        
    if direction!='horizontal' and direction!='vertical':
        raise ValueError('Direction needs to be horizontal or vertical')

    if pattern!='1D' and pattern!='2D':
        raise ValueError('Pattern must be 1D or 2D')


    [x,y] = imSize
    
    mask = np.zeros([x,y])    
    if pattern=='1D':
        dim = x if direction=='horizontal'else y
        n_elems = int(rate*dim)
        ind = random.sample(range(0, dim, 1),n_elems) ##########modified###############
    
        if direction=='horizontal': 
            mask[ind,:] = 1
        else:
            mask[:,ind] = 1
    return torch.from_numpy(mask).float()

def gaussianPDF1D(n,rate,low_phase=4,density_std=0.16, scale_up=0):
    # """
    #     Constructs a 1D Gaussian pdf with specified parameters
    # >>> n:  int, specifies size of pdf
    # >>> rate: double in [0,1], undersampling rate
    # >>> low_phase: even int, number of low frequency phase encodes
    # >>> density_std: double, standard deviation of Gaussian pdf(normalized to [0,1])
    # >>> scale_up: double, additive scaling of Gaussian pdf (to prevent it to be zero on sides)
    # """
    #Various checks to ensure we're not running into too much problems ;)
    if low_phase%2:
        ValueError('Low phases must be an even number.') 

    if n%2:
        ValueError('Spatial dimension of the image must be even.') 
    if low_phase > round(rate*n):
        ValueError('Rate is not achievable with the number of fully sampled phase encodings desired.') 

    mid = n/2

    # Vectorising and normalising the values into the [0, 1] interval. 
    r = np.linspace(0,n-1,n)/(n-1)
    
    #Getting the low phase indices.
    ind = np.ones(n)
    ind[int(low_phase/2):int(n-low_phase/2)] = 0 #################modified##################
    ind = np.nonzero(np.fft.fftshift(ind))
    #Generating and normalising the Gaussian pdf
    pdf = scipy.stats.norm.pdf(r , .5,density_std)
    pdf = pdf+scale_up 
    #Scaling this up like this keeps the ratio between the smallest value of a scale 
    #and of an unscaled distribution more or less constant across different n values.
    pdf = pdf/sum(pdf)
    pdf[ind] = 1

    return pdf,ind
    

def gaussianMask1D(imSize,rate,low_phase=4,density_std=0.16, scale_up=0.5,direction='horizontal'):
    # """
    #     Constructs a Gaussian variable-density mask with 1D undersampling pattern
    # >>> imSize: 2D list of int, specifies size of mask to be obtained
    # >>> rate: double in [0,1], undersampling rate
    # >>> low_phase: even int, number of low frequency phase encodes
    # >>> density_std: double, standard deviation of Gaussian pdf(normalized to [0,1])
    # >>> scale_up: double, additive scaling of Gaussian pdf (to prevent it to be zero on sides)
    # >>> direction: str, horizontal or vertical
    # """
    if np.size(imSize) > 2:
        raise TypeError('imSize must be 2 dimensional')
    
    if 0 > rate or rate > 1:
        raise ValueError('Rate must be between 0 and 1')
        
    if direction!='horizontal' and direction!='vertical':
        raise ValueError('Direction needs to be horizontal or vertical')

    [x,y] = imSize
    
    mask = np.zeros([x,y]) if direction=='horizontal' else np.zeros([y,x])   
    n = int(x if direction=='horizontal' else y)


    [pdf,ind] = gaussianPDF1D(n,rate,low_phase,density_std,scale_up);

    n_samples = int(round(rate*n))

    # Setting the indices which are sampled in the pdf to zero and
    # renormalising it.
    pdf[ind] = 0
    pdf = pdf/sum(pdf)
    mask[ind,:] = 1;
    n_samples = n_samples - np.size(ind)
    ind = np.random.choice(range(0,n), n_samples, p=pdf)
    mask[ind,:]=1;

    if direction=='vertical':
        mask=mask.transpose()
    return torch.from_numpy(np.fft.fftshift(mask)).float()

"""
    2. Data augmentation methods
        a. Elastic transformation
        b. Data generation for MRI with augmentation 

"""
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    im_real = map_coordinates(np.real(image), indices, order=1, mode='reflect').reshape(image.shape)
    im_imag = map_coordinates(np.imag(image), indices, order=1, mode='reflect').reshape(image.shape)
    return TF.to_tensor(im_real[:,:,np.newaxis]), TF.to_tensor(im_imag[:,:,np.newaxis])
   
"""
    3. Data manipulation functions
        a. Change dimensions with to5d and to4d 
            -> Manipulations to make the data compatiable with FFT and ConvNet respectively
        b. tensor_abs -> magnitude tensor
        c. forward_op and backward_op: Undersampled Fourier transform and backward operation.
"""
    
def to5d(x):
    # """
    # >>> INPUT:
    #     >>> x: tensor, should have dimension [batch_size, 2, image_size, image_size]
    # >>> OUTPUT:
    #     >>> tensor of dimension [batch_size, 1, image_size, image_size, 2]
    # """
    return torch.transpose(torch.unsqueeze(x,4),1,4)

def to4d(x,image_size=256):
    # """
    # >>> INPUT:
    #     >>> x: tensor, should have dimension [batch_size, 1, image_size, image_size, 2]
    # >>> OUTPUT:
    #     >>> tensor of dimension [batch_size, 2, image_size, image_size]
    # """
    x =  torch.transpose(x,1,4)
    return x.view(-1,2,image_size,image_size)

def tensor_abs(img):
    """
    # >>> INPUT:
    #     >>> x: tensor, should have dimension [batch_size, 1, image_size, image_size, 2]
    # >>> OUTPUT: Magnitude tensor out of the combination of each complex dimension
    #     >>> tensor of dimension [batch_size, 1, image_size, image_size]
    # """
    return torch.sqrt(img[:,:,:,:,0]**2+img[:,:,:,:,1]**2)

def mask_mult(x,mask):
    # """
    # >>> INPUT:
    #     >>> x: tensor, should have dimension [batch_size, 1, image_size, image_size, 2]
    # >>> OUTPUT: tensor multiplied element-wise in each complex dimension
    #     >>> undersampled tensor, should have dimension [batch_size, 1, image_size, image_size, 2]
    # """
    f_real = torch.unsqueeze(x[:, :, :, :, 0] * mask,4)
    f_imag = torch.unsqueeze(x[:, :, :, :, 1] * mask,4)
    return  torch.cat([f_real, f_imag], dim = 4)

def mask_mult_new(x,mask):
    # """
    # >>> INPUT:
    #     >>> x: tensor, should have dimension [batch_size, 1, image_size, image_size, 2]
    # >>> OUTPUT: tensor multiplied element-wise in each complex dimension
    #     >>> undersampled tensor, should have dimension [batch_size, 1, image_size, image_size, 2]
    # """
    mask = torch.unsqueeze(mask, 1)
    f_real = torch.unsqueeze(x[:, :, :, :, 0] * mask,4)
    f_imag = torch.unsqueeze(x[:, :, :, :, 1] * mask,4)
    return  torch.cat([f_real, f_imag], dim = 4)

def forward_op(x,mask,image_size=256):
    # """
    #     >>> Takes the FFT of a tensor x and undersamples it
    #
    #     >>> x: tensor, should have dimension [batch_size, 1, image_size, image_size, 2]
    #             this is the tensor of which the fft is taken (dimension 2 at the end for real and imaginary)
    #     >>> mask: tensor, the mask of shape [image_size, image_size]
    #
    #     >>> outputs:
    #             >>> a tensor of dimensions [batch_size, 1, image_size, image_size, 2]
    #                 which contains the undersampled fft of x
    # """
    x = x.view(-1, 1, image_size,image_size, 2)
    f_fft = torch.fft(x, signal_ndim = 2,normalized=True)
    return mask_mult(f_fft,mask)

def forward_op_new(x,mask,image_size=256):
    # """
    #     >>> Takes the FFT of a tensor x and undersamples it
    #
    #     >>> x: tensor, should have dimension [batch_size, 1, image_size, image_size, 2]
    #             this is the tensor of which the fft is taken (dimension 2 at the end for real and imaginary)
    #     >>> mask: tensor, the mask of shape [image_size, image_size]
    #
    #     >>> outputs:
    #             >>> a tensor of dimensions [batch_size, 1, image_size, image_size, 2]
    #                 which contains the undersampled fft of x
    # """
    x = x.view(-1, 1, image_size,image_size, 2)
    f_fft = torch.fft(x, signal_ndim = 2,normalized=True)
    return mask_mult_new(f_fft,mask)
    
def backward_op(f,image_size=256):
    # """
    #     >>> Takes the inverse FFT of a tensor f and maps it to the format by which
    #         it will be processed by the subsequent NN
    #
    #     >>> f: tensor, should have dimension [batch_size, 1, image_size, image_size, 2]
    #         contains the undersampled (or not) Fourier spectrum of an image
    #     >>> mask: tensor, the mask of shape [image_size, image_size]
    #
    #     >>> outputs:
    #             >>> a tensor of dimensions [batch_size, 2] which contains the image signal x
    # """
    f.view(-1, 1, image_size,image_size, 2)
    x = torch.ifft(f, signal_ndim = 2,normalized=True)#.float()
    return to4d(x)

class MRIDataSet(Dataset):
    """
        MRI dataset.
        Loads the complex dataset from the given path.
        .mat file contains an x_original variable of size
        (x,y,number training, number slices)
        Then, it applies the following transformation on each channel of the data:
        1. Flipping horizontally with probability .5
        2. Random rotation between -180 and 180, translation [0.2,0.2]
        3. Elastic deformation as prescribed in [Simard2003] - I'm less sure about the relevance of this one
    """

    def __init__(self, mat_file, mask = None,external_mask = None):
        """
        Args:
            mat_file (string): Path to the mat file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.ground_truth = hdf5storage.loadmat(mat_file)['x_original']
        [nx,ny,ntr,ns] = self.ground_truth.shape
        self.ground_truth = self.ground_truth.reshape([nx,ny,ntr*ns])
        self.mask = mask
        self.external_mask = external_mask
    def __len__(self):
        return self.ground_truth.shape[2]*5

    def __getitem__(self, index):
        index = index%self.ground_truth.shape[2]
        sample = self.ground_truth[:,:,index]
        sample = sample/np.max(abs(sample));
        sample = sample[:,:,np.newaxis]
        
        # float32 conversion
        sample_re = np.float32(np.real(sample))
        sample_im = np.float32(np.imag(sample))
        
        # conversion to pil image to be made compatible with affine transform
        sample_re = TF.to_pil_image(sample_re)
        sample_im = TF.to_pil_image(sample_im) 
        
        # horizontal flip
        if random.random() < .5:
            sample_re = TF.hflip(sample_re)
            sample_im = TF.hflip(sample_im)
            
        # affine transformation
        angle, translations, scale, shear = transforms.RandomAffine.get_params([-180,180], [.02,.02], None ,None, 
                                                                               self.ground_truth.shape[0:2])
        sample_re = TF.affine(sample_re,angle, translations, scale,shear)
        sample_im = TF.affine(sample_im,angle, translations, scale,shear)
    
        # conversion to tensor
        sample_re = (TF.to_tensor(sample_re))
        sample_im = (TF.to_tensor(sample_im))
    
        # elastic distortion
        alpha = random.random()*40
        sigma = random.random()*3+5

        sample_re, sample_im = elastic_transform(sample_re[0,:,:].numpy() +1j*sample_im[0,:,:].numpy(),alpha,sigma)
        
        # stack real and imaginary parts.
        # Overall tensor of dimension [1,nx,ny,2]
        sample = torch.stack([sample_re, sample_im],1)
        
        masks = ['gaussianMask1D([256,256],0.1,10,density_std=0.08,scale_up=0.05)',

                 'gaussianMask1D([256,256],0.15,12,density_std=0.12,scale_up=0.05)',

                'gaussianMask1D([256,256],0.2,12,density_std=0.16,scale_up=0.05)',

                'gaussianMask1D([256,256],0.25,14,density_std=0.18,scale_up=0.1)',

                'gaussianMask1D([256,256],0.3,14,density_std=0.18,scale_up=0.15)',

                 'gaussianMask1D([256,256],0.35,16,density_std=0.2,scale_up=0.2)',

                'gaussianMask1D([256,256],0.4,18,density_std=0.2,scale_up=0.25)',

                 'gaussianMask1D([256,256],0.45,20,density_std=0.2,scale_up=0.3)',

                'gaussianMask1D([256,256],0.5,22,density_std=0.2,scale_up=0.35)']
        if self.external_mask == None:
            if self.mask == None:
                idx = np.random.randint(9)
            else:
                idx = self.mask
            mask = eval(masks[idx])
        else:
            mask = torch.load(self.external_mask)
        observed = backward_op(forward_op(to5d(sample),mask))
        
        sample = torch.squeeze(sample, 0)
        observed = torch.squeeze(observed, 0)
        return sample, observed, mask
    
    def random_get(self, batchsize):
        xs = []
        for i in range(batchsize):
            idx = torch.randint(low=0, high=self.__len__(), size=(1,))
            x, _, _ = self.__getitem__(idx)
            xs.append(x.unsqueeze(0))
        return torch.cat(xs)
