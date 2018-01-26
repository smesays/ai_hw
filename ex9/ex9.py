import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.autograd import Variable

import matplotlib.pyplot as plt
#from tqdm import tqdm

# if missing, install with pip install easydict
plt.rcParams['figure.figsize'] = (12.0, 6.0) # set default size of plots
plt.rcParams['image.cmap'] = 'gray'

#Config
CUDA = torch.cuda.is_available()
TEST_BATCH_SIZE = 10
BATCH_SIZE = 128
LR = 0.001
MOMENTUM = 1e-5
LOG_INTERVAL = 100
EPOCHS = 101
# switch between denoising (True) and plain autoencoder (False)
IS_NOISY = False

# Functions and Classes
class FashionMNIST(datasets.MNIST):
    """ Fashion-MNIST dataset: https://github.com/zalandoresearch/fashion-mnist"""
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
        
def add_white_noise(x, factor=0.5, stddev=1):
    # add white noise to tensor
    noise = x.clone().normal_(0, stddev)
    return x + noise * factor

class AddWhiteNoise(object):
    def __init__(self, stddev=1, noise_factor=0.5):
        self.stddev = stddev
        self.factor = noise_factor

    def __call__(self, tensor):
        """ add white noise to tensor 
        (this needs to be done before normalization!)
        
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """   
        tensor = add_white_noise(tensor, factor=self.factor, stddev=self.stddev)
        return tensor.clamp_(0, 1) # clip boundaries to range [0, 1]
        
def to_img(x, shape=(28, 28)):
    x = 0.5 * (x + 1) # move to range [0, 1]
    x = x.clamp(0, 1) # clip boundaries
    x = x.view(x.size(0), 1, shape[0], shape[1]) # reshape
    return x
    
# Data Preparation
train_mean, train_std = (0.5,), (0.5,) #(0.28604,), (0.35302,)
train_loader = DataLoader(
    FashionMNIST('./data_fmnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(train_mean, train_std)  
                   ])),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    FashionMNIST('./data_fmnist', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(train_mean, train_std)  
                   ])),
    batch_size=TEST_BATCH_SIZE, shuffle=False)
    
# display a batch using make grid
'''
in_img = next(iter(train_loader))[0]
inputs = torch.stack([UnNormalize(train_mean, train_std)(in_img[i]) for i in range(in_img.shape[0])])
noisy_inputs = AddWhiteNoise(noise_factor=0.5)(inputs.clone())
plt.imshow(transforms.ToPILImage()(make_grid(noisy_inputs, nrow=16)))
plt.axis('off')
plt.show()
plt.imshow(transforms.ToPILImage()(make_grid(inputs, nrow=16)))
plt.axis('off')
plt.show()
'''
#Net Architecture
class Autoencoder(nn.Module):
    def __init__(self, input_shape=(28, 28)):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(input_shape), 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.ReLU(True),
            nn.Linear(16, 8),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 8),  #(8, 16)
            nn.ReLU(True),
            nn.Linear(8, 32), #(16, 32)
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True), 
            nn.Linear(128, np.prod(input_shape)), 
            nn.Tanh(), # mapping output to -1, 1 range
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
#Training
# create the model and move to GPU if available
model = Autoencoder()
if CUDA:
    model.cuda()

# initialize the optimzation algorithm
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=MOMENTUM)

# use mean squared error loss as reconstruction loss
criterion = nn.MSELoss()

# define train loop
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # reshape to vector
        data = data.view(data.size(0), -1)
        # move to cuda if available
        if CUDA:
            data = data.cuda()
        # add noise
        if IS_NOISY:
            data_in = add_white_noise(data.clone())
            data_in.clamp_(-train_mean[0]/train_std[0], (1-train_mean[0])/train_std[0])
        else:
            data_in = data
        # convert to Variable
        data, data_in = Variable(data), Variable(data_in)
        # forward: evaluate with model
        output = model(data_in)
        loss = criterion(output, data)
        # backward: compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

# define test loop
def test(show_plot=False):
    model.eval()
    test_loss = 0
    for i, (data, target) in enumerate(test_loader):
        # reshape to vector
        data = data.view(data.size(0), -1)
        # move to cuda if available
        if CUDA:
            data = data.cuda()
        # add noise
        if IS_NOISY:
            data_in = add_white_noise(data.clone())
            data_in.clamp_(-train_mean[0]/train_std[0], (1-train_mean[0])/train_std[0])
        else:
            data_in = data
        # convert to Variable
        data, data_in = Variable(data, volatile=True), Variable(data_in, volatile=True)
        # forward: evaluate with model
        output = model(data_in)
        test_loss += nn.MSELoss(size_average=False)(output, data).data[0] # sum up batch loss
	'''        
        if show_plot == 1 and i == 0:
            inp = make_grid(to_img(data_in.cpu().data))
            out = make_grid(to_img(output.cpu().data))
            target = make_grid(to_img(data.cpu().data))
            if IS_NOISY:
                plt.imshow(transforms.ToPILImage()(inp))
                plt.axis('off')
                plt.show()
            plt.imshow(transforms.ToPILImage()(out))
            plt.axis('off')
            plt.show()
            plt.imshow(transforms.ToPILImage()(target))
            plt.axis('off')
            plt.show()
	'''        
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        
    return test_loss
    
best_loss = 1000. # something large
# run training
test(show_plot=True)
for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test_loss = test(show_plot=epoch%10 == 1)
    if test_loss < best_loss:
        best_loss = test_loss
        if IS_NOISY:
            torch.save(model.state_dict(), 'best_fc_denoising_autoencoder.pth')
        else:
            torch.save(model.state_dict(), 'best_fc_autoencoder.pth')
# 
print('The best(lowest) achieved loss: {:.3f}\n'.format(best_loss))

