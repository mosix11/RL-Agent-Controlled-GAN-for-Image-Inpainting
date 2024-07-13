import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
import torchvision
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import socket
import datetime

from src.dataset import CelebA
from src.models import AutoEncoder, LatentGAN, DDPG
from src.trainers import AETrainer, GANTrainer, RLTrainer

from src.utils import utils, nn_utils

cpu = nn_utils.get_cpu_device()
gpu = nn_utils.get_gpu_device()

test_dir = Path('outputs/test')
if not test_dir.exists():
    os.mkdir(test_dir)
writer = SummaryWriter(test_dir)



class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, x, epsilon=1e-8):
        return x * (((x**2).mean(dim=1, keepdim=True) + epsilon).rsqrt())
    
def Upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x

class ConstrainedLayer(nn.Module):
    r"""
    A handy refactor that allows the user to:
    - initialize one layer's bias to zero
    - apply He's initialization at runtime
    """

    def __init__(self,
                 module,
                 equalized=True,
                 lrMul=1.0,
                 initBiasToZero=True):
        r"""
        equalized (bool): if true, the layer's weight should evolve within
                         the range (-1, 1)
        initBiasToZero (bool): if true, bias will be initialized to zero
        """

        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if initBiasToZero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lrMul
            self.weight = getLayerNormalizationFactor(self.module) * lrMul

    def forward(self, x):

        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x

def getLayerNormalizationFactor(x):
    r"""
    Get He's constant for the given layer
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """
    size = x.weight.size()
    fan_in = np.prod(size[1:])

    return np.sqrt(2.0 / fan_in)

class EqualizedConv2d(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 kernelSize,
                 padding=0,
                 bias=True,
                 **kwargs):
        r"""
        A nn.Conv2d module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            kernelSize (int): size of the convolutional kernel
            padding (int): convolution's padding
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Conv2d(nChannelsPrevious, nChannels,
                                            kernelSize, padding=padding,
                                            bias=bias),
                                  **kwargs)

class EqualizedLinear(ConstrainedLayer):

    def __init__(self,
                 nChannelsPrevious,
                 nChannels,
                 bias=True,
                 **kwargs):
        r"""
        A nn.Linear module with specific constraints
        Args:
            nChannelsPrevious (int): number of channels in the previous layer
            nChannels (int): number of channels of the current layer
            bias (bool): with bias ?
        """

        ConstrainedLayer.__init__(self,
                                  nn.Linear(nChannelsPrevious, nChannels,
                                  bias=bias), **kwargs)

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class InversionNet(nn.Module):
    def __init__(self,
                 depthScale0,
                 initBiasToZero=True,
                 leakyReluLeak=0.2,
                 latentVectorSize=512,
                 dimInput=3,
                 equalizedlR=True):
        r"""
        Build an inversion network for a progressive GAN model

        Args:
            - depthScale0 (int): depth of the lowest resolution scales
            - initBiasToZero (bool): should we set the bias to zero when a new scale is added
            - leakyReluLeak (float): leakyness of the leaky relu activation function
            - latentVectorSize (int): size of the latent vector to be output
            - dimInput (int): 3 (RGB input), 1 (grey-scale input)
            - equalizedlR (bool): use equalized learning rate
        """
        super(InversionNet, self).__init__()

        # Initialization parameters
        self.initBiasToZero = initBiasToZero
        self.equalizedlR = equalizedlR
        self.dimInput = dimInput

        # Initialize the scales
        self.scalesDepth = [depthScale0]
        self.scaleLayers = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()

        # Initialize the last layer
        self.initDecisionLayer(latentVectorSize)

        # Layer 0
        self.groupScaleZero = nn.ModuleList()
        self.fromRGBLayers.append(EqualizedConv2d(dimInput, depthScale0, 1,
                                                  equalized=equalizedlR,
                                                  initBiasToZero=initBiasToZero))

        self.groupScaleZero.append(EqualizedConv2d(depthScale0, depthScale0,
                                                   3, padding=1,
                                                   equalized=equalizedlR,
                                                   initBiasToZero=initBiasToZero))

        self.groupScaleZero.append(EqualizedLinear(depthScale0 * 16,
                                                   depthScale0,
                                                   equalized=equalizedlR,
                                                   initBiasToZero=initBiasToZero))

        # Initialize the upscaling parameters
        self.alpha = 0

        # Leaky relu activation
        self.leakyRelu = torch.nn.LeakyReLU(leakyReluLeak)

    def addScale(self, depthNewScale):
        depthLastScale = self.scalesDepth[-1]
        self.scalesDepth.append(depthNewScale)

        self.scaleLayers.append(nn.ModuleList())
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale,
                                                    depthNewScale,
                                                    3,
                                                    padding=1,
                                                    equalized=self.equalizedlR,
                                                    initBiasToZero=self.initBiasToZero))
        self.scaleLayers[-1].append(EqualizedConv2d(depthNewScale,
                                                    depthLastScale,
                                                    3,
                                                    padding=1,
                                                    equalized=self.equalizedlR,
                                                    initBiasToZero=self.initBiasToZero))

        self.fromRGBLayers.append(EqualizedConv2d(self.dimInput,
                                                  depthNewScale,
                                                  1,
                                                  equalized=self.equalizedlR,
                                                  initBiasToZero=self.initBiasToZero))

    def setNewAlpha(self, alpha):
        r"""
        Update the value of the merging factor alpha

        Args:
            - alpha (float): merging factor, must be in [0, 1]
        """
        if alpha < 0 or alpha > 1:
            raise ValueError("alpha must be in [0,1]")

        if not self.fromRGBLayers:
            raise AttributeError("Can't set an alpha layer if only the scale 0 is defined")

        self.alpha = alpha

    def initDecisionLayer(self, latentVectorSize):
        self.decisionLayer = EqualizedLinear(self.scalesDepth[0],
                                             latentVectorSize,
                                             equalized=self.equalizedlR,
                                             initBiasToZero=self.initBiasToZero)

    def forward(self, x):
        # Alpha blending
        if self.alpha > 0 and len(self.fromRGBLayers) > 1:
            y = F.avg_pool2d(x, (2, 2))
            y = self.leakyRelu(self.fromRGBLayers[-2](y))

        # From RGB layer
        x = self.leakyRelu(self.fromRGBLayers[-1](x))

        # Caution: we must explore the layers group in reverse order!
        # Explore all scales before 0
        mergeLayer = self.alpha > 0 and len(self.scaleLayers) > 1
        for groupLayer in reversed(self.scaleLayers):
            for layer in groupLayer:
                x = self.leakyRelu(layer(x))
            x = nn.AvgPool2d((2, 2))(x)
            if mergeLayer:
                mergeLayer = False
                x = self.alpha * y + (1 - self.alpha) * x

        # Now the scale 0
        x = self.leakyRelu(self.groupScaleZero[0](x))
        x = x.view(-1, num_flat_features(x))
        x = self.leakyRelu(self.groupScaleZero[1](x))

        # Output the latent vector
        latent_vector = self.decisionLayer(x)
        return latent_vector



model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-256',
                       pretrained=True, useGPU=True)

dataset = CelebA(root_dir=Path('./data').absolute(), img_size=(256, 256), batch_size=256, num_workers=2, mask=False)

inverse_net = InversionNet()
inverse_net.to(gpu)
criterion = nn.MSELoss()
lr = 1e-5
optimizer = optim.Adam(inverse_net.parameters(), lr=lr)


for iterations in range(200000):
    
    z_batch = torch.randn(64, 512, device=gpu)
    with torch.no_grad():
        generated_images = model.test(z_batch)
    z_pred = inverse_net(generated_images)
    
    optimizer.zero_grad()

    loss = criterion(z_pred, z_batch)
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    if (iterations+1) % 1000 == 0:
        print(loss.item())
        
    if (iterations+1) % 4000 == 0:
        print('Ecaluating')
        z_batch = torch.randn((10, 512), device=gpu)
        with torch.no_grad():
            generated_images = model.test(z_batch)
            z_pred = inverse_net(generated_images)
            reconst = model.test(z_pred)
        
        writer.add_images('Z Original', (generated_images + 1) / 2, global_step=0)
        writer.add_images('Z Recons', (reconst + 1) / 2, global_step=0)
        writer.flush()
    
torch.save(inverse_net, './weights/inv.pt')





# grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)
# print(generated_images.shape)
# writer.add_images('Original', (generated_images + 1) / 2, global_step=0)
# writer.flush()

# saving_dir = Path('./weights')
# AE_weights_path = saving_dir.joinpath(Path('AE.pt'))
# lGAN_weights_path = saving_dir.joinpath(Path('lGAN.pt'))


# AE = torch.load(AE_weights_path).to(gpu).eval()
# lgan = torch.load(lGAN_weights_path).to(gpu).eval()

# dataset = CelebA(root_dir=Path('./data').absolute(), batch_size=256, num_workers=2, mask=False)

# Create an instance of the InverseNetwork
# inverse_net = InverseGAN()
# def weights_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)

# inverse_net.apply(weights_init)
# inverse_net.to(gpu)
# # Define a loss function and optimizer
# criterion = nn.MSELoss()
# lr = 1e-5
# optimizer = optim.Adam(inverse_net.parameters(), lr=lr)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

# current_lr = lr
# for epoch in range(2000):
#     # dataloader = dataset.get_train_dataloader()

#     epoch_loss = 0
#     for i in range(100):
#         z_batch = torch.randn((256, 8), device=gpu)
        
#         with torch.no_grad():
#             GFV_fake = lgan.generator(z_batch)
#         z_pred = inverse_net(GFV_fake)
        
#         optimizer.zero_grad()

#         loss = criterion(z_pred, z_batch)

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
    
#     # Print the loss for every epoch
#     print(f'Epoch [{epoch+1}/{2000}], Loss: {epoch_loss/100}')
#     lr_scheduler.step(epoch_loss/100)
#     if not lr_scheduler.get_last_lr() == current_lr:
#         current_lr = lr_scheduler.get_last_lr()
#         print('Learning rate updated. New lr is {}'.format(current_lr))
    
#     if (epoch+1)%50 == 0:
#         z_batch = torch.randn((10, 8), device=gpu)
#         GFV_orig = lgan.generator(z_batch)
#         imgs_orig = AE.decode_GFV(GFV_orig)
        
#         inverse_net.eval()
#         z_pred = inverse_net(GFV_orig)
#         inverse_net.train()
#         GFV_recons = lgan.generator(z_pred)
        
#         val_loss = criterion(z_pred, z_batch)
#         print(f'Epoch [{epoch+1}/{200}], Validation Loss: {val_loss}')
        
#         recons = AE.decode_GFV(GFV_recons)
#         recons = (recons + 1) / 2
#         writer.add_images('Z Original', (imgs_orig + 1) / 2, global_step=0)
#         writer.add_images('Z Recons', recons, global_step=0)
#         writer.flush()
    
# torch.save(inverse_net, './weights/inv.pt')
# inverse_net = torch.load('./weights/inv.pt')

# for epoch in range(5):
#     dataloader = dataset.get_train_dataloader()
#     epoch_loss = 0
#     for i, batch in enumerate(dataloader):  # Assuming you have a DataLoader for your 1024-dimensional data
#         # Get the input 1024-dimensional vector
#         inputs = batch.to(gpu)  # Shape: [batch_size, 1024]
        
#         with torch.no_grad():
#             gfv = AE.get_GFV(inputs)
#         # Zero the parameter gradients
#         optimizer.zero_grad()

#         # Forward pass through the inverse network
#         z_pred = inverse_net(gfv)

#         # Generate the 1024-dimensional vector using the pre-trained generator
#         # with torch.no_grad():
#         outputs = lgan.generator(z_pred)

#         # Compute the loss between the original inputs and the generated outputs
#         loss = criterion(outputs, gfv)

#         # Backward pass and optimize
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
    

#     print(f'Epoch [{epoch+1}/{200}], Loss: {epoch_loss/len(dataloader)}')
#     lr_scheduler.step(epoch_loss/len(dataloader))
#     if not lr_scheduler.get_last_lr() == current_lr:
#         current_lr = lr_scheduler.get_last_lr()
#         print('Learning rate updated. New lr is {}'.format(current_lr))
    
#     if (epoch+1)%5 == 0:
#         samples = None
#         for i, batch in enumerate(dataset.get_val_dataloader()):
#             if i < 1 :
#                 samples = batch
#             else:
#                 break
#         with torch.no_grad():
#             samples = samples.to(gpu)
#             gfv = AE.get_GFV(samples)
#             inverse_net.eval()
#             z = inverse_net(gfv)
#             # print(torch.std_mean(z))
#             inverse_net.train()
#             gen_gfv = lgan.generator(z)
#         val_loss = criterion(gen_gfv, gfv)
#         print(f'Epoch [{epoch+1}/{200}], Validation Loss: {val_loss}')
#         recons = AE.decode_GFV(gen_gfv)
#         recons = (recons + 1) / 2
#         writer.add_images('Original', (samples + 1) / 2, global_step=0)
#         writer.add_images('Recons', recons, global_step=0)
#         writer.flush()



