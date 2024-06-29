import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
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

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
    
class InverseGAN(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, GFV_dim=1024, image_size=64, conv_dim=64):
        super(InverseGAN, self).__init__()
        self.imsize = image_size
        self.GFV_dim = GFV_dim
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(1, conv_dim, 3, 2, 2)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 3, 2, 1))) # 4,2,1
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 3, 2, 2)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize == 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 3, 2, 2)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim*2
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.Conv2d(curr_dim, 2, 3))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

        self.size_correct = nn.ConvTranspose1d(GFV_dim, 26*26, 1)

    def forward(self, x):
        x = x.unsqueeze(2) # [B, 1024, 1]
        x = self.size_correct(x) # [B, 26*26, 1]
        x = x.transpose(1, 2) # [B, 1, 26*26]
        x = x.view(-1, 1, 26, 26) # [B, 1, 26, 26]
        
        out = self.l1(x) # [B, 64, 14, 14]
        out = self.l2(out) # [B, 128, 7, 7]
        out = self.l3(out) # [B, 256, 5, 5]
        out,p1 = self.attn1(out) # [B, 256, 5, 5]
        out=self.l4(out) # [B, 512, 4, 4]
        out,p2 = self.attn2(out) # [B, 512, 4, 4]
        out = self.last(out) # [B, 2, 2, 2]
        out = out.view(-1, 8)
        return out

cpu = nn_utils.get_cpu_device()
gpu = nn_utils.get_gpu_device()

saving_dir = Path('./weights')
AE_weights_path = saving_dir.joinpath(Path('AE.pt'))
lGAN_weights_path = saving_dir.joinpath(Path('lGAN.pt'))
test_dir = Path('outputs/test')
if not test_dir.exists():
    os.mkdir(test_dir)
writer = SummaryWriter(test_dir)

AE = torch.load(AE_weights_path).to(gpu).eval()
lgan = torch.load(lGAN_weights_path).to(gpu).eval()

dataset = CelebA(root_dir=Path('./data').absolute(), batch_size=256, num_workers=2, mask=False)

# Create an instance of the InverseNetwork
inverse_net = InverseGAN()
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

inverse_net.apply(weights_init)
inverse_net.to(gpu)
# Define a loss function and optimizer
criterion = nn.MSELoss()
lr = 1e-5
optimizer = optim.Adam(inverse_net.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

current_lr = lr
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
inverse_net = torch.load('./weights/inv.pt')

for epoch in range(5):
    dataloader = dataset.get_train_dataloader()
    epoch_loss = 0
    # for i, batch in enumerate(dataloader):  # Assuming you have a DataLoader for your 1024-dimensional data
        # # Get the input 1024-dimensional vector
        # inputs = batch.to(gpu)  # Shape: [batch_size, 1024]
        
        # with torch.no_grad():
        #     gfv = AE.get_GFV(inputs)
        # # Zero the parameter gradients
        # optimizer.zero_grad()

        # # Forward pass through the inverse network
        # z_pred = inverse_net(gfv)

        # # Generate the 1024-dimensional vector using the pre-trained generator
        # # with torch.no_grad():
        # outputs = lgan.generator(z_pred)

        # # Compute the loss between the original inputs and the generated outputs
        # loss = criterion(outputs, gfv)

        # # Backward pass and optimize
        # loss.backward()
        # optimizer.step()
        # epoch_loss += loss.item()
    
    # Print the loss for every epoch
    # print(f'Epoch [{epoch+1}/{200}], Loss: {epoch_loss/len(dataloader)}')
    # lr_scheduler.step(epoch_loss/len(dataloader))
    # if not lr_scheduler.get_last_lr() == current_lr:
    #     current_lr = lr_scheduler.get_last_lr()
    #     print('Learning rate updated. New lr is {}'.format(current_lr))
    
    if (epoch+1)%5 == 0:
        samples = None
        for i, batch in enumerate(dataset.get_val_dataloader()):
            if i < 1 :
                samples = batch
            else:
                break
        with torch.no_grad():
            samples = samples.to(gpu)
            gfv = AE.get_GFV(samples)
            inverse_net.eval()
            z = inverse_net(gfv)
            # print(torch.std_mean(z))
            inverse_net.train()
            gen_gfv = lgan.generator(z)
        val_loss = criterion(gen_gfv, gfv)
        print(f'Epoch [{epoch+1}/{200}], Validation Loss: {val_loss}')
        recons = AE.decode_GFV(gen_gfv)
        recons = (recons + 1) / 2
        writer.add_images('Original', (samples + 1) / 2, global_step=0)
        writer.add_images('Recons', recons, global_step=0)
        writer.flush()
# dataset = CelebA(root_dir=Path('./data').absolute())
# model = AutoEncoder()
# trainer = AETrainer(max_epochs=50, run_on_gpu=True, do_validation=True)
# trainer.fit(model, dataset)
# train_loader = dataset.get_train_dataloader()
# for img in train_loader:
#     print(img.shape)
#     break


# z_latent = torch.randn((1, 16), device=gpu)
# lgan = LatentGAN()
# lgan.to(gpu)
# gen = lgan.generator
# disc = lgan.discriminator

# gfv = gen(z_latent)
# score = disc(gfv)
# print(score.shape)





# samples = None
# for i, batch in enumerate(dataset.get_train_dataloader()):
#     if i < 1 :
#         # print(batch[0].shape)
#         samples = batch
#     else:
#         break

# m_batch = samples[1]
# m_batch = m_batch.to(gpu)
# recons = AE.predict(m_batch)
# recons = (recons + 1) / 2
# samples = torch.cat([(samples[0] + 1) / 2, (samples[1] + 1) / 2], dim=0)
# writer.add_images('Test Mask', samples, global_step=0)
# writer.add_images('recons', recons, global_step=0)
# writer.flush()


