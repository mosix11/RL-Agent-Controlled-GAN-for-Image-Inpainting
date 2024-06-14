import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from torch.nn.utils.parametrizations import spectral_norm
import numpy as np

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
    
    
class Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim=8, gfv_dim=2048, conv_dim=64):
        super(Generator, self).__init__()
        self.gfv_dim = gfv_dim
        repeat_num = int(np.log2(gfv_dim // 64))  # Calculate repeat_num based on gfv_dim
        mult = 2 ** repeat_num

        layer1 = [
            spectral_norm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)),
            nn.BatchNorm2d(conv_dim * mult),
            nn.ReLU()
        ]

        curr_dim = conv_dim * mult
        layer2 = [
            spectral_norm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU()
        ]

        curr_dim = int(curr_dim / 2)
        layer3 = [
            spectral_norm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU()
        ]

        curr_dim = int(curr_dim / 2)
        layer4 = [
            spectral_norm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)),
            nn.BatchNorm2d(int(curr_dim / 2)),
            nn.ReLU()
        ]

        curr_dim = int(curr_dim / 2)
        last = [
            nn.ConvTranspose2d(curr_dim, gfv_dim, 4, 2, 1),
            nn.Tanh()
        ]

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(128, 'relu')
        self.attn2 = Self_Attn(64, 'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)  # (batch_size, 8, 1, 1)
        out = self.l1(z)  # (batch_size, conv_dim * mult, 4, 4)
        out = self.l2(out)  # (batch_size, conv_dim * mult / 2, 8, 8)
        out = self.l3(out)  # (batch_size, conv_dim * mult / 4, 16, 16)
        out, p1 = self.attn1(out)  # (batch_size, 128, 16, 16)
        out = self.l4(out)  # (batch_size, conv_dim * mult / 8, 32, 32)
        out, p2 = self.attn2(out)  # (batch_size, 64, 32, 32)
        out = self.last(out)  # (batch_size, gfv_dim, 64, 64)
        out = F.adaptive_avg_pool2d(out, (1, 1))  # (batch_size, gfv_dim, 1, 1)
        return out.view(out.size(0), -1), p1, p2  # (batch_size, gfv_dim)





# class Generator(nn.Module):
#     """Generator."""

#     def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64):
#         super(Generator, self).__init__()
#         self.imsize = image_size
#         layer1 = []
#         layer2 = []
#         layer3 = []
#         last = []

#         repeat_num = int(np.log2(self.imsize)) - 3
#         mult = 2 ** repeat_num # 8
#         layer1.append(spectral_norm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
#         layer1.append(nn.BatchNorm2d(conv_dim * mult))
#         layer1.append(nn.ReLU())

#         curr_dim = conv_dim * mult

#         layer2.append(spectral_norm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
#         layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
#         layer2.append(nn.ReLU())

#         curr_dim = int(curr_dim / 2)

#         layer3.append(spectral_norm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
#         layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
#         layer3.append(nn.ReLU())

#         if self.imsize == 64:
#             layer4 = []
#             curr_dim = int(curr_dim / 2)
#             layer4.append(spectral_norm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
#             layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
#             layer4.append(nn.ReLU())
#             self.l4 = nn.Sequential(*layer4)
#             curr_dim = int(curr_dim / 2)

#         self.l1 = nn.Sequential(*layer1)
#         self.l2 = nn.Sequential(*layer2)
#         self.l3 = nn.Sequential(*layer3)

#         last.append(nn.ConvTranspose2d(curr_dim, 3, 4, 2, 1))
#         last.append(nn.Tanh())
#         self.last = nn.Sequential(*last)

#         self.attn1 = Self_Attn( 128, 'relu')
#         self.attn2 = Self_Attn( 64,  'relu')

#     def forward(self, z):
#         z = z.view(z.size(0), z.size(1), 1, 1)
#         out=self.l1(z)
#         out=self.l2(out)
#         out=self.l3(out)
#         out,p1 = self.attn1(out)
#         out=self.l4(out)
#         out,p2 = self.attn2(out)
#         out=self.last(out)

#         return out, p1, p2

class Discriminator(nn.Module):
    """Discriminator."""

    def __init__(self, gfv_dim=2048, conv_dim=64):
        super(Discriminator, self).__init__()
        repeat_num = int(np.log2(gfv_dim // 64))  # Calculate repeat_num based on gfv_dim

        layer1 = [
            spectral_norm(nn.Conv2d(gfv_dim, conv_dim * 8, 4, 2, 1)),
            nn.LeakyReLU(0.1)
        ]

        curr_dim = conv_dim * 8
        layer2 = [
            spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1)
        ]

        curr_dim = curr_dim * 2
        layer3 = [
            spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1)
        ]

        curr_dim = curr_dim * 2
        layer4 = [
            spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)),
            nn.LeakyReLU(0.1)
        ]

        curr_dim = curr_dim * 2
        last = [
            nn.Conv2d(curr_dim, 1, 4),
            nn.Sigmoid()
        ]

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        x = x.view(x.size(0), x.size(1), 1, 1)  # (batch_size, gfv_dim, 1, 1)
        out = self.l1(x)  # (batch_size, conv_dim * 8, 4, 4)
        out = self.l2(out)  # (batch_size, conv_dim * 16, 8, 8)
        out = self.l3(out)  # (batch_size, conv_dim * 32, 16, 16)
        out, p1 = self.attn1(out)  # (batch_size, 256, 16, 16)
        out = self.l4(out)  # (batch_size, conv_dim * 64, 32, 32)
        out, p2 = self.attn2(out)  # (batch_size, 512, 32, 32)
        out = self.last(out)  # (batch_size, 1, 1, 1)
        return out.view(out.size(0)), p1, p2  # (batch_size,)



# class Discriminator(nn.Module):
#     """Discriminator, Auxiliary Classifier."""

#     def __init__(self, batch_size=64, image_size=64, conv_dim=64):
#         super(Discriminator, self).__init__()
#         self.imsize = image_size
#         layer1 = []
#         layer2 = []
#         layer3 = []
#         last = []

#         layer1.append(spectral_norm(nn.Conv2d(3, conv_dim, 4, 2, 1)))
#         layer1.append(nn.LeakyReLU(0.1))

#         curr_dim = conv_dim

#         layer2.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
#         layer2.append(nn.LeakyReLU(0.1))
#         curr_dim = curr_dim * 2

#         layer3.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
#         layer3.append(nn.LeakyReLU(0.1))
#         curr_dim = curr_dim * 2

#         if self.imsize == 64:
#             layer4 = []
#             layer4.append(spectral_norm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
#             layer4.append(nn.LeakyReLU(0.1))
#             self.l4 = nn.Sequential(*layer4)
#             curr_dim = curr_dim*2
#         self.l1 = nn.Sequential(*layer1)
#         self.l2 = nn.Sequential(*layer2)
#         self.l3 = nn.Sequential(*layer3)

#         last.append(nn.Conv2d(curr_dim, 1, 4))
#         self.last = nn.Sequential(*last)

#         self.attn1 = Self_Attn(256, 'relu')
#         self.attn2 = Self_Attn(512, 'relu')

#     def forward(self, x):
#         out = self.l1(x)
#         out = self.l2(out)
#         out = self.l3(out)
#         out,p1 = self.attn1(out)
#         out=self.l4(out)
#         out,p2 = self.attn2(out)
#         out=self.last(out)

#         return out.squeeze(), p1, p2


class LatentGAN(nn.Module):
    
    def __init__(self, z_dim=8, GFV_dim=2048, lambda_gp=10) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.GFV_dim = GFV_dim
        self.lambda_gp = lambda_gp
        
        self.generator = Generator(z_dim, GFV_dim)
        self.discriminator = Discriminator(GFV_dim)
        
        
    
    def forward(self):
        pass
    
    
    def D_training_step(self, GFV_real, latent_z):
        GFV_fake, _, _ = self.generator(latent_z)
        
        d_out_real, _, _ = self.discriminator(GFV_real)
        d_out_fake, _, _ = self.discriminator(GFV_fake)
        
        loss = self.D_loss(GFV_real, GFV_fake, d_out_real, d_out_fake)
        return loss
    
    def G_training_step(self, latent_z):
        GFV_fake, _, _ = self.generator(latent_z)
        d_out_fake,_,_ = self.discriminator(GFV_fake)
        loss = self.G_loss(d_out_fake)
        return loss
    
    
    # wgan-gp loss for discriminator
    def D_loss(self, GFV_real, GFV_fake, d_out_real, d_out_fake):
        adv_loss = self.discriminator_loss(d_out_real, d_out_fake)
        gp_loss = self.gradient_penalty(GFV_real, GFV_fake)
        return adv_loss + self.lambda_gp * gp_loss
        
    def G_loss(self, d_out_fake):
        return self.generator_loss(d_out_fake)
    
    def discriminator_loss(self, real_scores, fake_scores):
        return -torch.mean(real_scores) + torch.mean(fake_scores)
    
    def generator_loss(self, fake_scores):
        return -torch.mean(fake_scores)
    
    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real_data.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)

        interpolated_scores = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        return ((gradients_norm - 1) ** 2).mean()
    
    def reconstruction_loss(real_features, generated_features):
        return F.mse_loss(real_features, generated_features)
    
    
    def generate(self, latent_z):
        self.eval()
        with torch.no_grad():
            GFV, _, _ = self.generator(latent_z)
            return GFV