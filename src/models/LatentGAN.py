import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from torch.nn.utils.parametrizations import spectral_norm
from torch.nn import Parameter
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
    
    
class Generator(nn.Module):
    """Generator."""

    def __init__(self, z_dim=8, GFV_dim=1024, image_size=64, conv_dim=64):
        super(Generator, self).__init__()
        self.imsize = image_size
        self.z_dim = z_dim
        self.GFV_dim = GFV_dim
        
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 3, 2, 2))) 
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 3, 2, 2)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())


        if self.imsize == 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            curr_dim = int(curr_dim / 2)

       # self.ln = nn.Sequential(*layern)
        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(64, 1, 2, 2, 1)) # curr_dim
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu') #128
        self.attn2 = Self_Attn( 64,  'relu')
        self.size_correct = nn.ConvTranspose1d(26*26, GFV_dim, 1)


    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1) # [B, 8, 1, 1]
        out=self.l1(z)  # [B, 512, 4, 4]
        out=self.l2(out)   # [B, 256, 5, 5]
        out=self.l3(out)  # [B, 128, 7, 7]
        out,p1 = self.attn1(out) # [B, 128, 7, 7]
        out=self.l4(out) # [B, 64, 14, 14]
        out,p2 = self.attn2(out) # [B, 64, 14, 14]
        out=self.last(out) # [B, 1, 26, 26]

        # print('\n\n', out.shape, '\n\n')
        out = out.view(-1, 1, 26*26) # [B, 1, 26*26]
        out = out.transpose(1, 2) # [B, 26*26, 1]
        out = self.size_correct(out) # [B, 1024, 1]
        out = out.squeeze(2) # [B, 1024]
        # out = out.transpose(2, 1) # [B, 1, 1024]
        # out = out.view(-1,1,1,self.GFV_dim)
        return out

class Discriminator(nn.Module):


    def __init__(self, GFV_dim=1024, image_size=64, conv_dim=64):
        super(Discriminator, self).__init__()
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

        last.append(nn.Conv2d(curr_dim, 1, 4))
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
        out=self.last(out) # [B, 1, 1, 1]
        # print('\n\n', out.shape, '\n\n')
        return out.squeeze(2, 3)

    

class LatentGAN(nn.Module):
    
    def __init__(self, z_dim=16, GFV_dim=1024, lambda_gp=10) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.GFV_dim = GFV_dim
        self.lambda_gp = lambda_gp
        
        self.generator = Generator(z_dim, GFV_dim)
        self.discriminator = Discriminator(GFV_dim)
        
        
    
    def forward(self):
        pass
    
    
    def D_training_step(self, GFV_real, latent_z):
        GFV_fake= self.generator(latent_z)
        d_out_real = self.discriminator(GFV_real)
        d_out_fake = self.discriminator(GFV_fake)
        loss = self.D_loss(GFV_real, GFV_fake, d_out_real, d_out_fake)
        return loss
    
    def G_training_step(self, latent_z):
        GFV_fake= self.generator(latent_z)
        d_out_fake = self.discriminator(GFV_fake)
        loss = self.G_loss(d_out_fake)
        return loss
    
    
    # wgan-gp loss for discriminator
    def D_loss(self, GFV_real, GFV_fake, d_out_real, d_out_fake):
        adv_loss = -torch.mean(d_out_real) + torch.mean(d_out_fake)
        # print('\n\n adversarial loss ', adv_loss, '\n\n')
        gp_loss = self.gradient_penalty(GFV_real, GFV_fake)
        # print('\n\n gradient penalty loss ', self.lambda_gp* gp_loss, '\n\n')
        return adv_loss + self.lambda_gp * gp_loss
        # return adv_loss
        
    def G_loss(self, d_out_fake):
        return -torch.mean(d_out_fake)
    
    
    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.size(0)
        epsilon = torch.rand(batch_size, 1, device=real_data.device)
        interpolated = epsilon * real_data + (1 - epsilon) * fake_data
        interpolated.requires_grad_(True)
        interpolated_scores = self.discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores, requires_grad=False).to(real_data.device),
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
            GFV = self.generator(latent_z)
            return GFV