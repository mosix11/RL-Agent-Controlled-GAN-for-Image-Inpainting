import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
        
class Encoder(nn.Module):
    def __init__(self, GFV_dim=1024, dropout=0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # (128x128x3) -> (64x64x64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (64x64x64) -> (32x32x128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (32x32x128) -> (16x16x256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # (16x16x256) -> (8x8x512)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)  # (8x18x512) -> (4x4x1024)
        self.fc = nn.Linear(4 * 4 * 1024, GFV_dim)  # Latent vector
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.batchnorm_fc = nn.BatchNorm1d(GFV_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = F.relu(self.batchnorm1(self.conv1(X)))
        X = F.relu(self.batchnorm2(self.conv2(X)))
        X = F.relu(self.batchnorm3(self.conv3(X)))
        X = F.relu(self.batchnorm4(self.conv4(X)))
        X = F.relu(self.conv5(X))
        X = X.view(X.size(0), -1)  # Flatten
        X = self.dropout(X)
        X = self.fc(X)
        X = self.batchnorm_fc(X)
        return X

class Decoder(nn.Module):
    def __init__(self, GFV_dim=1024, dropout=0.5) -> None:
        super().__init__()
        self.fc = nn.Linear(GFV_dim, 4 * 4 * 1024)  # Latent vector to 4x4x1024
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)  # (4x4x1024) -> (8x8x512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # (8x8x512) -> (16x16x256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # (16x16x256) -> (32x32x128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # (32x32x128) -> (64x64x64)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)  # (64x64x64) -> (128x128x3)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        X = self.fc(X)
        X = X.view(X.size(0), 1024, 4, 4)  # Reshape
        X = F.relu(self.batchnorm1(self.deconv1(X)))
        X = F.relu(self.batchnorm2(self.deconv2(X)))
        X = F.relu(self.batchnorm3(self.deconv3(X)))
        X = F.relu(self.batchnorm4(self.deconv4(X)))
        X = torch.tanh(self.deconv5(X))  # Output with tanh to keep pixel values between -1 and 1
        return X   
    
class AutoEncoder(nn.Module):
    def __init__(self, GFV_dim=1024) -> None:
        super().__init__()
        self.GFV_dim = GFV_dim
        self.encoder = Encoder(GFV_dim)
        self.decoder = Decoder(GFV_dim)
        
        self.perceptual_loss = ResNetPerceptualLoss()
        self.tv_loss = TVLoss(weight=1e-5)
        self.mse_loss = nn.MSELoss(reduction='mean')
        
    def forward(self, X):
        GFV = self.encoder(X)
        output = self.decoder(GFV)
        return output
    
    def training_step(self, batch):
        l = self.loss(self(batch), batch)
        return l
    
    def validation_step(self, batch):
        Y_hat = self(batch)
        return self.loss(Y_hat, batch)
    
    def loss(self, Y_hat, Y):
        loss_mse = self.mse_loss(Y_hat, Y)
        loss_perceptual = self.perceptual_loss(Y_hat, Y)
        loss_tv = self.tv_loss(Y_hat)
        loss = loss_mse + 0.1 * loss_perceptual + loss_tv
        # loss = loss_mse + 0.1 * loss_perceptual
        return loss
        
    def get_GFV(self, X):
        return self.encoder(X)
    
    def decode_GFV(self, GFV):
        return self.decoder(GFV)
    
    def predict(self, X):
        self.eval()
        with torch.no_grad():
            if len(X.shape) == 4:
                return self(X)
            elif len(X.shape) == 3:
                X = X.unsqueeze(0)
                preds = self(X)
                return preds.squeeze(0)
    
    
    
class PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(PerceptualLoss, self).__init__()
        self.vgg = torchvision.models.vgg16(pretrained=True).features.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layers = layers if layers is not None else [3, 8, 15, 22]

    def forward(self, x, y):
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += F.mse_loss(x, y)
        return loss
    
class ResNetPerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(ResNetPerceptualLoss, self).__init__()
        resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet_layers = nn.ModuleList(list(resnet.children())[:-2]).eval()  # Exclude the last fully connected layer
        for param in self.resnet_layers.parameters():
            param.requires_grad = False
        self.layers = layers if layers is not None else [3, 4, 5, 6]

    def forward(self, x, y):
        loss = 0.0
        for i, layer in enumerate(self.resnet_layers):
            x = layer(x)
            y = layer(y)
            if i in self.layers:
                loss += F.mse_loss(x, y)
        return loss
    
class TVLoss(nn.Module):
    def __init__(self, weight):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
    
    

        
# class SelfAttention(nn.Module):
#     def __init__(self, in_dim):
#         super(SelfAttention, self).__init__()
#         self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         batch_size, C, width, height = x.size()
#         proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(batch_size, -1, width * height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = F.softmax(energy, dim=-1)
#         proj_value = self.value_conv(x).view(batch_size, -1, width * height)
        
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, C, width, height)
        
#         out = self.gamma * out + x
#         return out

# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
#         self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
#         self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
#         self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)
#         self.fc = nn.Linear(8 * 8 * 1024, 1024)
#         self.batchnorm1 = nn.BatchNorm2d(64)
#         self.batchnorm2 = nn.BatchNorm2d(128)
#         self.batchnorm3 = nn.BatchNorm2d(256)
#         self.batchnorm4 = nn.BatchNorm2d(512)
#         self.attn1 = SelfAttention(128)
#         self.attn2 = SelfAttention(256)
#         self.attn3 = SelfAttention(512)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x1 = F.relu(self.batchnorm1(self.conv1(x)))
#         x2 = F.relu(self.batchnorm2(self.conv2(x)))
#         x2 = self.attn1(x2)
#         x3 = F.relu(self.batchnorm3(self.conv3(x2)))
#         x3 = self.attn2(x3)
#         x4 = F.relu(self.batchnorm4(self.conv4(x3)))
#         x4 = self.attn3(x4)
#         x5 = F.relu(self.conv5(x4))
#         x5 = x5.view(x5.size(0), -1)
#         x5 = self.dropout(x5)
#         x5 = self.fc(x5)
#         return x5, [x1, x2, x3, x4]

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.fc = nn.Linear(1024, 8 * 8 * 1024)
#         self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
#         self.batchnorm1 = nn.BatchNorm2d(512)
#         self.batchnorm2 = nn.BatchNorm2d(256)
#         self.batchnorm3 = nn.BatchNorm2d(128)
#         self.batchnorm4 = nn.BatchNorm2d(64)
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x, skip_connections):
#         x = self.fc(x)
#         x = x.view(x.size(0), 1024, 8, 8)
#         x = F.relu(self.batchnorm1(self.deconv1(x)) + skip_connections[3])
#         x = F.relu(self.batchnorm2(self.deconv2(x)) + skip_connections[2])
#         x = F.relu(self.batchnorm3(self.deconv3(x)) + skip_connections[1])
#         x = F.relu(self.batchnorm4(self.deconv4(x)) + skip_connections[0])
#         x = torch.sigmoid(self.deconv5(x))
#         return x
