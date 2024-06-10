import torch
import torch.nn as nn
import torch.nn.functional as F


# class Encoder(nn.modules):
#     def __init__(self) -> None:
#         super().__init__()
        
#     def forward(self, X):
#         pass

# class Decoder(nn.modules):
#     def __init__(self) -> None:
#         super().__init__()
        
#     def forward(self, X):
#         pass

# class AutoEncoder(nn.modules):
#     def __init__(self) -> None:
#         super().__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
        
        
#     def forward(self, X):
#         GFV = self.encoder(X)
#         output = self.decoder(GFV)
#         return output
        
class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)  # (256x256x3) -> (128x128x64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)  # (128x128x64) -> (64x64x128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # (64x64x128) -> (32x32x256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)  # (32x32x256) -> (16x16x512)
        self.conv5 = nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1)  # (16x16x512) -> (8x8x1024)
        self.fc = nn.Linear(8 * 8 * 1024, 1024)  # Latent vector
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        X = F.relu(self.batchnorm1(self.conv1(X)))
        X = F.relu(self.batchnorm2(self.conv2(X)))
        X = F.relu(self.batchnorm3(self.conv3(X)))
        X = F.relu(self.batchnorm4(self.conv4(X)))
        X = F.relu(self.conv5(X))
        X = X.view(X.size(0), -1)  # Flatten
        X = self.dropout(X)
        X = self.fc(X)
        return X

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1024, 8 * 8 * 1024)  # Latent vector to 8x8x1024
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)  # (8x8x1024) -> (16x16x512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # (16x16x512) -> (32x32x256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # (32x32x256) -> (64x64x128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # (64x64x128) -> (128x128x64)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)  # (128x128x64) -> (256x256x3)
        self.batchnorm1 = nn.BatchNorm2d(512)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        X = self.fc(X)
        X = X.view(X.size(0), 1024, 8, 8)  # Reshape
        X = F.relu(self.batchnorm1(self.deconv1(X)))
        X = F.relu(self.batchnorm2(self.deconv2(X)))
        X = F.relu(self.batchnorm3(self.deconv3(X)))
        X = F.relu(self.batchnorm4(self.deconv4(X)))
        X = torch.sigmoid(self.deconv5(X))  # Output with sigmoid to keep pixel values between 0 and 1
        return X   
    
class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, X):
        GFV = self.encoder(X)
        output = self.decoder(GFV)
        return output
    
    def training_step(self, batch):
        l = self.loss(self(batch), batch)
        return l
    
    def validation_step(self, batch):
        Y_hat = self(batch)
        return self.loss(Y_hat, batch), self.accuracy(Y_hat, batch)
    
    def loss(self, Y_hat, Y, averaged=True):
        # print(Y_hat.shape)
        # print(Y.shape)
        # Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        # Y = Y.reshape((-1,))
        return F.mse_loss(
        Y_hat, Y, reduction='mean' if averaged else 'none')
        
        
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
