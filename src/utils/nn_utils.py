import torch
import os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import collections
import utils
from torch.utils.data import Dataset
from typing import Any

def get_gpu_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else : return None
    
def get_cpu_device():
    return torch.device('cpu')

class DataModule():

    def __init__(self, root='./data/', num_workers=8) -> None:
        
        if not os.path.exists(root):
            os.mkdir(root)
        self.root = root
        self.num_workers = num_workers

    
    def get_dataloader(self, train=True):
        pass


    def train_dataloader(self):
        return self.get_dataloader(train=True)
    

    def val_dataloader(self):
        return self.get_dataloader(train=False)


class SimpleListDataset(Dataset):
    
    def __init__(self, data_list:list) -> None:
        super().__init__()
        self.data_list = data_list
        
    def __getitem__(self, index) -> Any:
        return self.data_list[index]
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    
def init_constant(module, const):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, const)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.constant_(module.weight, const)
        nn.init.zeros_(module.bias)

def init_uniform(module, l=0, u=1):
    if type(module) == nn.Linear:
        nn.init.uniform_(module.weight, a=l, b=u)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.uniform_(module.weight, a=l, b=u)
        nn.init.zeros_(module.bias)

def init_normal(module, mean=0, std=0.01):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=mean, std=std)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.normal_(module.weight, mean=mean, std=std)
        nn.init.zeros_(module.bias)

def init_xavier_uniform(module, gain=1):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

def init_xavier_normal(module, gain=1):
    if type(module) == nn.Linear:
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.Conv2d:
        nn.init.xavier_normal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_normal_(module._parameters[param])
    

def lazy_layer_initialization(model, dummy_input, init_method=None):
    model(*dummy_input)
    if init_method is not None:
        model.apply(init_method)

def vgg_block_lazy(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def NiN_block_lazy(out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU())

def NiN_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )
    

class InceptionLazy(nn.Module):
    # c1--c4 are the number of output channels for each branch
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)


class Inception(nn.Module):
    # in_c is the number of input channels to the Inception module
    # c1--c4 are the number of output channels for each branch
    def __init__(self, in_c, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # Branch 1
        self.b1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # Branch 2
        self.b2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.b2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Branch 3
        self.b3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.b3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Branch 4
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.Conv2d(in_c, c4, kernel_size=1)


    def forward(self, X):
        b1 = F.relu(self.b1_1(X))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(X))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(X))))
        b4 = F.relu(self.b4_2(self.b4_1(X)))
        return torch.cat((b1, b2, b3, b4), dim=1)





class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_1x1_conv=False, strides=1) -> None:
        super().__init__()

        if strides != 1 and use_1x1_conv == False:
            raise ValueError("""
                using a stride bigger than 1 results in the shape mismatch between output of convolution layers and input 
                while adding them. use use_1x1_conv=True to transform the input shape and match it with the output of the 
                convolution layers
                """)
        if in_channels != out_channels and use_1x1_conv == False:
            raise Exception("""
                If the number of input channels and output channels are different you have to use the 1x1 convolution layer on the
                inputs to match their shape (channel wise in this case) with the output of residual block.
                """)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1_conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNextBlock(nn.Module):
    """The ResNeXt block."""
    def __init__(self, in_channels, out_channels, groups=32, bottleneck_multiplier=1, use_1x1_conv=False, strides=1):
        super().__init__()

        if strides != 1 and use_1x1_conv == False:
            raise Exception("""
                using a stride bigger than 1 results in the shape mismatch between output of convolution layers and input 
                while adding them. use use_1x1_conv=True to transform the input shape and match it with the output of the 
                convolution layers
                """)
        
        if in_channels != out_channels and use_1x1_conv == False:
            raise Exception("""
                If the number of input channels and output channels are different you have to use the 1x1 convolution layer on the
                inputs to match their shape (channel wise in this case) with the output of residual block.
                """)
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise Exception("""
                Number of input channels and output channels must be divisible by the number of groups.
                """)
        bot_channels = int(round(out_channels * bottleneck_multiplier))
        self.conv1 = nn.Conv2d(in_channels, bot_channels, kernel_size=1, stride=1)
    
        self.conv2 = nn.Conv2d(bot_channels, bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=bot_channels//groups)
        self.conv3 = nn.Conv2d(bot_channels, out_channels, kernel_size=1, stride=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        if use_1x1_conv:
            self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                       stride=strides)
            self.bn4 = nn.BatchNorm2d(out_channels)
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)


# Squeeze and Excitation Block from SE NET
class SEBlock(nn.Module):

    def __init__(self, C, r=16) -> None:
        super().__init__()
        
        self.C = C
        self.r = r
        self.globpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(C, C//r)
        self.fc2 = nn.Linear(C//r, r)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, X):
        # X.shape = [N, C, H, W] 
        Y = self.globpool(X)
        Y = torch.flatten(Y, 1) 
        Y = self.relu(self.fc1(Y))
        Y = self.sigmoid(self.fc2(Y))
        # Y.shape = [N, C, 1, 1] 
        Y = Y[:, :, None, None]
        return X * Y


# TODO add SEBlock to the residual block
# class ResidualBlockWithSEBlock(nn.Module):

#     def __init__(self, in_channels, out_channels, use_1x1_conv=False, strides=1) -> None:
#         super().__init__()

#         if strides != 1 and use_1x1_conv == False:
#             raise ValueError("""
#                 using a stride bigger than 1 results in the shape mismatch between output of convolution layers and input 
#                 while adding them. use use_1x1_conv=True to transform the input shape and match it with the output of the 
#                 convolution layers
#                 """)
#         if in_channels != out_channels and use_1x1_conv == False:
#             raise Exception("""
#                 If the number of input channels and output channels are different you have to use the 1x1 convolution layer on the
#                 inputs to match their shape (channel wise in this case) with the output of residual block.
#                 """)
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=strides)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
#         if use_1x1_conv:
#             self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides)
#         else:
#             self.conv3 = None

#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)

#     def forward(self, X):
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         Y += X
#         return F.relu(Y)

class DenseBlock(nn.Module):

    def __init__(self, in_channels, num_convs, num_channels) -> None:
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(self.conv_block(in_channels, num_channels))
            in_channels += num_channels
        self.net = nn.Sequential(*layer)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
    
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            # Concatenate input and output of each block along the channels
            X = torch.cat((X, Y), dim=1)
        return X
    

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor 
    # 1D valid_lens specifies valid length for each batch However the 2d valid_lens specifies valid lengths also 
    # for each sample in each batch aswell meaning samples in the same batch can have different lengths.
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)



class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)


def calculate_ouput_dim(net, input_dim=None):
    
    batch_size = 8
    for child in net.children():
        if isinstance(child, nn.Sequential):
            child = child[0]
        for layer in child.modules():
            if isinstance(layer, nn.Sequential):
                layer = layer[0]
            
            if isinstance(layer, nn.Linear):
                (W_1, in_size) = list(net.parameters())[0].shape
                dummy_input = torch.randn((batch_size, in_size))
                output = net(dummy_input)
            elif isinstance(layer, nn.Conv2d):
                (K, in_C, K_h, K_w) = list(net.parameters())[0].shape
                if input_dim == None:
                    dummy_input = torch.randn(batch_size, in_C, 1024, 1024)
                    print('Since the input dimension was not specified the default input size was set to be (1024, 1024)')
                else:
                    if input_dim[0] != in_C:
                        error_str = 'input channels {} doesnt match the input channel size of network {}'.format(input_dim[0], in_C)
                        raise Exception(error_str)
                    dummy_input = torch.randn(batch_size, in_C, input_dim[1], input_dim[2])
                
                output = net(dummy_input)
            break
        break
    
    return output.shape


class Vocab():
    """Vocabulary for text."""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        counter = collections.Counter(tokens)

        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens
        self.idx_to_token = list(sorted(set(['<unk>'] + reserved_tokens + [
            token for token, freq in self.token_freqs if freq >= min_freq])))
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']
    
    
