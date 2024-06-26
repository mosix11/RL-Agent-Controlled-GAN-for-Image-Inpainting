import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

class InverseNetwork(nn.Module):
    def __init__(self):
        super(InverseNetwork, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 8)
        self.bn5 = nn.BatchNorm1d(8)
        
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = torch.relu(self.bn4(self.fc4(x)))
        x = self.bn5(self.fc5(x))  # Note: No activation on the output layer
        return x

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
inverse_net = InverseNetwork()
inverse_net.to(gpu)
# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(inverse_net.parameters(), lr=1e-5)

for epoch in range(200):
    dataloader = dataset.get_train_dataloader()
    for batch in dataloader:  # Assuming you have a DataLoader for your 1024-dimensional data
        # Get the input 1024-dimensional vector
        inputs = batch.to(gpu)  # Shape: [batch_size, 1024]
        gfv = AE.get_GFV(inputs)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass through the inverse network
        z_pred = inverse_net(gfv)

        # Generate the 1024-dimensional vector using the pre-trained generator
        outputs = lgan.generator(z_pred)

        # Compute the loss between the original inputs and the generated outputs
        loss = criterion(outputs, gfv)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
    
    # Print the loss for every epoch
    print(f'Epoch [{epoch+1}/{200}], Loss: {loss.item()}')
    
    if (epoch+1)%20 == 0:
        samples = None
        for i, batch in enumerate(dataset.get_val_dataloader()):
            if i < 1 :
                samples = batch
            else:
                break
        samples = samples.to(gpu)
        gfv = AE.get_GFV(inputs)
        inverse_net.eval()
        z = inverse_net(gfv)
        inverse_net.train()
        gen_gfv = lgan.generator(z)
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


