import torch
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Which model to train. Options \{AE, lGAN, RL\}", type=str, default=None)
    parser.add_argument("-s", "--save", help="Directory for saving trained models", type=str, default="./weights")
    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark = True
    
    
    if args.model == None:
        raise RuntimeError('You have to specify which model you want to train. Choose one among \{AE, lGAN, RL\}')
    saving_dir = Path(args.save)
    if not saving_dir.exists():
        os.mkdir(saving_dir)
    AE_weights_path = saving_dir.joinpath(Path('AE.pt'))
    lGAN_weights_path = saving_dir.joinpath(Path('lGAN.pt'))
    RL_Agent_weights_path = saving_dir.joinpath(Path('RL.pt'))
    
    if args.model == 'AE':
        dataset = CelebA(root_dir=Path('./data').absolute(), batch_size=64, num_workers=2)
        model = AutoEncoder(GFV_dim=1024)
        trainer = AETrainer(max_epochs=150, run_on_gpu=True, do_validation=True, use_lr_schduler=True)
        trainer.fit(model, dataset, resume=True)
        print('Saving model\'s weights')
        torch.save(model, AE_weights_path)

        
    elif args.model == 'lGAN':
        dataset = CelebA(root_dir=Path('./data').absolute(), batch_size=256, num_workers=2)
        if not AE_weights_path.exists():
            raise RuntimeError('You have to first train the auto encoder!!')
        AE = torch.load(AE_weights_path)
        lgan = LatentGAN(z_dim=8, GFV_dim=1024)
        trainer = GANTrainer(max_epochs=2500, run_on_gpu=True, do_validation=True)
        trainer.fit(lgan, AE, dataset, resume=False)
        print('Saving model\'s weights')
        torch.save(lgan, lGAN_weights_path)
        
    elif args.model == 'RL':
        dataset = CelebA(root_dir=Path('./data').absolute(), batch_size=1, num_workers=2, mask=True)
        if not AE_weights_path.exists():
            raise RuntimeError('You have to first train the auto encoder!!')
        if not lGAN_weights_path.exists():
            raise RuntimeError('You have to first train the latent GAN!!')
        AE = torch.load(AE_weights_path)
        lgan = torch.load(lGAN_weights_path)
        RLPolicy = DDPG(state_dim=1024, action_dim=8, action_value_range=2.6)
        trainer = RLTrainer(max_episodes=1e6, run_on_gpu=True, do_validation=True)
        trainer.fit(RLPolicy, lgan, AE, dataset, resume=True)
        torch.save(RLPolicy, RL_Agent_weights_path)
        
        
    else:
        raise RuntimeError('Wrong model type!! Choose one among \{AE, lGAN, RL\}')
    
    