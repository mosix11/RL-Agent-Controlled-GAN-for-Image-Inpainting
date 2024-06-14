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
from src.models import AutoEncoder, LatentGAN
from src.trainers import AETrainer, GANTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Which model to train. Options \{AE, lGAN, RL\}", type=str, default=None)
    parser.add_argument("-s", "--save", help="Directory for saving trained models", type=str, default="./weights")
    args = parser.parse_args()
    
    
    
    if args.model == None:
        raise RuntimeError('You have to specify which model you want to train. Choose one among \{AE, lGAN, RL\}')
    saving_dir = Path(args.save)
    if not saving_dir.exists():
        os.mkdir(saving_dir)
    AE_weights_path = saving_dir.joinpath(Path('AE.pt'))
    lGAN_weights_path = saving_dir.joinpath(Path('lGAN.pt'))
    
    if args.model == 'AE':
        dataset = CelebA(root_dir=Path('./data').absolute())
        model = AutoEncoder(GFV_dim=2048)
        trainer = AETrainer(max_epochs=200, run_on_gpu=True, do_validation=True)
        trainer.fit(model, dataset)
        print('Saving model\'s weights')
        torch.save(model, AE_weights_path)

        
    elif args.model == 'lGAN':
        dataset = CelebA(root_dir=Path('./data').absolute())
        if not AE_weights_path.exists():
            raise RuntimeError('You have to first train the auto encoder!!')
        AE = torch.load(AE_weights_path)
        lgan = LatentGAN(z_dim=8, GFV_dim=2048)
        trainer = GANTrainer(max_epochs=300, run_on_gpu=True, do_validation=True)
        trainer.fit(lgan, AE, dataset, resume=True)
        print('Saving model\'s weights')
        torch.save(lgan, lGAN_weights_path)
        
        
        
    elif args.model == 'RL':
        pass
    else:
        raise RuntimeError('Wrong model type!! Choose one among \{AE, lGAN, RL\}')
    
    