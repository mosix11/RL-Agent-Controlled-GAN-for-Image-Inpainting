import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import reverb
from ..utils import nn_utils
import os
import socket
import datetime
from pathlib import Path
import time


class RLTrainer():
    
    def __init__(self, max_episodes=1e6, replay_buffer_size=1e6, batch_size=100,
                 lr_A=1e-4, lr_C=1e-2, initial_exploration_episodes=1e4, evaluation_freq=5e3,
                 exploration_noise=0.1, discount_factor=0.99, tau=5e-3, policy_noise=0.2, policy_noise_clip=0.5,
                 policy_freq=2,
                 run_on_gpu=False, do_validation=True, write_summery=True,
                 outputs_dir:Path = Path('./outputs')) -> None:
        
        specifier_path = Path('RL/')
        if not outputs_dir.exists:
            os.mkdir(outputs_dir)
        outputs_dir = outputs_dir.joinpath(specifier_path)
        if not outputs_dir.exists():
            os.mkdir(outputs_dir)
        self.outputs_dir = outputs_dir
        
        self.cpu = nn_utils.get_cpu_device()
        self.gpu = nn_utils.get_gpu_device()
        if self.gpu == None and run_on_gpu:
            raise RuntimeError("""gpu device not found!""")
        self.run_on_gpu = run_on_gpu
        
        self.max_episodes = max_episodes
        self.initial_exploration_episodes = initial_exploration_episodes
        self.evaluation_freq = evaluation_freq
        self.lr_A = lr_A
        self.lr_C = lr_C
        self.batch_size = batch_size
        self.do_val = do_validation


        self.rb_server = reverb.Server(tables=[
            reverb.Table(
                name='replay_buffer',
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=replay_buffer_size,
                rate_limiter=reverb.rate_limiters.MinSize(1)),
            ],
        )
        self.rb_client = reverb.Client(f'localhost:{self.rb_server.port}')
        print(self.rb_client.server_info())
        self.batch_size = batch_size
        
        self.write_sum = write_summery
        if self.write_sum:
            self.writer = SummaryWriter(self.outputs_dir.joinpath('tensorboard/').joinpath(socket.gethostname()+'-'+str(datetime.datetime.now())))
            
        self.checkpoints_dir = self.outputs_dir.joinpath('checkpoints/')
        if not self.checkpoints_dir.exists():
            os.mkdir(self.checkpoints_dir)
        
        
        
        
    def add_experience(self, observation, action, reward, next_observation, done):
        
        with self.rb_client.trajectory_writer(num_keep_alive_refs=1) as writer:
            writer.append((observation, action, reward, next_observation, done))
            
            # Creating an item with a trajectory
            # In this example, we store a single transition as a trajectory
            # Adjust `num_keep_alive_refs` and `sequence_length` based on your use case
            writer.create_item(
                table='experience_replay',
                priority=1.0,
                trajectory=writer.history[-1:]
            )
            
            # Block until the item has been inserted and confirmed by the server.
            writer.flush()
            
        
        
    def sample_experiences(self):
        # print(list(self.rb_client.sample('experience_replay', num_samples=self.rb_batch_size)))
        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=f'localhost:{self.rb_server.port}',
            table='experience_replay',
            max_in_flight_samples_per_worker=2 * self.batch_size
        )
        dataloader = iter(dataset.batch(self.batch_size))
        return next(dataloader)
    
    
    def prepare_batch(self, batch):
        if self.run_on_gpu:
            batch = [item.to(self.gpu) for item in batch]
        return batch
    
    def prepare_model(self, DDPG, GAN, AE, state_dict=None):
        if state_dict:
            DDPG.load_state_dict(state_dict)
        if self.run_on_gpu:
            DDPG.to(self.gpu)
            GAN.to(self.gpu)
            AE.to(self.gpu)

        self.DDPG = DDPG
        self.GAN = GAN
        self.AE = AE
        self.GAN.eval()
        self.AE.eval()
        
    def configure_optimizers(self, state_dict_A=None, state_dict_C=None):
        optim_A = torch.optim.Adam(self.DDPG.actor.parameters(), lr=self.lr_A)
        optim_C = torch.optim.Adam(self.DDPG.discriminator.parameters(), lr=self.lr_C)
        if state_dict_A and state_dict_C:
            optim_A.load_state_dict(state_dict_A)
            optim_C.load_state_dict(state_dict_C)
            
        return optim_A, optim_C
    
    
    def fit(self, DDPG, GAN, AE, data, resume=False):
        
        pass