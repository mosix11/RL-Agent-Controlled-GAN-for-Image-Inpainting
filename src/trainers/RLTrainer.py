import reverb.platform
import reverb.platform.checkpointers_lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import reverb
import tensorflow as tf
from ..models.RLAgent import Environment
from ..utils import nn_utils, utils
import os
import socket
import datetime
from pathlib import Path
import time


class RLTrainer():
    
    def __init__(self, max_episodes=2e6,
                 replay_buffer_size=3e5,
                 state_dim=1024,
                 action_dim=8,
                 policy_training_batch_size=128,
                #  lr_A=1e-3, lr_C=1e-3,   
                 lr_A=1e-4, lr_C=1e-4,
                #  lr_A=1e-5, lr_C=1e-4,
                #  lr_A=3e-4, lr_C=3e-4,
                #  lr_A=3e-4, lr_C=3e-4,
                 initial_exploration_episodes=2e4,
                #  initial_exploration_episodes=0,
                 evaluation_freq=5e3,
                 exploration_noise=0.1,
                 target_policy_updt_freq=2,
                 run_on_gpu=False,
                 do_validation=True,
                 write_summery=True,
                 log_frequency=5e2,
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
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episodes = int(max_episodes)
        self.policy_training_batch_size = policy_training_batch_size
        self.initial_exploration_episodes = int(initial_exploration_episodes)
        self.evaluation_freq = int(evaluation_freq)
        self.target_policy_updt_freq = target_policy_updt_freq
        self.lr_A = lr_A
        self.lr_C = lr_C
        self.do_val = do_validation

        self.exploration_noise = exploration_noise
        
        
        self.setup_replay_buffer(replay_buffer_size)
        self.write_sum = write_summery
        if self.write_sum:
            self.writer = SummaryWriter(self.outputs_dir.joinpath('tensorboard/').joinpath(socket.gethostname()+'-'+str(datetime.datetime.now())))
            self.log_frequency = log_frequency    
        
        self.checkpoints_dir = self.outputs_dir.joinpath('checkpoints/')
        if not self.checkpoints_dir.exists():
            os.mkdir(self.checkpoints_dir)
        
        
    def setup_replay_buffer(self, replay_buffer_size):
        self.reverb_checkpoints_dir = self.outputs_dir.joinpath('reverb_checkpoints/')
        if not self.reverb_checkpoints_dir.exists():
            os.mkdir(self.reverb_checkpoints_dir)
        
        checkpointer = reverb.platform.checkpointers_lib.DefaultCheckpointer(
                            path=str(self.reverb_checkpoints_dir))
        
        signature = {
            'obs': tf.TensorSpec(shape=[self.state_dim], dtype=tf.float32),  
            'act': tf.TensorSpec(shape=[self.action_dim], dtype=tf.float32), 
            'nxt_obs': tf.TensorSpec(shape=[self.state_dim], dtype=tf.float32),  
            'rwrd': tf.TensorSpec(shape=[], dtype=tf.float32), 
            'done': tf.TensorSpec(shape=[], dtype=tf.bool)
        }
        self.rb_server = reverb.Server(tables=[
            reverb.Table(
                name='replay_buffer',
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=int(replay_buffer_size),
                rate_limiter=reverb.rate_limiters.MinSize(1),
                signature=signature
                ),
            ],
            checkpointer=checkpointer
        )
        self.rb_client = reverb.Client(f'localhost:{self.rb_server.port}')
        print('\n\nReverb table size : ', self.rb_client.server_info()['replay_buffer'].current_size, '\n\n')
        
        
    def add_experience(self, observation, action, next_observation, reward, done):
        
        with self.rb_client.trajectory_writer(num_keep_alive_refs=1) as writer:
            writer.append({
                'obs': observation,
                'act': action,
                'nxt_obs': next_observation,
                'rwrd': reward,
                'done': done
            })
            
            # Create TrajectoryColumns for each element in the transition
            trajectory_columns = {
                'obs': writer.history['obs'][-1],
                'act':writer.history['act'][-1],
                'nxt_obs': writer.history['nxt_obs'][-1],
                'rwrd': writer.history['rwrd'][-1],
                'done': writer.history['done'][-1]
            }
            
            # Create the item with the trajectory
            writer.create_item(
                table='replay_buffer',
                priority=1.0,
                trajectory=trajectory_columns
            )
            
            # Block until the item has been inserted and confirmed by the server.
            writer.flush()
            
        
        
    def sample_experiences(self):
        # print(list(self.rb_client.sample('experience_replay', num_samples=self.rb_batch_size)))
        dataset = reverb.TrajectoryDataset.from_table_signature(
            server_address=f'localhost:{self.rb_server.port}',
            table='replay_buffer',
            max_in_flight_samples_per_worker=2 * self.policy_training_batch_size
        )
        dataloader = iter(dataset.batch(self.policy_training_batch_size))
        batch = next(dataloader)
        return self.prepare_experience_batch(batch)
    
    def prepare_experience_batch(self, replay_sample):
        obs = torch.from_numpy(replay_sample.data['obs'].numpy()).to(self.gpu if self.run_on_gpu else self.cpu)
        act = torch.from_numpy(replay_sample.data['act'].numpy()).to(self.gpu if self.run_on_gpu else self.cpu)
        rwrd = torch.from_numpy(replay_sample.data['rwrd'].numpy()).to(self.gpu if self.run_on_gpu else self.cpu)
        nxt_obs = torch.from_numpy(replay_sample.data['nxt_obs'].numpy()).to(self.gpu if self.run_on_gpu else self.cpu)
        done = torch.from_numpy(replay_sample.data['done'].numpy()).to(self.gpu if self.run_on_gpu else self.cpu)
        return obs, act, nxt_obs, rwrd, done
    
    def prepare_data(self, data):
        self.train_dataloader = data.get_train_dataloader()
        self.train_dl_iterator = iter(self.train_dataloader)
        self.val_dataloader = data.get_val_dataloader()
        self.val_dl_iterator = iter(self.val_dataloader)
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
    
    def prepare_batch(self, batch):
        if self.run_on_gpu:
            batch = [item.to(self.gpu) for item in batch]
        return batch
    
    def get_train_batch(self):
        # Index 0 is complete image and index 1 is masked image
        try:
            batch = self.prepare_batch(next(self.train_dl_iterator))
        except:
            self.train_dl_iterator = iter(self.train_dataloader)
            batch = self.prepare_batch(next(self.train_dl_iterator))
        return batch
    
    def get_val_batch(self):
        # Index 0 is complete image and index 1 is masked image
        try:
            batch = self.prepare_batch(next(self.val_dl_iterator))
        except:
            self.val_dl_iterator = iter(self.train_dataloader)
            batch = self.prepare_batch(next(self.val_dl_iterator))
        return batch
    
    def prepare_models(self, DDPG, GAN, AE, state_dict=None):
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
        
    def prepare_env(self):
        env = Environment(self.GAN, self.AE, self.action_dim, self.state_dim)
        if self.run_on_gpu:
            env.to(self.gpu)
        self.env = env    
        
    def configure_optimizers(self, state_dict_A=None, state_dict_C=None):
        optim_A = torch.optim.Adam(self.DDPG.actor.parameters(), lr=self.lr_A)
        optim_C = torch.optim.Adam(self.DDPG.critic.parameters(), lr=self.lr_C)
        if state_dict_A and state_dict_C:
            optim_A.load_state_dict(state_dict_A)
            optim_C.load_state_dict(state_dict_C)
            
        self.optim_A = optim_A
        self.optim_C = optim_C
    
    
    def fit(self, DDPG, GAN, AE, data, resume=False):
        self.prepare_data(data)
        if resume:
            self.prepare_models(DDPG, GAN, AE)
            self.prepare_env()
            self.configure_optimizers()
            # self.current_episode = self.rb_client.server_info()['replay_buffer'].current_size
            self.current_episode = 0
        else:
            self.prepare_models(DDPG, GAN, AE)
            self.prepare_env()
            self.configure_optimizers()
            self.current_episode = 0
        
        for self.current_episode in range(self.current_episode, self.max_episodes):
            self.train()
            
        if self.write_sum:
            self.writer.flush()


    def train(self):
        if ((self.current_episode + 1) % 500 == 0) : print('#########  Starting Episode {} #########'.format(self.current_episode + 1))
        
        # Batch of ground truth images and corresponding masked images
        # Index 0 is complete image and index 1 is masked image
        gt_input_batch, masked_input_batch = self.get_train_batch()
        # masked_input_batch = gt_input_batch
        
        # Convert the masked image to GFV
        current_state = self.env.get_state(masked_input_batch)
        
        # In the in initial exploration phase we run pure random uniform policy
        if self.current_episode < self.initial_exploration_episodes:
            # 99 percent of the values drawn from a normal distribution with zero mean and unit variance fall in
            # the range [-2.6, 2.6]. This range can be changed by tuning action_value_range parameter of DDPG in train.py script
            action = torch.FloatTensor(masked_input_batch.shape[0], self.action_dim).uniform_(-self.DDPG.action_value_range, self.DDPG.action_value_range).to(masked_input_batch.device)
            if self.current_episode == 0: print('\n\n******* Starting Initial Random Exploration Phase *******\n\n')
            elif self.current_episode == self.initial_exploration_episodes - 1 : print('\n\n******* Ending Initial Random Exploration Phase *******\n\n')
        
        else:
            if self.current_episode == self.initial_exploration_episodes : print('\n\n******* From Now on, Actions Are Taken From Policy *******\n\n')
            action = self.DDPG.select_action(current_state)
            if self.exploration_noise != 0:
                action = (action + torch.randn(action.shape, device=action.device)* self.exploration_noise).clamp(-self.DDPG.action_value_range, self.DDPG.action_value_range)
        
        
        next_state, reward, done, losses = self.env(current_state, action, gt_input_batch)
        
        
        if (self.current_episode+1) % self.log_frequency == 0:
            self.writer.add_scalar('Reward --> Maximize', reward.detach().cpu().numpy(), self.current_episode)
            self.writer.add_scalar('Loss/Discriminator --> Maximize', losses[0].detach().cpu().numpy(), self.current_episode)
            self.writer.add_scalar('Loss/GFV --> Minimize', losses[1].detach().cpu().numpy(), self.current_episode)
            self.writer.add_scalar('Loss/Action Norm --> Minimize', losses[2].detach().cpu().numpy(), self.current_episode)
            self.writer.add_scalar('Loss/Image Reconstruction --> Minimize', losses[3].detach().cpu().numpy(), self.current_episode)
        
            
        self.add_experience(current_state.detach().cpu().squeeze(0).numpy(),
                            action.detach().cpu().squeeze(0).numpy(),
                            next_state.detach().cpu().squeeze(0).numpy(),
                            reward.detach().cpu().numpy(),
                            done)
        if (self.current_episode+1) % 1000 == 0:
            print(action)
            
        if (self.current_episode+1) % int(3e5) == 0:
            print('******* Checkpointing Replay Buffer Experiences *******')
            self.rb_client.checkpoint()
        
        # if done and self.current_episode > self.policy_training_batch_size:
        if done and self.current_episode > self.initial_exploration_episodes:
            # states, actions, next_states, rewards, dones = self.sample_experiences()
            experience_batch = self.sample_experiences()
            
            critic_loss = self.DDPG.C_training_step(experience_batch)
            
            self.optim_C.zero_grad()
            critic_loss.backward()
            self.optim_C.step()

            # actor_loss = self.DDPG.A_training_step(experience_batch)

            # self.optim_A.zero_grad()
            # actor_loss.backward()
            # self.optim_A.step()
                   
            if (self.current_episode + 1) % self.target_policy_updt_freq == 0:
                actor_loss = self.DDPG.A_training_step(experience_batch)
    
                self.optim_A.zero_grad()
                actor_loss.backward()
                self.optim_A.step()
                
                self.DDPG.update_target_networks()
            
            if self.write_sum and (self.current_episode + 1) % self.log_frequency == 0:
                self.writer.add_scalar('Loss/Actor', actor_loss.detach().cpu().numpy(), self.current_episode)
                self.writer.add_scalar('Loss/Critic', critic_loss.detach().cpu().numpy(), self.current_episode)
                
            
            
            if (self.current_episode + 1) % self.evaluation_freq == 0:
                print('#########  Evaluating Policy #########')
                self.DDPG.eval()
                self.evaluate_policy()
                self.DDPG.train()
                
    def evaluate_policy(self):
        
        # Sample 10 masked images and get the unmasked version from network and show
        imgs = []
        for i in range(10):
            # Batch of ground truth images and corresponding masked images
            # Index 0 is complete image and index 1 is masked image
            gt_input_batch, masked_input_batch = self.get_val_batch()
            # masked_input_batch = gt_input_batch
            
            with torch.no_grad():
                # Convert the masked image to GFV
                current_state = self.env.get_state(masked_input_batch)
                done = False
                
                 
                while not done:
                    # Action By Agent and collect reward
                    action = self.DDPG.select_action(current_state)
                    next_state, reward, done = self.env(current_state, action)
                    
                
                recons_batch = self.AE.decode_GFV(next_state)
            
            imgs.append((masked_input_batch+1)/2)
            imgs.append((recons_batch+1)/2)
            imgs.append((gt_input_batch+1)/2)
            
        
        self.writer.add_images('Image Reconstruction', torch.cat(imgs, dim=0), global_step=0)
            

        
        
            