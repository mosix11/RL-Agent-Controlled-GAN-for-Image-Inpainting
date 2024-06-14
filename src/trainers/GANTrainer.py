import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ..utils import nn_utils
import os
import socket
import datetime
from pathlib import Path

class GANTrainer():
    
    def __init__(self, max_epochs=300, D_lr:float=4e-4, G_lr=1e-4,
                 run_on_gpu=False, do_validation=True, write_summery=True,
                 outputs_dir:Path = Path('./outputs')) -> None:
        
        specifier_path = Path('GAN/')
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

        self.max_epochs = max_epochs
        self.D_lr = D_lr
        self.G_lr = G_lr
        
        self.do_val = do_validation

        self.write_sum = write_summery
        if self.write_sum:
            self.writer = SummaryWriter(self.outputs_dir.joinpath('tensorboard/').joinpath(socket.gethostname()+'-'+str(datetime.datetime.now())))
        
        self.checkpoints_dir = self.outputs_dir.joinpath('checkpoints/')
        if not self.checkpoints_dir.exists():
            os.mkdir(self.checkpoints_dir)
            
            
    def prepare_data(self, data):
        self.train_dataloader = data.get_train_dataloader()
        self.val_dataloader = data.get_val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)
        
        
    def prepare_batch(self, batch):
        if self.run_on_gpu:
            batch = batch.to(self.gpu)
        return batch

    def prepare_model(self, GAN, AE, state_dict=None):
        if self.run_on_gpu:
            GAN.to(self.gpu)
            AE.to(self.gpu)
        if state_dict:
            GAN.load_state_dict(state_dict, map=self.gpu if self.run_on_gpu else self.cpu)
        self.GAN = GAN
        self.AE = AE

    def configure_optimizers(self, state_dict_G=None, state_dict_D=None):
        optim_G = torch.optim.Adam(self.GAN.generator.parameters(), lr=self.G_lr, betas=(0.0, 0.9))
        optim_D = torch.optim.Adam(self.GAN.discriminator.parameters(), lr=self.D_lr, betas=(0.0, 0.9))     
        if state_dict_G and state_dict_D:
            optim_G.load_state_dict(state_dict_G)
            optim_D.load_state_dict(state_dict_D)
            
        return optim_G, optim_D


    def fit(self, GAN, AE, data, resume=False):
        self.prepare_data(data)
        if resume:
            checkpoint = torch.load(
                self.checkpoints_dir.joinpath('/ckp.pt')
            )
            if checkpoint:    
                self.prepare_model(GAN, AE, checkpoint['model_state'])
                self.optim_G, self.optim_D = self.configure_optimizers(checkpoint['optim_G_state'], checkpoint['optim_D_state'])
                self.epoch = checkpoint['epoch']
            else:
                self.prepare_model(GAN, AE)
                self.optim_G, self.optim_D = self.configure_optimizers()
                self.epoch = 0
        else:
            self.prepare_model(GAN, AE)
            self.optim_G, self.optim_D = self.configure_optimizers()
            self.epoch = 0
        
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()

        if self.write_sum:
            self.writer.flush()


        

    def fit_epoch(self):
        print('#########  Starting Epoch {} #########'.format(self.epoch + 1))
        
        # ******** Training Part ********
        self.GAN.train()
        self.AE.eval()
        
        
        fixed_latent_z = self.prepare_batch(torch.randn(50, self.GAN.z_dim))
        
        epoch_train_loss_D = 0.0
        epoch_train_loss_G = 0.0
        for i, real_imgs in enumerate(self.train_dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = self.prepare_batch(real_imgs)
            real_GFV = self.AE.encoder(real_imgs)
            
            # train disc
            latent_z = self.prepare_batch(torch.randn(batch_size, self.GAN.z_dim))
            d_loss = self.GAN.D_training_step(real_GFV, latent_z)
            self.optim_D.zero_grad()
            d_loss.backward()
            self.optim_D.step()
            
            epoch_train_loss_D += d_loss.detach().cpu().numpy()
            
            # train gen
            latent_z = self.prepare_batch(torch.randn(batch_size, self.GAN.z_dim))
            g_loss = self.GAN.G_training_step(latent_z)
            self.optim_G.zero_grad()
            g_loss.backward()
            self.optim_G.step()
            
            epoch_train_loss_G += g_loss.detach().cpu().numpy()
            
            
                
        if self.write_sum:
            self.writer.add_scalar('Loss/Train Discriminator', epoch_train_loss_D/self.num_train_batches, self.epoch)
            self.writer.add_scalar('Loss/Train Generator', epoch_train_loss_G/self.num_train_batches, self.epoch)
            
                
        
        # ******** Saving Checkpoint ********
        if (self.epoch+1) % 5 == 0:
            print('Saving chekpoint...\n')
            path = self.checkpoints_dir.joinpath('ckp.pt')
            torch.save({
                'model_state': self.GAN.state_dict(),
                'optim_G_state': self.optim_G.state_dict(),
                'optim_D_state': self.optim_D.state_dict(),
                'epoch': self.epoch,
            }, path)    
            
        # ******** Validation Part ********
        if self.val_dataloader is None or not self.do_val:
            return
        
        self.GAN.eval()
        # for i, real_imgs in enumerate(self.val_dataloader):
        #     with torch.no_grad():
        #         batch_size = real_imgs.size(0)
        #         real_imgs = self.prepare_batch(real_imgs)
        #         real_GFV = self.AE.encoder(real_imgs)
        #         latent_z = torch.randn(batch_size, self.GAN.z_dim)
        #         d_loss = self.GAN.D_training_step(real_GFV, latent_z)
                
        #         loss = self.model.validation_step(self.prepare_batch(batch))
        #         epoch_val_loss += loss.detach().cpu().numpy()
            
            
        # ******** Writing Summary and Stats to Tensorboard ********
        if self.write_sum:
            if (self.epoch+1) % 5 == 0:
                GFV = self.GAN.generate(fixed_latent_z)
                reconstructed_imgs = self.AE.decode_GFV(GFV)
                self.writer.add_images('Image Reconstruction', reconstructed_imgs, global_step=0)