import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ..utils import nn_utils
import os
import socket
import datetime
from pathlib import Path
import time
from tqdm import tqdm

class GANTrainer():
    
    def __init__(self, max_epochs=2500, D_lr:float=5e-4, G_lr=1e-4,
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
        if state_dict:
            GAN.load_state_dict(state_dict)
        if self.run_on_gpu:
            GAN.to(self.gpu)
            AE.to(self.gpu)

        self.GAN = GAN
        self.AE = AE

    def configure_optimizers(self, state_dict_G=None, state_dict_D=None):
        optim_G = torch.optim.Adam(self.GAN.generator.parameters(), lr=self.G_lr, betas=(0.5, 0.999))
        optim_D = torch.optim.Adam(self.GAN.discriminator.parameters(), lr=self.D_lr, betas=(0.5, 0.999))
        if state_dict_G and state_dict_D:
            optim_G.load_state_dict(state_dict_G)
            optim_D.load_state_dict(state_dict_D)
            
        return optim_G, optim_D
            
            
    def fit(self, GAN, AE, data, resume=False):
        self.prepare_data(data)
        if resume:
            if self.checkpoints_dir.joinpath('ckp.pt').exists():
                checkpoint = torch.load(
                    self.checkpoints_dir.joinpath('ckp.pt')
                )  
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

        
        self.scaler_G = torch.cuda.amp.GradScaler()
        self.scaler_D = torch.cuda.amp.GradScaler()
        for self.epoch in range(self.epoch, self.max_epochs):
            self.fit_epoch()
            
        
        # self.fit_epoch()

        if self.write_sum:
            self.writer.flush()


        

    def fit_epoch(self):
        
        # ******** Training Part ********
        self.GAN.train()
        self.AE.eval()
        
        
        fixed_latent_z = self.prepare_batch(torch.randn(56, self.GAN.z_dim))
        
        # total_mean, total_std = 0, 0
        # for i, batch in enumerate(self.train_dataloader):
        #     real_GFV = self.AE.encoder(self.prepare_batch(batch))
        #     del batch
        #     batch_std, batch_mean = torch.std_mean(real_GFV.detach().cpu())
        #     total_mean = ((i) * total_mean + batch_mean) / (i+1)
        #     total_std = ((i) * total_std + batch_std) / (i+1)
            
        # print('mean: ', total_mean)
        # print('std: ', total_std)

        # sample_reconstructed = self.model.predict(self.prepare_batch(sample_batch)).to(sample_batch.device)
        # change the range from -1 to 1 to 0 to 1
        # sample_batch = (sample_batch + 1) / 2
        # sample_reconstructed = (sample_reconstructed + 1) / 2
        # combined_images = torch.cat([sample_batch, sample_reconstructed], dim=3)  # dim=3 stacks them horizontally
        # self.writer.add_images('Val Image Reconstruction', combined_images, global_step=0)
        
        
        # latent_z = self.prepare_batch(torch.randn(56, self.GAN.z_dim))
        # randomGFV = self.GAN.generate(latent_z)
        # print('fake :', torch.std_mean(randomGFV))
        # reconstructed_imgs = self.AE.decode_GFV(randomGFV)
        # reconstructed_imgs= (reconstructed_imgs + 1) / 2
        # reconstructed_imgs = reconstructed_imgs.clamp_(0, 1)
        # self.writer.add_images('Random Image Reconstruction', reconstructed_imgs, global_step=0)
        
        # return
        epoch_train_loss_D = 0.0
        epoch_train_loss_G = 0.0
        for i, real_imgs in tqdm(enumerate(self.train_dataloader), total=self.num_train_batches, desc="Processing Training Epoch {}".format(self.epoch+1)):
            batch_size = real_imgs.size(0)
            real_imgs = self.prepare_batch(real_imgs)
            
            with torch.no_grad():
                real_GFV = self.AE.encoder(real_imgs)

            # ******** Training Discriminator ********
            self.optim_D.zero_grad()
            self.optim_G.zero_grad()
            
            latent_z = self.prepare_batch(torch.randn(batch_size, self.GAN.z_dim))
            
            d_loss = self.GAN.D_training_step(real_GFV, latent_z)
            d_loss.backward()
            self.optim_D.step()
            # with torch.cuda.amp.autocast():
            #     d_loss = self.GAN.D_training_step(real_GFV, latent_z)

            # self.scaler_D.scale(d_loss).backward()
            # self.scaler_D.step(self.optim_D)
            # self.scaler_D.update()
            
            epoch_train_loss_D += d_loss.detach().cpu().numpy()
            
            # ******** Training Generator ********
            self.optim_D.zero_grad()
            self.optim_G.zero_grad()
            
            latent_z = self.prepare_batch(torch.randn(batch_size, self.GAN.z_dim))
            
            g_loss = self.GAN.G_training_step(latent_z)
            g_loss.backward()
            self.optim_G.step()
            # with torch.cuda.amp.autocast():
            #     g_loss = self.GAN.G_training_step(latent_z)
                
            # self.scaler_G.scale(g_loss).backward()
            # self.scaler_G.step(self.optim_G)
            # self.scaler_G.update()
            
            epoch_train_loss_G += g_loss.detach().cpu().numpy()

            
            
        if self.write_sum:
            self.writer.add_scalar('Loss/Train Discriminator', epoch_train_loss_D/self.num_train_batches, self.epoch)
            self.writer.add_scalar('Loss/Train Generator', epoch_train_loss_G/self.num_train_batches, self.epoch)
            
                
        
        # ******** Saving Checkpoint ********
        if (self.epoch+1) % 50 == 0:
            print('Saving chekpoint...\n')
            path = self.checkpoints_dir.joinpath('ckp.pt')
            torch.save({
                'model_state': self.GAN.state_dict(),
                'optim_G_state': self.optim_G.state_dict(),
                'optim_D_state': self.optim_D.state_dict(),
                'epoch': self.epoch+1,
            }, path)    
            
        # ******** Validation Part ********
        if self.val_dataloader is None or not self.do_val:
            return
        
        self.GAN.eval()
            
        # ******** Writing Summary and Stats to Tensorboard ********
        if self.write_sum:
            if (self.epoch+1) % 20 == 0:
                GFV = self.GAN.generate(fixed_latent_z)
                reconstructed_imgs = self.AE.decode_GFV(GFV)
                reconstructed_imgs= (reconstructed_imgs + 1) / 2
                reconstructed_imgs = reconstructed_imgs.clamp_(0, 1)
                self.writer.add_images('Image Reconstruction', reconstructed_imgs, global_step=0)
                