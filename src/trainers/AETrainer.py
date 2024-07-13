import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ..utils import nn_utils
import os
import socket
import datetime
from pathlib import Path

class AETrainer():

    def __init__(self, max_epochs=150, lr:float=1e-4, optimizer_type="adam", use_lr_schduler=False,
                 run_on_gpu=False, do_validation=True, write_summery=True,
                 outputs_dir:Path = Path('./outputs')):
        
        specifier_path = Path('AE/')
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
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.use_lr_schduler = use_lr_schduler
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

    def prepare_model(self, model, state_dict=None):
        if state_dict:
            model.load_state_dict(state_dict)
        if self.run_on_gpu:
            model.to(self.gpu)
        self.model = model

    def configure_optimizers(self, state_dict=None):
        if self.optimizer_type == "adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "adamw":
            optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "sgd":
            optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type == "rmsprop":
            optim = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        else:
            raise RuntimeError("Invalide optimizer type")
        if state_dict:
            optim.load_state_dict(state_dict)
        return optim
    
    def configure_lr_scheduler(self, optimizer, state_dict=None):
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        if state_dict:
            lr_scheduler.load_state_dict(state_dict)
        return lr_scheduler
        
    def fit(self, model, data, resume=False):
        self.prepare_data(data)
        if resume:
            if self.checkpoints_dir.joinpath('ckp.pt').exists():
                checkpoint = torch.load(
                    self.checkpoints_dir.joinpath('ckp.pt')
                )
                self.prepare_model(model, checkpoint['model_state'])
                self.optim = self.configure_optimizers(checkpoint['optim_state'])
                self.epoch = checkpoint['epoch']
                if self.use_lr_schduler: self.lr_scheduler = self.configure_lr_scheduler(self.optim, checkpoint['lr_sch_state'])
            else:
                self.prepare_model(model)
                self.optim = self.configure_optimizers()
                self.epoch = 0
                if self.use_lr_schduler: self.lr_scheduler = self.configure_lr_scheduler(self.optim)                
        else:
            self.prepare_model(model)
            self.optim = self.configure_optimizers()
            self.epoch = 0
            if self.use_lr_schduler: self.lr_scheduler = self.configure_lr_scheduler(self.optim)
        
        
        self.scaler = torch.cuda.amp.GradScaler()
        for self.epoch in range(self.epoch, self.max_epochs):
            self.fit_epoch()

        if self.write_sum:
            self.writer.flush()


        

    def fit_epoch(self):
        print('#########  Starting Epoch {} #########'.format(self.epoch + 1))
        
        # ******** Training Part ********
        self.model.train()
        epoch_train_loss = 0.0
        for i, batch in enumerate(self.train_dataloader):
            
            self.optim.zero_grad()
            with torch.cuda.amp.autocast():
                loss = self.model.training_step(self.prepare_batch(batch))
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
            
            epoch_train_loss += loss.detach().cpu().numpy()
                
        if self.write_sum:
                self.writer.add_scalar('Loss/Train', epoch_train_loss/self.num_train_batches, self.epoch+1)
                
                
        # ******** Saving Checkpoint ********
        if (self.epoch+1) % 5 == 0:
            print('Saving chekpoint...\n')
            path = self.checkpoints_dir.joinpath('ckp.pt')
            torch.save({
                'model_state': self.model.state_dict(),
                'optim_state': self.optim.state_dict(),
                'epoch': self.epoch,
                'lr_sch_state': self.lr_scheduler.state_dict() if self.use_lr_schduler else None
            }, path)    
            
        # ******** Validation Part ********
        if self.val_dataloader is None or not self.do_val:
            return
        self.model.eval()
        epoch_val_loss = 0.0
        for i, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                loss = self.model.validation_step(self.prepare_batch(batch))
                epoch_val_loss += loss.detach().cpu().numpy()
                
        
        if self.use_lr_schduler:
            self.lr_scheduler.step(epoch_val_loss/self.num_val_batches)
            print(self.lr_scheduler.get_last_lr())
            
            
        # ******** Writing Summary and Stats to Tensorboard ********
        if self.write_sum:
            self.writer.add_scalar('Loss/Val', epoch_val_loss/self.num_val_batches, self.epoch+1)
            if (self.epoch+1) % 5 == 0:
                sample_batch = None
                for i, batch in enumerate(self.val_dataloader):
                    sample_batch = batch
                    break
                sample_reconstructed = self.model.predict(self.prepare_batch(sample_batch)).to(sample_batch.device)
                # change the range from -1 to 1 to 0 to 1
                sample_batch = (sample_batch + 1) / 2
                sample_reconstructed = (sample_reconstructed + 1) / 2
                combined_images = torch.cat([sample_batch, sample_reconstructed], dim=3)  # dim=3 stacks them horizontally
                self.writer.add_images('Val Image Reconstruction', combined_images, global_step=0)
                
              
    
                