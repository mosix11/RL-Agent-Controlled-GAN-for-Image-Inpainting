import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from ..utils import nn_utils
import os
import socket
import datetime
from pathlib import Path

from ray import tune
from ray.train import Checkpoint


class AETrainer():

    def __init__(self, max_epochs=150, lr:float=1e-4, optimizer_type="adam", use_lr_schduler=False,
                 run_on_gpu=False, gradient_clip_val=0, do_validation=True, write_summery=True,
                 outputs_dir:Path = Path('./outputs'), ray_tuner=None):
        
        self.max_epochs = max_epochs
        self.gradient_clip_val = gradient_clip_val
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.use_lr_schduler = use_lr_schduler
        self.do_val = do_validation
        self.outputs_dir = outputs_dir
        if not outputs_dir.exists:
            os.mkdir(outputs_dir)
        self.checkpoints_dir = self.outputs_dir.joinpath('checkpoints/')
        if not self.checkpoints_dir.exists():
            os.mkdir(self.checkpoints_dir)
        self.ray_tuner = ray_tuner
        if ray_tuner:
            if not do_validation:
                raise RuntimeError("In order to use ray tuner the validation step in each epoch should be done")
        
        self.write_sum = write_summery
        self.cpu = nn_utils.get_cpu_device()
        self.gpu = nn_utils.get_gpu_device()
        
        if self.gpu == None and run_on_gpu:
            raise RuntimeError("""gpu device not found!""")
        self.run_on_gpu = run_on_gpu
        
        if self.write_sum:
            self.writer = SummaryWriter(self.outputs_dir.joinpath('tensorboard/').joinpath(socket.gethostname()+'-'+str(datetime.datetime.now())))

    def prepare_data(self, data):
        self.train_dataloader = data.get_train_dataloader()
        self.val_dataloader = data.get_val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_batch(self, batch):
        if self.run_on_gpu:
            # batch = [a.to(self.gpu) for a in batch]
            batch = batch.to(self.gpu)
        return batch

    def prepare_model(self, model, state_dict=None):
        if self.run_on_gpu:
            model.to(self.gpu)
        if state_dict:
            model.load_state_dict(state_dict, map=self.gpu if self.run_on_gpu else self.cpu)
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
            optim.load_state_dict(state_dict, map=self.gpu if self.run_on_gpu else self.cpu)
        return optim
    
    def configure_lr_scheduler(self, optimizer, state_dict=None):
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=60)
        if state_dict:
            lr_scheduler.load_state_dict(state_dict, map=self.gpu if self.run_on_gpu else self.cpu)
        return lr_scheduler
        
    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def fit(self, model, data, checkpoint=None):
        self.prepare_data(data)
        if checkpoint:
            self.prepare_model(model, checkpoint['model_state'])
            self.configure_optimizers(checkpoint['optim_state'])
            self.epoch = checkpoint['epoch']
            if self.use_lr_schduler: self.lr_scheduler = self.configure_lr_scheduler(self.optim, checkpoint['lr_sch_state'])
        else:
            self.prepare_model(model)
            self.optim = self.configure_optimizers()
            self.epoch = 0
            if self.use_lr_schduler: self.lr_scheduler = self.configure_lr_scheduler(self.optim)
        
                
        # self.train_loss_hist = []
        # self.val_loss_hist = []
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
        # self.model.save_plot()
        if self.write_sum:
            self.writer.flush()


        

    def fit_epoch(self):
        print('#########  Entering Epoch {} #########'.format(self.epoch + 1))
        self.model.train()
        epoch_train_loss = 0.0
        for i, batch in enumerate(self.train_dataloader):
            loss = self.model.training_step(self.prepare_batch(batch))
            epoch_train_loss += loss.detach().cpu().numpy()
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)

                self.optim.step()
                
        if self.write_sum:
                self.writer.add_scalar('Loss/Train', epoch_train_loss/self.num_train_batches, self.epoch)
                
        if self.ray_tuner:
            path = self.checkpoints_dir.joinpath('/ckp.pt')

            torch.save({
                'model_state': self.model.state_dict(),
                'optim_state': self.optim.state_dict(),
                'epoch': self.epoch,
                'lr_sch_state': self.lr_scheduler.state_dict() if self.use_lr_schduler else None
            }, path)
            checkpoint = Checkpoint.from_directory(self.checkpoints_dir)
            


            
        if self.val_dataloader is None or not self.do_val:
            return
        self.model.eval()
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0
        for i, batch in enumerate(self.val_dataloader):
            with torch.no_grad():
                loss, acc = self.model.validation_step(self.prepare_batch(batch))
                epoch_val_loss += loss.detach().cpu().numpy()
                epoch_val_acc += acc.cpu().numpy()
        
        if self.use_lr_schduler:
            self.lr_scheduler.step(epoch_val_loss/self.num_val_batches)
            print(self.lr_scheduler.get_last_lr())
        if self.write_sum:
            self.writer.add_scalar('Loss/Val', epoch_val_loss/self.num_val_batches, self.epoch)
            self.writer.add_scalar('Acc/Val', epoch_val_acc/self.num_val_batches, self.epoch)

        if self.ray_tuner:
            self.ray_tuner.report(
                {'loss': epoch_val_loss/self.num_val_batches, 'accuracy': epoch_val_acc/self.num_val_batches},
                checkpoint=checkpoint
            )
            # self.ray_tuner.report(
            #     loss=epoch_val_loss/self.num_val_batches, accuracy=epoch_val_acc/self.num_val_batches
            # )