from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


from src.dataset import CelebA
from src.models import AutoEncoder, LatentGAN
from src.trainers import AETrainer
from src.utils import utils, nn_utils

from torch.utils.tensorboard import SummaryWriter

# dataset = CelebA(root_dir=Path('./data').absolute())
# model = AutoEncoder()
# trainer = AETrainer(max_epochs=50, run_on_gpu=True, do_validation=True)
# trainer.fit(model, dataset)
# train_loader = dataset.get_train_dataloader()
# for img in train_loader:
#     print(img.shape)
#     break

cpu = nn_utils.get_cpu_device()
gpu = nn_utils.get_gpu_device()
# z_latent = torch.randn((1, 16), device=gpu)
# lgan = LatentGAN()
# lgan.to(gpu)
# gen = lgan.generator
# disc = lgan.discriminator

# gfv = gen(z_latent)
# score = disc(gfv)
# print(score.shape)


dataset = CelebA(root_dir=Path('./data').absolute(), batch_size=10, num_workers=2, mask=True)
test_dir = Path('outputs/test')
if not test_dir.exists():
    os.mkdir(test_dir)
writer = SummaryWriter(test_dir)
samples = None
for i, batch in enumerate(dataset.get_train_dataloader()):
    if i < 1 :
        # print(batch[0].shape)
        samples = batch
    else:
        break




saving_dir = Path('./weights')
AE_weights_path = saving_dir.joinpath(Path('AE.pt'))
AE = torch.load(AE_weights_path)
AE.to(gpu)
m_batch = samples[1]
m_batch = m_batch.to(gpu)
recons = AE.predict(m_batch)
recons = (recons + 1) / 2
samples = torch.cat([(samples[0] + 1) / 2, (samples[1] + 1) / 2], dim=0)
writer.add_images('Test Mask', samples, global_step=0)
writer.add_images('recons', recons, global_step=0)
writer.flush()