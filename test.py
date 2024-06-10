from pathlib import Path

from src.dataset import CelebA
from src.models import AutoEncoder
from src.trainers import AETrainer


dataset = CelebA(root_dir=Path('./data').absolute())
model = AutoEncoder()
trainer = AETrainer(run_on_gpu=True, do_validation=False)
trainer.fit(model, dataset)
# train_loader = dataset.get_train_dataloader()
# for img in train_loader:
#     print(img.shape)
#     break