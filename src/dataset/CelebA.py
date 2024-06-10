import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms

from PIL import Image


import os
import sys
from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, folder_path:Path, transform=None) -> None:
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, fname))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

class CelebA():
    
    def __init__(self,
                 root_dir:Path = Path('./data').absolute(),
                 outputs_dir = Path('./outputs').absolute(),
                 batch_size:int = 32,
                 img_size:tuple = (256, 256),
                 normalize:bool = True,
                 normalization_stats:tuple = (0.5, 0.5),
                 num_workers:int = 1,
                 valset_ratio:float = 0.1,
                 testset_ratio:float = 0.1,
                 seed:int = None,
                 device=None) -> None:
        

        
        if not root_dir.exists():
            raise RuntimeError("The root directory does not exist!")
        if not outputs_dir.exists:
            raise RuntimeError("The output directory does not exist!")
        
        self.batch_size = batch_size
        self.img_size = img_size
        self.root_dir = root_dir
        self.output_dir = outputs_dir
        self.num_workers = num_workers
        self.trainset_ration = 1 - (valset_ratio + testset_ratio)
        self.valset_ratio = valset_ratio
        self.testset_ratio = testset_ratio
        self.seed = seed
        self.device = device
        
        transformations = [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
        if normalize:
            transformations.append(transforms.Normalize(*normalization_stats))
        self.transformations = transforms.Compose(transformations)
        
        self._init_loaders()
        
        
        
    def get_train_dataloader(self):
        return self.train_loader
    
    def get_val_dataloader(self):
        return self.val_loader
    
    def get_test_dataloader(self):
        return self.test_loader
    
    
    def _init_loaders(self):
        image_dataset = ImageDataset(self.root_dir.joinpath(Path('CelebA')), transform=self.transformations)
        dataset_size = len(image_dataset)
        # train_size = int((1 - self.valset_ratio - self.testset_ratio) * dataset_size)
        # val_size = int(self.valset_ratio * dataset_size)
        # test_size = dataset_size - train_size - val_size
        
        if self.seed:
            generator = torch.Generator().manual_seed(self.seed)
            trainset, valset, testset = random_split(image_dataset, [self.trainset_ration, self.valset_ratio, self.testset_ratio], generator=generator)
        else:
            trainset, valset, testset = random_split(image_dataset, [self.trainset_ration, self.valset_ratio, self.testset_ratio])
                    
        self.train_loader = self._build_dataloader(trainset)
        self.val_loader = self._build_dataloader(valset)
        self.test_loader = self._build_dataloader(testset)
        
        
    def _build_dataloader(self, dataset):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        return dataloader