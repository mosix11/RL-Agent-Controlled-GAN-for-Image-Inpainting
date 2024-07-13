import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms.v2 as transforms 

from torchvision.io import read_image


import os
import sys
from pathlib import Path

class ImageDataset(Dataset):
    def __init__(self, folder_path:Path, transform=None, mask=False) -> None:
        self.folder_path = folder_path
        self.mask = mask
        self.transform = transform
        self.image_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, fname))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(img_path)
        
        if self.mask:
            img_mask = self.mask_image(image)
        
        if self.transform:
            image = self.transform(image)
            if self.mask: img_mask = self.transform(img_mask)
        
        if self.mask:
            return (image, img_mask)
        else:
            return image
        
        
    def mask_image(self, img):
        mask = self.create_random_mask()
        mask = torch.from_numpy(mask).unsqueeze(0).expand_as(img)
        return img * mask
    
    def create_random_mask(self, image_size=(256, 256), mask_min_size=50, mask_max_size=100):
        img_height, img_width = image_size

        # Central square coordinates
        x1, y1 = 48, 48
        x2, y2 = 208, 208

        # Define the random mask size
        mask_w = np.random.randint(mask_min_size, mask_max_size)
        mask_h = np.random.randint(mask_min_size, mask_max_size)

        # Randomly select a position within the central square
        mask_x = np.random.randint(x1, x2 - mask_w)
        mask_y = np.random.randint(y1, y2 - mask_h)

        # Create the mask for 3 channels
        mask = np.ones((img_height, img_width), dtype=np.uint8)
        mask[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w] = 0

        return mask

class CelebA():
    
    def __init__(self,
                 root_dir:Path = Path('./data').absolute(),
                 outputs_dir = Path('./outputs').absolute(),
                 batch_size:int = 256,
                 img_size:tuple = (128, 128),
                 mask=False,
                 num_workers:int = 2,
                 valset_ratio:float = 0.05,
                 testset_ratio:float = 0.05,
                 seed:int = 11,
                 device=None) -> None:
        

        
        if not root_dir.exists():
            raise RuntimeError("The root directory does not exist!")
        if not outputs_dir.exists:
            raise RuntimeError("The output directory does not exist!")
        
        self.batch_size = batch_size
        self.img_size = img_size
        self.mask = mask
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
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
        self.transformations = transforms.Compose(transformations)
        
        self._init_loaders()
        
        
        
    def get_train_dataloader(self):
        return self.train_loader
    
    def get_val_dataloader(self):
        return self.val_loader
    
    def get_test_dataloader(self):
        return self.test_loader
    
    
    def _init_loaders(self):
        image_dataset = ImageDataset(self.root_dir.joinpath(Path('CelebA')), transform=self.transformations, mask=self.mask)
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
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        return dataloader
    
    
    # def mask_batch(self, batch):
    #     masks = self.create_random_mask_batch(self.batch_size)
    #     masks = torch.from_numpy(masks).unsqueeze(1).expand_as(batch)
    #     return batch * masks
    
    # def mask_image(self, img):
    #     mask = self.create_random_mask()
    #     mask = torch.from_numpy(mask).unsqueeze(0).expand_as(img)
    #     return img * mask
    
    # def create_random_mask(self, image_size=(256, 256), mask_min_size=30, mask_max_size=70):
    #     img_height, img_width = image_size

    #     # Central square coordinates
    #     x1, y1 = 32, 32
    #     x2, y2 = 224, 224

    #     # Define the random mask size
    #     mask_w = np.random.randint(mask_min_size, mask_max_size)
    #     mask_h = np.random.randint(mask_min_size, mask_max_size)

    #     # Randomly select a position within the central square
    #     mask_x = np.random.randint(x1, x2 - mask_w)
    #     mask_y = np.random.randint(y1, y2 - mask_h)

    #     # Create the mask for 3 channels
    #     mask = np.ones((img_height, img_width), dtype=np.uint8)
    #     mask[mask_y:mask_y + mask_h, mask_x:mask_x + mask_w] = 0

    #     return mask
    
    
    # def create_random_mask_batch(self, batch_size, image_size=(256, 256), mask_min_size=30, mask_max_size=70):
    #     img_height, img_width = image_size
    #     masks = np.ones((batch_size, img_height, img_width), dtype=np.uint8)

    #     # Central square coordinates
    #     x1, y1 = 32, 32
    #     x2, y2 = 224, 224

    #     for i in range(batch_size):
    #         # Define the random mask size
    #         mask_w = np.random.randint(mask_min_size, mask_max_size)
    #         mask_h = np.random.randint(mask_min_size, mask_max_size)

    #         # Randomly select a position within the central square
    #         mask_x = np.random.randint(x1, x2 - mask_w)
    #         mask_y = np.random.randint(y1, y2 - mask_h)

    #         # Apply the mask
    #         masks[i, mask_y:mask_y + mask_h, mask_x:mask_x + mask_w] = 0

    #     return masks