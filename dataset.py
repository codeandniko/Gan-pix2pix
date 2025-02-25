from PIL import Image
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import kagglehub
import config

# Download latest version
path = kagglehub.dataset_download("vikramtiwari/pix2pix-dataset")

print("Path to dataset files:", path)

class MapDataset(Dataset): # type: ignore
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.list_files[idx])
        imge = np.array(Image.open(img_path))    
        input_image = imge[:, :512, :]
        target_image = imge[:, 512:, :]
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]
        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]
        return input_image, target_image
        
