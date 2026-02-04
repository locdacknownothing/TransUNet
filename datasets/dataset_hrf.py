import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from skimage import io, transform
from PIL import Image
import pandas as pd
import tiler

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        
        # image is H, W, C or H, W
        if len(image.shape) == 2:
            x, y = image.shape
            c = 1
        else:
            x, y, c = image.shape
            
        if x != self.output_size[0] or y != self.output_size[1]:
             if c == 3:
                 image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
             else:
                 image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            
             label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        # Transpose to C, H, W
        if len(image.shape) == 3:
             image = image.transpose(2, 0, 1) # H, W, C -> C, H, W
        else:
             image = np.expand_dims(image, axis=0) # H, W -> 1, H, W

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class HRF_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None, tile_size=224, overlap=0.5, *args, **kwargs):
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        self.tile_size = tile_size
        self.overlap = overlap
        
        csv_map = {
            'train': 'train.csv',
            'val': 'val.csv',
            'test': 'test.csv'
        }
        
        csv_file = os.path.join(base_dir, csv_map.get(split, 'test.csv'))
        
        self.data_df = pd.read_csv(csv_file)
        self.image_paths = self.data_df['im_paths'].tolist()
        self.label_paths = self.data_df['gt_paths'].tolist()
        
        if len(self.image_paths) > 0:
            # Read first image to determine shape
            e_img = io.imread(self.image_paths[0])
            self.img_shape = e_img.shape
            # If grayscale
            if len(self.img_shape) == 2:
                self.img_shape = (self.img_shape[0], self.img_shape[1], 1)
            self.lbl_shape = (self.img_shape[0], self.img_shape[1], 1)
        else:
            # Fallback (HRF typically 3504x2336)
            self.img_shape = (2336, 3504, 3) 
            self.lbl_shape = (2336, 3504, 1)

        # Define Tiler for this specific image shape
        self.img_tiler = tiler.Tiler(
            data_shape=self.img_shape,
            tile_shape=(self.tile_size, self.tile_size, self.img_shape[-1]),
            overlap=(int(self.tile_size * self.overlap), int(self.tile_size * self.overlap), 0),
            channel_dimension=2,
            mode='reflect'
        )
        
        self.lbl_tiler = tiler.Tiler(
            data_shape=self.lbl_shape,
            tile_shape=(self.tile_size, self.tile_size, 1),
            overlap=(int(self.tile_size * self.overlap), int(self.tile_size * self.overlap), 0),
            channel_dimension=2,
            mode='reflect'
        )
        self.tiles_per_image = len(self.img_tiler)

    def __len__(self):
        if self.split == 'train':
            return len(self.image_paths) * self.tiles_per_image
        else:
            return len(self.image_paths)

    def __getitem__(self, idx):
        if self.split == 'train':
            img_idx = idx // self.tiles_per_image
            tile_id = idx % self.tiles_per_image
        else:
            img_idx = idx
            tile_id = -1 # Not used for validation (full image)

        img_path = self.image_paths[img_idx]
        label_path = self.label_paths[img_idx]
        
        image = io.imread(img_path)
        label = io.imread(label_path)
        
        if len(label.shape) == 3:
            label = np.squeeze(label)
            if len(label.shape) == 3:
                 label = label[:, :, 0]
        
        # Binarize label
        label = (label > 0).astype(np.float32)
        
        # Normalize image z-score
        image = image.astype(np.float32)
        if image.std() > 0:
            image = (image - image.mean()) / (image.std() + 1e-8)
        else:
             image = image - image.mean()
        
        # Tiling logic for training
        if self.split == 'train':
            # Handle channels
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            # Check shape consistency
            if image.shape != self.img_shape:
                # Basic resize attempt if needed, or error out. 
                # Given HRF images are usually consistent, we stick to strict check for now or basic warning.
                 if image.shape != self.img_shape:
                     raise ValueError(f"Image shape {image.shape} does not match reference shape {self.img_shape}")

            # Label needs channel dim because of tiler expected data_shape.
            label = np.expand_dims(label, axis=-1)
            
            if label.shape != self.lbl_shape:
                 raise ValueError(f"Label shape {label.shape} does not match reference shape {self.lbl_shape}")
            
            actual_len = len(self.img_tiler)
            tile_id = tile_id % actual_len
            
            image_tile = self.img_tiler.get_tile(image, tile_id)
            label_tile = self.lbl_tiler.get_tile(label, tile_id)
            
            # Squeeze back
            label_tile = label_tile.squeeze(-1)
            
            if image_tile.shape[-1] == 1:
                image_tile = image_tile.squeeze(-1)
                
            sample = {'image': image_tile, 'label': label_tile}
        else:
            label = (label > 0.5).astype(np.uint8)
            sample = {'image': image, 'label': label}

        if self.transform and self.split == 'train':
            sample = self.transform(sample)
        elif self.split != 'train':
            # Validation: Format for Pytorch
            # H, W, C -> C, H, W
            if len(image.shape) == 3:
                image = image.transpose(2, 0, 1)
            else:
                image = np.expand_dims(image, axis=0)
            
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label.long()}
            
        sample['case_name'] = os.path.basename(img_path).replace('.JPG', '').replace('.jpg', '').replace('.png', '')
        return sample
