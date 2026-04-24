import os
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io, transform
import pandas as pd
import tiler


class DriveDataset(Dataset):
    def __init__(self, base_dir, split, transform=None, img_size=224, *args, **kwargs):
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        self.img_size = img_size

        csv_map = {
            'train': 'train.csv',
            'val': 'val.csv',
            'test': 'test.csv'
        }
        
        csv_file = os.path.join(base_dir, csv_map.get(split, 'test.csv'))
        
        self.data_df = pd.read_csv(csv_file)
        self.image_paths = self.data_df['im_paths'].tolist()
        self.label_paths = self.data_df['gt_paths'].tolist()
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = io.imread(img_path)
        label = io.imread(label_path)
        
        # Handle 3D label loading (e.g. GIF (1, H, W) or RGB (H, W, 3))
        if len(label.shape) == 3:
            label = np.squeeze(label)
            # If still 3D (e.g. RGB label loaded as such), take first channel
            if len(label.shape) == 3:
                 label = label[:, :, 0]
        
        # Drive label: 0 bg, >0 vessel.
        # Robust binarization
        label = (label > 0).astype(np.float32)
                
        # Normalize image: Z-score normalization (per image) which is common for medical/contrast variations
        # Also TransUNet/R50 expects somewhat normalized inputs.
        # Check if we need to scale to 0-1 first?
        # If we do z-score, 0-255 or 0-1 base doesn't 'matter' for the shape, but mean value changes.
        # Let's simple z-score.
        if image.std() > 0:
            image = (image - image.mean()) / (image.std() + 1e-8)
        else:
             image = image - image.mean()

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        else:
            # Default formatting if no transform
            # Pre-resize
            # Using skimage transform resize
            # preserve_range=True means we keep the values, but they become float.
            # If input was 0-255, output is 0-255.
            image = transform.resize(image, (self.img_size, self.img_size), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)
            label = transform.resize(label, (self.img_size, self.img_size), order=0, preserve_range=True, anti_aliasing=False)

            # Transpose H, W, C -> C, H, W
            if len(image.shape) == 3:
                image = image.transpose(2, 0, 1)
            else:
                image = np.expand_dims(image, axis=0)
            
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label.long()}
            
        sample['case_name'] = os.path.basename(img_path).replace('.tif', '')
        return sample


class DriveTileDataset(DriveDataset):
    def __init__(self, base_dir, split, transform=None, img_size=224, overlap=0.5, *args, **kwargs):
        super().__init__(base_dir, split, transform, img_size, *args, **kwargs)

        # Mask
        self.mask_paths = self.data_df['mask_paths'].tolist() if 'mask_paths' in self.data_df.columns else None
        
        # Tiling parameters
        self.tile_size = img_size
        self.overlap = overlap

        if len(self.image_paths) > 0:
            # Read first image to determine shape
            e_img = io.imread(self.image_paths[0])
            self.img_shape = e_img.shape
            # If grayscale, shape might be (H, W)
            if len(self.img_shape) == 2:
                self.img_shape = (self.img_shape[0], self.img_shape[1], 1)
            
            self.lbl_shape = (self.img_shape[0], self.img_shape[1], 1)
        else:
            self.img_shape = (584, 565, 3)
            self.lbl_shape = (584, 565, 1)
        
        # Define Tiler for this specific image shape (to be safe if sizes vary slightly)
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
        mask_path = self.mask_paths[img_idx] if self.mask_paths else None
        
        image = io.imread(img_path)
        label = io.imread(label_path)
        mask = io.imread(mask_path) if mask_path else None
        
        if len(label.shape) == 3:
            label = np.squeeze(label)
            if len(label.shape) == 3:
                 label = label[:, :, 0]
                 
        if mask is not None and len(mask.shape) == 3:
            mask = np.squeeze(mask)
            if len(mask.shape) == 3:
                 mask = mask[:, :, 0]
        
        # Binarize label
        label = (label > 0).astype(np.float32)
        if mask is not None: mask = (mask > 0).astype(np.float32)
        
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
            # image_tile: H, W, C
            # label_tile: H, W, 1 -> H, W
            label_tile = label_tile.squeeze(-1)
            
            # image_tile might be H,W,1 if original was gray. If RGB, H,W,3.
            if image_tile.shape[-1] == 1:
                image_tile = image_tile.squeeze(-1)
                
            sample = {'image': image_tile, 'label': label_tile}
            
            if mask is not None:
                mask = np.expand_dims(mask, axis=-1)
                mask_tile = self.lbl_tiler.get_tile(mask, tile_id)
                mask_tile = mask_tile.squeeze(-1)
                sample['fov_mask'] = mask_tile
        else:
            label = (label > 0.5).astype(np.uint8)
            sample = {'image': image, 'label': label}
            if mask is not None:
                mask = (mask > 0.5).astype(np.uint8)
                sample['fov_mask'] = mask

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
            if mask is not None:
                sample['fov_mask'] = torch.from_numpy(mask.astype(np.float32))
            
        sample['case_name'] = os.path.basename(img_path).replace('.tif', '')
        return sample
