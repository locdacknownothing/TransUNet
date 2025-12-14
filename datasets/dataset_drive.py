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
            # Zoom expects channel first if we want to zoom all channels? No, zoom is spatial.
            # If image is H, W, 3:
            if c == 3:
                 # Resize using skimage might be easier or ndimage.zoom per channel
                 # Be careful with channels.
                 # Let's use resize from skimage or PIL before converting to numpy in __getitem__?
                 # But RandomGenerator is passed to Dataset.
                 # Let's assume input is already resized in __getitem__ for simplicity or handle here.
                 # To simplify, we'll implement simple resize if needed, but better to ensure __getitem__ provides correct size 
                 # or we use scipy zoom for each channel.
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

class Drive_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None, *args, **kwargs):
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        
        # Determine CSV file based on split
        # We assume dataset_drive logic maps 'train' to train.csv and 'test'/'val' accordingly
        # But get_public_data.py creates train.csv, val.csv, test.csv
        csv_map = {
            'train': 'train.csv',
            'val': 'val.csv',
            'test': 'test.csv'
        }
        
        csv_file = os.path.join(base_dir, csv_map.get(split, 'test.csv'))
        
        # We need to handle paths. The CSV contains relative paths like "data/DRIVE/images/..."
        # But base_dir might be "/app/data/DRIVE".
        # If we run on modal, CSV paths might need adjustment if they are relative to "src/references/lwnet"
        # but we are mounting data differently.
        # Let's assume we regenerate CSVs or fix paths.
        # Ideally, we read the CSV and prepend base_dir if the CSV paths are just filenames, 
        # but they seem to be "data/DRIVE/images/..."
        # We can strip "data/DRIVE/" prefix.
        
        self.data_df = pd.read_csv(csv_file)
        self.image_paths = self.data_df['im_paths'].tolist()
        self.label_paths = self.data_df['gt_paths'].tolist()
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # CSV now contains absolute paths generated during preparation
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
        
        # Pre-resize
        # Using skimage transform resize
        # preserve_range=True means we keep the values, but they become float.
        # If input was 0-255, output is 0-255.
        image = transform.resize(image, (224, 224), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)
        
        # Normalize image: Z-score normalization (per image) which is common for medical/contrast variations
        # Also TransUNet/R50 expects somewhat normalized inputs.
        # Check if we need to scale to 0-1 first?
        # If we do z-score, 0-255 or 0-1 base doesn't 'matter' for the shape, but mean value changes.
        # Let's simple z-score.
        if image.std() > 0:
            image = (image - image.mean()) / (image.std() + 1e-8)
        else:
             image = image - image.mean()

        label = transform.resize(label, (224, 224), order=0, preserve_range=True, anti_aliasing=False)
        label = (label > 0.5).astype(np.uint8)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        else:
            # Default formatting if no transform
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
