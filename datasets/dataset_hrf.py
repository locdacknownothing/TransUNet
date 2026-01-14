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
            
        if len(image.shape) == 3:
             image = image.transpose(2, 0, 1)
        else:
             image = np.expand_dims(image, axis=0)

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class HRF_dataset(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        
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
        # CSV contains absolute paths
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        image = io.imread(img_path)
        label = io.imread(label_path)
        
        # Handle 3D label loading
        if len(label.shape) == 3:
            label = np.squeeze(label)
            if len(label.shape) == 3:
                 label = label[:, :, 0]
        
        # Robust binarization
        label = (label > 0).astype(np.float32)
        
        # Resize to 224x224
        image = transform.resize(image, (224, 224), order=3, preserve_range=True, anti_aliasing=True).astype(np.float32)
        
        # Z-score normalization
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
            if len(image.shape) == 3:
                image = image.transpose(2, 0, 1)
            else:
                image = np.expand_dims(image, axis=0)
            
            image = torch.from_numpy(image.astype(np.float32))
            label = torch.from_numpy(label.astype(np.float32))
            sample = {'image': image, 'label': label.long()}
            
        sample['case_name'] = os.path.basename(img_path).replace('.JPG', '').replace('.jpg', '').replace('.png', '')
        return sample
