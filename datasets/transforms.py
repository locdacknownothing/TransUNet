import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom


def random_rot_flip(image, label, mask=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    if mask is not None: mask = np.rot90(mask, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    if mask is not None: mask = np.flip(mask, axis=axis).copy()
    if mask is not None: return image, label, mask
    return image, label

def random_rotate(image, label, mask=None):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    if mask is not None: mask = ndimage.rotate(mask, angle, order=0, reshape=False)
    if mask is not None: return image, label, mask
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        mask = sample.get('fov_mask', None)

        if random.random() > 0.5:
            if mask is not None: image, label, mask = random_rot_flip(image, label, mask)
            else: image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            if mask is not None: image, label, mask = random_rotate(image, label, mask)
            else: image, label = random_rotate(image, label)
        
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
            if mask is not None: mask = zoom(mask, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            
        # Transpose to C, H, W
        if len(image.shape) == 3:
             image = image.transpose(2, 0, 1) # H, W, C -> C, H, W
        else:
             image = np.expand_dims(image, axis=0) # H, W -> 1, H, W

        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        
        if mask is not None:
            mask = torch.from_numpy(mask.astype(np.float32))
            sample = {'image': image, 'label': label.long(), 'fov_mask': mask}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample
