import cv2
import numpy as np
import torch
import torch.nn.functional as F
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import SimpleITK as sitk
import tiler

from losses import *


class VesselLoss(nn.Module):
    def __init__(self, n_classes, w_ce=0.4, w_dice=0.2, w_topology=0.4, alpha=None, bg_index=None):
        super(VesselLoss, self).__init__()
        self.n_classes = n_classes
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_topology = w_topology

        # Initialize individual loss components
        self.ce_loss = FocalLoss(alpha=alpha)
        # self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(n_classes)

        # self.topology_loss = VesselTopologyLoss(n_classes, bg_index=bg_index)
        self.topology_loss = SoftClDiceLoss(n_classes, bg_index=bg_index)

    def forward(self, inputs, targets, softmax=True):
        # Calculate individual losses
        loss_ce = self.ce_loss(inputs, targets.long())
        loss_dice = self.dice_loss(inputs, targets, softmax=softmax)
        loss_topology = self.topology_loss(inputs, targets, softmax=softmax)

        # Combine losses with specified weights
        total_loss = (self.w_ce * loss_ce) + \
                     (self.w_dice * loss_dice) + \
                     (self.w_topology * loss_topology)

        # Return total loss and individual components for logging
        return total_loss, loss_ce, loss_dice, loss_topology


def calculate_ce_weights(label_batch, num_classes):
    total_pixels = label_batch.numel()
    weights = []
    for c in range(num_classes):
        class_pixels = (label_batch == c).sum().float()
        # Inverse frequency weighting, add epsilon to prevent division by zero
        class_weight = total_pixels / (class_pixels + 1e-5)
        weights.append(class_weight)
        
    weights = torch.tensor(weights, device=label_batch.device)
    # Normalize weights so they sum to num_classes
    weights = weights / weights.sum() * num_classes
    return weights    


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list


def test_single_image(image, label, net, classes, patch_size=224, test_save_path=None, case=None):
    # image: (C, H, W) or (H, W)
    # label: (H, W)
    
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    
    if len(image.shape) == 3:
        # C, H, W
        c, x, y = image.shape
        if x != patch_size or y != patch_size:
            input_img = zoom(image, (1, patch_size / x, patch_size / y), order=3)
        else:
            input_img = image
        input_image = torch.from_numpy(input_img).unsqueeze(0).float().cuda() # (1, C, patch_H, patch_W)
    elif len(image.shape) == 2:
        # H, W
        x, y = image.shape
        if x != patch_size or y != patch_size:
            input_img = zoom(image, (patch_size / x, patch_size / y), order=3)
        else:
            input_img = image
        input_image = torch.from_numpy(input_img).unsqueeze(0).unsqueeze(0).float().cuda() # (1, 1, patch_H, patch_W)
    
    net.eval()
    with torch.no_grad():
        outputs = net(input_image)
        # outputs: (1, num_classes, patch_H, patch_W)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy() # (patch_H, patch_W)
        
        if x != patch_size or y != patch_size:
            prediction = zoom(out, (x / patch_size, y / patch_size), order=0)
        else:
            prediction = out
        
    metric_list = []
    # label is 0/1 for binary. classes=2.
    # If binary, prediction is 0 or 1.
    # calculate_metric_percase expects pred and gt.
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        # Normalize image to 0-255 for saving
        # image is float.
        if len(image.shape) == 3:
             img_save = np.transpose(image, (1, 2, 0)) # H, W, C
        else:
             img_save = image
             
        img_save = (img_save - img_save.min()) / (img_save.max() - img_save.min() + 1e-8) * 255
        img_save = img_save.astype(np.uint8)
        
        cv2.imwrite(test_save_path + '/'+case + "_img.png", img_save)
        cv2.imwrite(test_save_path + '/'+case + "_gt.png", (label*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+case + "_pred.png", (prediction*255).astype(np.uint8))
        
    return metric_list


def test_single_image_tiler(image, label, net, classes, tile_size=224, overlap=0.5, batch_size=4, test_save_path=None, case=None, fov_mask=None):
    """
    Inference using tiler library for overlap-tile strategy.
    """
    image = image.squeeze(0).cpu().detach().numpy() # C, H, W or H, W
    label = label.squeeze(0).cpu().detach().numpy() # H, W
    # if fov_mask is not None:
    #     fov_mask = fov_mask.squeeze(0).cpu().detach().numpy() # H, W
    
    # Check if image is CHW or HW. Tiler expects HWC or HW.
    # If CHW (3, H, W), transpose to HWC.
    if len(image.shape) == 3:
        image = image.transpose(1, 2, 0)
    elif len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        
    # shape is now (H, W, C)
        
    # Initialize Tiler for image
    img_tiler = tiler.Tiler(
        data_shape=image.shape,
        tile_shape=(tile_size, tile_size, image.shape[-1]),
        overlap=overlap,
        channel_dimension=2,
    )
    
    # Initialize Tiler for mask (probabilities)
    # We want to store probabilities for each class
    mask_tiler = tiler.Tiler(
        data_shape=(image.shape[0], image.shape[1], classes),
        tile_shape=(tile_size, tile_size, classes),
        overlap=overlap,
        channel_dimension=2,
    )
    
    # Calculate padding if needed
    # new_shape, padding = img_tiler.calculate_padding()
    # img_tiler.recalculate(data_shape=new_shape)
    # mask_tiler.recalculate(data_shape=new_shape)
    
    # padded_img = np.pad(image, padding)
    
    mask_merger = tiler.Merger(
        tiler=mask_tiler, 
        # window="overlap-tile",
    )
        
    net.eval()
    for batch_id, batch in img_tiler(image, batch_size=batch_size, progress_bar=False):
        # batch is (B, H, W, C)
        # Model expects (B, C, H, W)
        batch_tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float().cuda()
        
        with torch.no_grad():
            outputs = net(batch_tensor)
            probs = torch.softmax(outputs, dim=1) # (B, Classes, H, W)
            
        # Tiler Merger expects (B, H, W, Classes)
        probs_np = probs.cpu().numpy().transpose(0, 2, 3, 1)
        mask_merger.add_batch(batch_id, batch_size, probs_np)
        
    # mask_pred_probs = mask_merger.merge(extra_padding=padding, dtype=np.float32)
    mask_pred_probs = mask_merger.merge(unpad=True, dtype=np.float32)
    # mask_pred_probs is (H, W, Classes)
    
    # Argmax
    prediction = np.argmax(mask_pred_probs, axis=-1)

    # Calculate metrics
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        # Normalize image to 0-255 for saving
        # image is float.
        if len(image.shape) == 3:
             img_save = image # H, W, C
        else:
             img_save = image
             
        img_save = (img_save - img_save.min()) / (img_save.max() - img_save.min() + 1e-8) * 255
        img_save = img_save.astype(np.uint8)

        print(img_save.shape, label.shape, prediction.shape)
        
        cv2.imwrite(test_save_path + '/'+case + "_img.png", img_save)
        cv2.imwrite(test_save_path + '/'+case + "_gt.png", (label*255).astype(np.uint8))
        cv2.imwrite(test_save_path + '/'+case + "_pred.png", (prediction*255).astype(np.uint8))
        
    return metric_list


# Default TransUNet uses combined Dice + CE loss
def calculate_dice_ce_loss(pred, target, num_classes, ce_weight=0.5):
    ce_loss_fn = CrossEntropyLoss()
    dice_loss_fn = DiceLoss(num_classes)
    
    loss_ce = ce_loss_fn(pred, target.long())
    loss_dice = dice_loss_fn(pred, target, softmax=True)
    total_loss = ce_weight * loss_ce + (1 - ce_weight) * loss_dice

    return {
        "total_loss": total_loss, 
        "loss_dice": loss_dice, 
        "loss_ce": loss_ce
    }


def calculate_vessel_loss(pred, target, num_classes, alpha=None, bg_index=0):
    vessel_loss_fn = VesselLoss(num_classes, alpha=alpha, bg_index=bg_index)
    total_loss, loss_ce, loss_dice, loss_topology = vessel_loss_fn(pred, target)
    return {
        "total_loss": total_loss, 
        "loss_ce": loss_ce, 
        "loss_dice": loss_dice, 
        "loss_topology": loss_topology
    }
 