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


class DiceLoss(nn.Module):
    def __init__(self, n_classes, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        count = 0
        for i in range(0, self.n_classes):
            if self.ignore_index is not None and i == self.ignore_index:
                continue
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
            count += 1
            
        return loss / count if count > 0 else loss


class VesselLoss(nn.Module):
    def __init__(self, n_classes, w_ce=0.5, w_dice=0.3, w_cldice=0.2, ce_weight=None, bg_index=0):
        super(VesselLoss, self).__init__()
        self.n_classes = n_classes
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.w_cldice = w_cldice

        # Initialize individual loss components
        self.ce_loss = CrossEntropyLoss(weight=ce_weight, reduction='none')
        self.dice_loss = DiceLoss(n_classes, ignore_index=bg_index)
        self.cldice_loss = SoftClDiceLoss(n_classes, bg_index=bg_index)

    def forward(self, inputs, targets, fov_mask=None, softmax=True):
        # Calculate individual losses
        loss_ce = self.ce_loss(inputs, targets.long())
        if fov_mask is not None:
            loss_ce = (loss_ce * fov_mask).sum() / (fov_mask.sum() + 1e-8)
        else:
            loss_ce = loss_ce.mean()

        inputs_prob = torch.softmax(inputs, dim=1) if softmax else inputs
        if fov_mask is not None:
            inputs_prob = inputs_prob * fov_mask.unsqueeze(1)
            targets = targets * fov_mask

        loss_dice = self.dice_loss(inputs_prob, targets, softmax=False)
        loss_cldice = self.cldice_loss(inputs_prob, targets, softmax=False)

        # Combine losses with specified weights
        total_loss = (self.w_ce * loss_ce) + \
                     (self.w_dice * loss_dice) + \
                     (self.w_cldice * loss_cldice)

        # Return total loss and individual components for logging
        return total_loss, loss_ce, loss_dice, loss_cldice
        

def soft_erode(img):
    if len(img.shape)==4:
        p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
        p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
        return torch.min(p1,p2)
    elif len(img.shape)==5:
        p1 = -F.max_pool3d(-img,(3,1,1),(1,1,1),(1,0,0))
        p2 = -F.max_pool3d(-img,(1,3,1),(1,1,1),(0,1,0))
        p3 = -F.max_pool3d(-img,(1,1,3),(1,1,1),(0,0,1))
        return torch.min(torch.min(p1, p2), p3)


def soft_dilate(img):
    if len(img.shape)==4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape)==5:
        return F.max_pool3d(img,(3,3,3),(1,1,1),(1,1,1))


def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, iters):
    img1  =  soft_open(img)
    skel  =  F.relu(img-img1)
    for i in range(iters):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img-img1)
        skel  =  skel +  F.relu(delta-skel*delta)
    return skel


class SoftClDiceLoss(nn.Module):
    def __init__(self, n_classes, iters=3, smooth=1e-5, bg_index=0):
        super(SoftClDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.iters = iters
        self.smooth = smooth
        self.bg_index = bg_index

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, y_pred, y_true, softmax=True):
        # y_pred: (B, C, H, W)
        # y_true: (B, H, W)
        if softmax:
            y_pred = torch.softmax(y_pred, dim=1)
        
        y_true = self._one_hot_encoder(y_true)
        
        total_loss = 0.0
        count = 0
        
        for i in range(self.n_classes):
            if self.bg_index is not None and i == self.bg_index:
                continue
                
            v_p = y_pred[:, i:i+1, ...]
            v_l = y_true[:, i:i+1, ...]
            
            t_p = soft_skel(v_p, self.iters)
            t_l = soft_skel(v_l, self.iters)
            
            tp = (torch.sum(t_p * v_l) + self.smooth) / (torch.sum(t_p) + self.smooth)
            tr = (torch.sum(t_l * v_p) + self.smooth) / (torch.sum(t_l) + self.smooth)
            
            cldice = 2.0 * (tp * tr) / (tp + tr + self.smooth)
            total_loss += 1.0 - cldice
            count += 1
            
        return total_loss / count if count > 0 else total_loss


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


def test_single_image(image, label, net, classes, patch_size=[224, 224], test_save_path=None, case=None):
    # image: (C, H, W) or (H, W)
    # label: (H, W)
    
    # image comes from test.py as tensor or numpy?
    # test.py: image, label = sampled_batch["image"], sampled_batch["label"]
    # In test_single_volume: image = image.squeeze(0).cpu().detach().numpy()
    # So input is Tensor (B, C, H, W) -> squeeze(0) -> (C, H, W)
    
    image = image.squeeze(0).cpu().detach().numpy()
    label = label.squeeze(0).cpu().detach().numpy()
    
    if len(image.shape) == 3:
        # C, H, W
        input_image = torch.from_numpy(image).unsqueeze(0).float().cuda() # (1, C, H, W)
    elif len(image.shape) == 2:
        # H, W
        input_image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().cuda() # (1, 1, H, W)
    
    net.eval()
    with torch.no_grad():
        outputs = net(input_image)
        # outputs: (1, num_classes, H, W)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy() # (H, W)
        
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
    if fov_mask is not None:
        fov_mask = fov_mask.squeeze(0).cpu().detach().numpy() # H, W
    
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
        overlap=(int(tile_size * overlap), int(tile_size * overlap), 0),
        channel_dimension=2,
        mode='reflect'
    )
    
    # Initialize Tiler for mask (probabilities)
    # We want to store probabilities for each class
    mask_tiler = tiler.Tiler(
        data_shape=(image.shape[0], image.shape[1], classes),
        tile_shape=(tile_size, tile_size, classes),
        overlap=(int(tile_size * overlap), int(tile_size * overlap), 0),
        channel_dimension=2,
        mode='reflect'
    )
    
    # Calculate padding if needed
    new_shape, padding = img_tiler.calculate_padding()
    img_tiler.recalculate(data_shape=new_shape)
    mask_tiler.recalculate(data_shape=new_shape)
    
    padded_img = np.pad(image, padding, mode="reflect")
    
    mask_merger = tiler.Merger(tiler=mask_tiler, window="overlap-tile")
        
    net.eval()
    for batch_id, batch in img_tiler(padded_img, batch_size=batch_size, progress_bar=False):
        # batch is (B, H, W, C)
        # Model expects (B, C, H, W)
        batch_tensor = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float().cuda()
        
        with torch.no_grad():
            outputs = net(batch_tensor)
            probs = torch.softmax(outputs, dim=1) # (B, Classes, H, W)
            
        # Tiler Merger expects (B, H, W, Classes)
        probs_np = probs.cpu().numpy().transpose(0, 2, 3, 1)
        mask_merger.add_batch(batch_id, batch_size, probs_np)
        
    mask_pred_probs = mask_merger.merge(extra_padding=padding, dtype=np.float32)
    # mask_pred_probs is (H, W, Classes)
    
    # Argmax
    prediction = np.argmax(mask_pred_probs, axis=-1)
    
    if fov_mask is not None:
        prediction = prediction * fov_mask
        
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


def calculate_vessel_loss(pred, target, num_classes, fov_mask=None, w_ce=0.6, w_dice=0.2, w_cldice=0.2, ce_weight=None, bg_index=0):
    vessel_loss_fn = VesselLoss(num_classes, w_ce=w_ce, w_dice=w_dice, w_cldice=w_cldice, ce_weight=ce_weight, bg_index=bg_index)
    total_loss, loss_ce, loss_dice, loss_cldice = vessel_loss_fn(pred, target, fov_mask=fov_mask)
    return {
        "total_loss": total_loss, 
        "loss_ce": loss_ce, 
        "loss_dice": loss_dice, 
        "loss_cldice": loss_cldice
    }
 