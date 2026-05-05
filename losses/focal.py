import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        gamma: focusing factor. 2.0 is a good default.
               Higher gamma → more focus on hard (thin vessel) pixels.
        alpha: optional per-class weight tensor, same as CE weight
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (B, C, H, W), targets: (B, H, W) long
        ce_loss = F.cross_entropy(inputs, targets.long(),
                                  weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)                        # p_t = prob of correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # down-weight easy pixels
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss
          