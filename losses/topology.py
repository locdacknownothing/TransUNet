import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, n_classes, iters=5, smooth=1e-5, bg_index=0, fp_penalty_weight=0.5):
        super(SoftClDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.iters = iters
        self.smooth = smooth
        self.bg_index = bg_index
        self.fp_penalty_weight = fp_penalty_weight

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

            # Penalises predicted skeleton pixels that fall OUTSIDE the GT vessel.
            # Isolated dots produce skeleton responses outside real vessels, so
            # this term directly suppresses them.
            fp_skel = torch.sum(t_p * (1.0 - v_l)) / \
                      (torch.sum(t_p) + self.smooth)
            
            total_loss += (1.0 - cldice) + self.fp_penalty_weight * fp_skel
            count += 1
            
        return total_loss / count if count > 0 else total_loss


class VesselTopologyLoss(nn.Module):
    """
    Topology-aware loss that directly penalizes vessel interruptions/breaks.

    SoftClDiceLoss measures skeleton *overlap*, but a gap in the middle of a vessel
    still gets partial credit. This loss uses three complementary mechanisms to
    explicitly detect and penalize breaks:

    1. Skeleton Gap Loss (w_gap):
       Penalizes low predicted probability *along* the GT skeleton.
       A break = prediction drops to zero where GT centerline is continuous.
       This is the most direct break signal.

    2. Endpoint Excess Loss (w_endpoint):
       Every break in a vessel creates exactly 2 spurious skeleton endpoints
       (the two cut ends). By counting excess endpoints in the predicted
       skeleton vs. GT skeleton, we get a count proportional to #breaks.
       Formula: excess_endpoints = max(0, pred_endpoints - gt_endpoints)

    3. Morphological Continuity Loss (w_continuity):
       Dilates the GT skeleton to build a "path corridor". Inside this corridor,
       the predicted probability should be uniformly high. Gaps inside the
       corridor = breaks that went undetected by gap loss due to soft values.

    Usage:
        loss_fn = VesselTopologyLoss(n_classes=2, bg_index=0)
        loss = loss_fn(pred_logits, target_mask)
    """

    def __init__(
        self,
        n_classes: int,
        iters: int = 3,
        smooth: float = 1e-5,
        bg_index: int = 0,
        w_gap: float = 0.4,
        w_endpoint: float = 0.4,
        w_continuity: float = 0.2,
        endpoint_temperature: float = 10.0,
    ):
        """
        Args:
            n_classes:            Total number of segmentation classes.
            iters:                Skeletonisation iterations (same as SoftClDiceLoss).
            smooth:               Numerical stability constant.
            bg_index:             Class index to skip (background).
            w_gap:                Weight for Skeleton Gap Loss.
            w_endpoint:           Weight for Endpoint Excess Loss.
            w_continuity:         Weight for Morphological Continuity Loss.
            endpoint_temperature: Sharpness of the soft endpoint detector.
                                  Higher = closer to hard binary endpoint map.
        """
        super(VesselTopologyLoss, self).__init__()
        self.n_classes = n_classes
        self.iters = iters
        self.smooth = smooth
        self.bg_index = bg_index
        self.w_gap = w_gap
        self.w_endpoint = w_endpoint
        self.w_continuity = w_continuity
        self.endpoint_temperature = endpoint_temperature

        # Fixed 8-connectivity kernel for neighbour counting — registered as buffer
        # so it moves with .cuda() / .to(device) automatically.
        kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
        kernel[0, 0, 1, 1] = 0.0          # exclude centre pixel
        self.register_buffer("_neighbor_kernel", kernel)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _one_hot(self, mask: torch.Tensor) -> torch.Tensor:
        """(B, H, W) int mask → (B, C, H, W) float one-hot."""
        return torch.cat(
            [((mask == i).unsqueeze(1)) for i in range(self.n_classes)], dim=1
        ).float()

    def _count_neighbors(self, skel: torch.Tensor) -> torch.Tensor:
        """
        For every pixel in `skel`, count the sum of its 8 neighbours.

        Args:
            skel: (B, 1, H, W) soft skeleton tensor in [0, 1].
        Returns:
            (B, 1, H, W) neighbour-sum tensor.
        """
        B, C, H, W = skel.shape
        # Convolve each batch×channel independently
        flat = skel.view(B * C, 1, H, W)
        kernel = self._neighbor_kernel.to(dtype=flat.dtype, device=flat.device)
        counts = F.conv2d(flat, kernel, padding=1)
        return counts.view(B, C, H, W)
    
    def _soft_endpoint_map(self, skel: torch.Tensor) -> torch.Tensor:
        """
        Differentiable approximation of the skeleton endpoint map.

        A skeleton pixel is an endpoint when it has ≤ 1 skeleton neighbour.
        We approximate the indicator with two stacked sigmoids:
            σ(T · (1.5 − n)) · σ(T · (n − 0.5))
        which peaks near n = 1 and falls off for n = 0 and n ≥ 2.

        Args:
            skel: (B, 1, H, W) soft skeleton in [0, 1].
        Returns:
            (B, 1, H, W) endpoint confidence map in [0, 1].
        """
        T = self.endpoint_temperature
        n = self._count_neighbors(skel)

        # Peaks at n ≈ 1; low for n = 0 (isolated pixel) and n ≥ 2 (branch / interior)
        indicator = (
            torch.sigmoid(T * (1.5 - n))
            * torch.sigmoid(T * (n - 0.5))
        )
        return indicator * skel  # only fire where skeleton exists

    # ------------------------------------------------------------------ #
    # Three loss components                                                #
    # ------------------------------------------------------------------ #

    def _skeleton_gap_loss(
        self, v_p: torch.Tensor, t_l: torch.Tensor
    ) -> torch.Tensor:
        """
        Skeleton Gap Loss.

        At every point where the GT skeleton (t_l) is active, the predicted
        probability (v_p) should also be high.  A break = v_p ≈ 0 while t_l ≈ 1.

            L_gap = Σ [ t_l · (1 − v_p) ] / ( Σ t_l + ε )

        Args:
            v_p: (B, 1, H, W) predicted class probability.
            t_l: (B, 1, H, W) soft GT skeleton.
        Returns:
            Scalar loss value in [0, 1].
        """
        gap = t_l * (1.0 - v_p)
        return gap.sum() / (t_l.sum() + self.smooth)

    def _endpoint_excess_loss(
        self, t_p: torch.Tensor, t_l: torch.Tensor
    ) -> torch.Tensor:
        """
        Endpoint Excess Loss.

        Each vessel break introduces 2 spurious endpoints in the predicted
        skeleton that do not exist in the GT skeleton.  Penalising the *excess*
        endpoint density is therefore directly proportional to break count.

            excess  = ReLU( endpoint_map(t_p) − endpoint_map(t_l) )
            L_ep    = Σ excess / ( Σ endpoint_map(t_l) + ε )

        The denominator normalises by the number of *legitimate* GT endpoints
        (vessel tips), making the loss scale-invariant.

        Args:
            t_p: (B, 1, H, W) soft predicted skeleton.
            t_l: (B, 1, H, W) soft GT skeleton.
        Returns:
            Scalar loss ≥ 0.
        """
        ep_pred = self._soft_endpoint_map(t_p)
        ep_gt   = self._soft_endpoint_map(t_l)

        excess = F.relu(ep_pred - ep_gt)
        return excess.sum() / (ep_gt.sum() + self.smooth)

    def _morphological_continuity_loss(
        self, v_p: torch.Tensor, t_l: torch.Tensor
    ) -> torch.Tensor:
        """
        Morphological Continuity Loss.

        The GT skeleton is dilated to build a tight "path corridor" around
        every vessel centreline.  Inside this corridor the predicted probability
        must remain high.  Any gap *within* the corridor that escaped the gap
        loss (e.g. due to soft skeleton values) is caught here.

            corridor   = soft_dilate(t_l)
            coverage   = Σ [ corridor · v_p ] / ( Σ corridor + ε )
            L_cont     = 1 − coverage

        Args:
            v_p: (B, 1, H, W) predicted class probability.
            t_l: (B, 1, H, W) soft GT skeleton.
        Returns:
            Scalar loss in [0, 1].
        """
        corridor  = soft_dilate(t_l)
        coverage  = (corridor * v_p).sum() / (corridor.sum() + self.smooth)
        return 1.0 - coverage

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        softmax: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            y_pred:  (B, C, H, W) raw logits (or probabilities if softmax=False).
            y_true:  (B, H, W)    integer class labels.
            softmax: Apply softmax to y_pred before processing.

        Returns:
            Scalar topology loss (weighted sum of the three components).
        """
        if softmax:
            y_pred = torch.softmax(y_pred, dim=1)

        y_true_oh = self._one_hot(y_true)   # (B, C, H, W)

        total_loss = torch.tensor(0.0, device=y_pred.device)
        count = 0

        for cls in range(self.n_classes):
            if cls == self.bg_index:
                continue

            v_p = y_pred[:, cls : cls + 1, ...]      # (B, 1, H, W) predicted prob
            v_l = y_true_oh[:, cls : cls + 1, ...]   # (B, 1, H, W) GT binary mask

            # Compute soft skeletons
            t_p = soft_skel(v_p, self.iters)   # predicted centreline
            t_l = soft_skel(v_l, self.iters)   # GT centreline

            # --- Component 1: skeleton gap ---
            l_gap = self._skeleton_gap_loss(v_p, t_l)

            # --- Component 2: endpoint excess ---
            l_ep = self._endpoint_excess_loss(t_p, t_l)

            # --- Component 3: morphological continuity ---
            l_cont = self._morphological_continuity_loss(v_p, t_l)

            class_loss = (
                self.w_gap        * l_gap
                + self.w_endpoint * l_ep
                + self.w_continuity * l_cont
            )
            total_loss = total_loss + class_loss
            count += 1

        return total_loss / count if count > 0 else total_loss
    