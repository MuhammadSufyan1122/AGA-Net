"""
Loss Functions for AGA-Net
Implements all losses from the paper:
- Dice Loss (Eq. 24)
- Focal Loss (Eq. 25)
- Segmentation Loss (Eq. 23)
- Classification Loss with label smoothing (Eq. 26)
- Heteroscedastic Uncertainty Loss (Eq. 21)
- Calibration Loss (Eq. 28)
- Federated Regularization Loss (Eq. 29)
- Total Adaptive Loss (Eq. 22)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────
# Basic Components
# ─────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Dice loss for segmentation (Eq. 24)"""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B,1,H,W,D) — probabilities after sigmoid
        target: (B,1,H,W,D) — binary masks
        """
        pred   = pred.view(pred.shape[0], -1)
        target = target.view(target.shape[0], -1)
        intersection = (pred * target).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / \
               (pred.sum(dim=1) + target.sum(dim=1) + self.smooth)
        return (1.0 - dice).mean()


class FocalLoss(nn.Module):
    """
    Focal loss to handle class imbalance (Eq. 25).
    alpha_t(1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred_logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        pred_logits: (B,1,H,W,D) — raw logits
        target:      (B,1,H,W,D) — binary
        """
        pred_logits = pred_logits.view(-1)
        target      = target.view(-1)

        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        p_t = torch.where(target == 1,
                          torch.sigmoid(pred_logits),
                          1 - torch.sigmoid(pred_logits))
        alpha_t = torch.where(target == 1,
                              torch.full_like(p_t, self.alpha),
                              torch.full_like(p_t, 1 - self.alpha))
        focal = alpha_t * ((1 - p_t) ** self.gamma) * bce
        return focal.mean()


class SegmentationLoss(nn.Module):
    """Combined Dice + Focal segmentation loss (Eq. 23)"""
    def __init__(self, alpha_focal: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.dice  = DiceLoss()
        self.focal = FocalLoss(alpha=alpha_focal, gamma=gamma)

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        return self.dice(prob, target) + self.focal(logits, target)


class ClassificationLoss(nn.Module):
    """
    Cross-entropy with label smoothing (Eq. 26).
    """
    def __init__(self, num_classes: int = 2,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.num_classes    = num_classes
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C)
        target: (B,) — class indices
        """
        # Standard cross-entropy component
        ce = F.cross_entropy(logits, target, reduction='mean')

        # Label smoothing component (Eq. 26, second term)
        log_prob = F.log_softmax(logits, dim=-1)
        smooth   = -log_prob.mean(dim=-1).mean()

        return (1 - self.label_smoothing) * ce + self.label_smoothing * smooth


class HeteroscedasticLoss(nn.Module):
    """
    Heteroscedastic uncertainty loss (Eq. 21):
    L = Σ [ ||y - ŷ||² / (2σ²) + ½ log(σ²) ]
    """
    def forward(self, logits: torch.Tensor,
                log_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C)
        log_var: (B, C) — predicted log variance
        target:  (B,) — class indices
        """
        B, C = logits.shape
        # One-hot target
        y_oh = F.one_hot(target, C).float()

        probs       = F.softmax(logits, dim=-1)
        diff_sq     = (y_oh - probs) ** 2
        var         = log_var.exp().clamp(min=1e-6)

        hetero = (diff_sq / (2 * var) + 0.5 * log_var).mean()
        return hetero


class CalibrationLoss(nn.Module):
    """
    Calibration uncertainty loss (Eq. 28).
    Penalises overconfident incorrect predictions.
    """
    def forward(self, probs: torch.Tensor,
                log_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        probs:   (B, C)
        log_var: (B, C)
        target:  (B,) — class indices
        """
        B, C    = probs.shape
        y_oh    = F.one_hot(target, C).float()
        var     = log_var.exp().clamp(min=1e-6)

        # (p_i - p̂_i)² / σ² + log σ²  (Eq. 28)
        calib = ((probs - y_oh) ** 2 / var + log_var).mean()
        return calib


class UncertaintyLoss(nn.Module):
    """
    Combined uncertainty loss (Eq. 21 + 28).
    λ_hetero * L_hetero + λ_calib * L_calib
    """
    def __init__(self, lam_hetero: float = 0.6,
                 lam_calib: float = 0.4):
        super().__init__()
        self.hetero = HeteroscedasticLoss()
        self.calib  = CalibrationLoss()
        self.lam_hetero = lam_hetero
        self.lam_calib  = lam_calib

    def forward(self, logits: torch.Tensor,
                log_var: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        lh = self.hetero(logits, log_var, target)
        lc = self.calib(probs, log_var, target)
        return self.lam_hetero * lh + self.lam_calib * lc


class FederatedRegularizationLoss(nn.Module):
    """
    Federated regularization loss (Eq. 29):
    L_fed = μ ||θ_local − θ_global||² + λ_dp ||N(0,σ_dp²)||²
    """
    def __init__(self, mu: float = 0.1,
                 lam_dp: float = 0.01):
        super().__init__()
        self.mu     = mu
        self.lam_dp = lam_dp

    def forward(self, local_params: list,
                global_params: list,
                sigma_dp: float) -> torch.Tensor:
        drift = sum(
            ((lp - gp.detach()) ** 2).sum()
            for lp, gp in zip(local_params, global_params)
        )
        dp_noise_sq = torch.tensor(sigma_dp ** 2,
                                   device=local_params[0].device)
        return self.mu * drift + self.lam_dp * dp_noise_sq


# ─────────────────────────────────────────────────────────────
# Adaptive Loss Weighter (Eqs. 30–34)
# ─────────────────────────────────────────────────────────────

class AdaptiveLossWeighter:
    """
    Dynamically adjusts λ_1..λ_4 throughout training.
    Implements magnitude normalisation (Eq. 31),
    convergence-based adjustment (Eq. 32),
    and federated heterogeneity adjustment (Eq. 33).
    """

    def __init__(self,
                 init_weights=(1.0, 0.8, 0.3, 0.1),
                 ema_decay: float = 0.9,
                 beta: float = 0.1,
                 gamma: float = 0.2,
                 window: int = 10):
        self.lambdas    = list(init_weights)
        self.ema        = [1.0, 1.0, 1.0, 1.0]   # EMA of |L_i|
        self.history    = [[] for _ in range(4)]   # sliding window
        self.ema_decay  = ema_decay
        self.beta       = beta
        self.gamma      = gamma
        self.window     = window
        self.init_w     = list(init_weights)

    def update(self, losses: list,
               client_loss_var: float = 0.0):
        """
        losses: [L_seg, L_cls, L_unc, L_fed]  — scalar floats
        client_loss_var: variance of losses across FL clients
        """
        for i, l in enumerate(losses):
            # EMA update
            self.ema[i] = self.ema_decay * self.ema[i] + \
                          (1 - self.ema_decay) * abs(l)
            self.history[i].append(l)
            if len(self.history[i]) > self.window:
                self.history[i].pop(0)

        # 1. Magnitude normalisation (Eq. 31)
        lam_base = [self.init_w[i] / max(self.ema[i], 1e-6)
                    for i in range(4)]

        # 2. Convergence-based adjustment (Eq. 32)
        lam_adj = []
        for i in range(4):
            if len(self.history[i]) >= 2:
                dl_dt = (self.history[i][-1] - self.history[i][0]) / \
                        max(len(self.history[i]) - 1, 1)
            else:
                dl_dt = 0.0
            lam_adj.append(lam_base[i] * (1 + self.beta * dl_dt))

        # 3. Federated heterogeneity adjustment for λ_4 (Eq. 33)
        lam_adj[3] = lam_base[3] * (1 + self.gamma * client_loss_var)

        # Clamp to positive
        self.lambdas = [max(l, 1e-4) for l in lam_adj]

    @property
    def weights(self):
        return tuple(self.lambdas)


# ─────────────────────────────────────────────────────────────
# Master Loss Combiner
# ─────────────────────────────────────────────────────────────

class AGANetLoss(nn.Module):
    """
    Full combined loss for AGA-Net training (Eq. 22 / 30).
    L_total = λ1·L_seg + λ2·L_cls + λ3·L_unc + λ4·L_fed
    """
    def __init__(self,
                 num_classes: int = 2,
                 init_weights=(1.0, 0.8, 0.3, 0.1),
                 label_smoothing: float = 0.1,
                 lam_hetero: float = 0.6,
                 lam_calib: float = 0.4,
                 mu_fed: float = 0.1,
                 lam_dp: float = 0.01,
                 adaptive: bool = True):
        super().__init__()
        self.seg_loss  = SegmentationLoss()
        self.cls_loss  = ClassificationLoss(num_classes, label_smoothing)
        self.unc_loss  = UncertaintyLoss(lam_hetero, lam_calib)
        self.fed_loss  = FederatedRegularizationLoss(mu_fed, lam_dp)
        self.weighter  = AdaptiveLossWeighter(init_weights) if adaptive \
                         else None
        self.fixed_w   = init_weights

    def forward(self,
                output: dict,
                seg_target: torch.Tensor,
                cls_target: torch.Tensor,
                local_params: Optional[list] = None,
                global_params: Optional[list] = None,
                sigma_dp: float = 0.19,
                client_loss_var: float = 0.0) -> dict:
        """
        output: dict from AGANet.forward()
        seg_target: (B,1,H,W,D) binary mask
        cls_target: (B,) long tensor
        """
        # Stage 1 losses
        l_seg = self.seg_loss(output['seg_logits'], seg_target)

        # Stage 2 losses
        l_cls = self.cls_loss(output['cls_logits'], cls_target)
        l_unc = self.unc_loss(output['cls_logits'],
                              output['cls_log_var'], cls_target)

        # Federated loss
        if local_params is not None and global_params is not None:
            l_fed = self.fed_loss(local_params, global_params, sigma_dp)
        else:
            l_fed = torch.tensor(0.0, device=l_seg.device)

        losses = [l_seg.item(), l_cls.item(),
                  l_unc.item(), l_fed.item() if isinstance(l_fed, torch.Tensor) else l_fed]

        # Determine weights
        if self.weighter is not None:
            self.weighter.update(losses, client_loss_var)
            w = self.weighter.weights
        else:
            w = self.fixed_w

        total = w[0]*l_seg + w[1]*l_cls + w[2]*l_unc + w[3]*l_fed

        return {
            'total': total,
            'seg':   l_seg,
            'cls':   l_cls,
            'unc':   l_unc,
            'fed':   l_fed,
            'weights': w
        }
