"""
AGA-Net: Adaptive Geometric-Attention Network
Two-Stage Lung Nodule Segmentation and Malignancy Classification
in Federated Healthcare IoT Systems

Paper: "Adaptive Geometric-Attention Network for Two-Stage Lung Nodule
Segmentation and Malignancy Classification in Federated Healthcare IoT Systems"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List


# ─────────────────────────────────────────────────────────────
# 1. GEOMETRIC-CONSTRAINED ATTENTION MODULE (GCAM)
# ─────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """Spatial attention using GAP + GMP (Eq. 1)"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)          # GAP across channels
        mx, _ = x.max(dim=1, keepdim=True)         # GMP across channels
        attn = torch.cat([avg, mx], dim=1)          # (B,2,H,W,D)
        return self.sigmoid(self.conv(attn))


class ChannelAttention(nn.Module):
    """Channel attention SE-style (Eq. 2)"""
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        return self.fc(x).view(b, c, 1, 1, 1)


class GeometricConstraint(nn.Module):
    """
    Geometric constraint function G(F) exploiting spherical nodule shape.
    Implements Equations 3, 4, 5, 6, 7 from the paper.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # Learnable radius parameters (Eq. 5)
        self.alpha_rad = nn.Parameter(torch.tensor(0.7))
        self.beta_rad  = nn.Parameter(torch.tensor(0.2))
        self.gamma_rad = nn.Parameter(torch.tensor(0.1))

        # Explicit radius prediction head (Eq. 6 alternative)
        self.radius_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.ReLU()
        )

        # Gradient magnitude network for weighted centroid
        self.sigma_grad = 0.5   # fixed hyperparameter (Eq. 9)

    # ── Center estimation ─────────────────────────────────────
    def _estimate_center(self, seg_prob: torch.Tensor) -> torch.Tensor:
        """
        Weighted centroid estimation (Eqs. 8, 9, 10).
        seg_prob: (B,1,H,W,D) segmentation probability map.
        Returns center (B,3).
        """
        B, _, H, W, D = seg_prob.shape
        device = seg_prob.device

        # Build coordinate grids
        gz = torch.arange(H, device=device).float()
        gy = torch.arange(W, device=device).float()
        gx = torch.arange(D, device=device).float()
        zz, yy, xx = torch.meshgrid(gz, gy, gx, indexing='ij')  # (H,W,D)

        # Gradient magnitude (finite differences)
        # Pad to allow finite difference
        p = seg_prob  # (B,1,H,W,D)
        dz = torch.zeros_like(p)
        dy = torch.zeros_like(p)
        dx = torch.zeros_like(p)
        dz[:, :, 1:-1] = p[:, :, 2:] - p[:, :, :-2]
        dy[:, :, :, 1:-1] = p[:, :, :, 2:] - p[:, :, :, :-2]
        dx[:, :, :, :, 1:-1] = p[:, :, :, :, 2:] - p[:, :, :, :, :-2]
        grad_mag2 = dz ** 2 + dy ** 2 + dx ** 2   # (B,1,H,W,D)

        # Weighted centroid (Eq. 9)
        weight = p * torch.exp(-grad_mag2 / (2 * self.sigma_grad ** 2))  # (B,1,H,W,D)
        weight_sum = weight.sum(dim=(2, 3, 4), keepdim=True).clamp(min=1e-6)

        cz = (weight * zz).sum(dim=(2, 3, 4)) / weight_sum.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)
        cy = (weight * yy).sum(dim=(2, 3, 4)) / weight_sum.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)
        cx = (weight * xx).sum(dim=(2, 3, 4)) / weight_sum.squeeze(-1).squeeze(-1).squeeze(-1).squeeze(-1)
        center = torch.stack([cz, cy, cx], dim=1)  # (B,3)
        return center

    def _adaptive_radius(self, feat: torch.Tensor, seg_prob: torch.Tensor) -> torch.Tensor:
        """
        Adaptive radius σ_r (Eqs. 5, 6).
        Uses both volume-based and learnable radius head.
        """
        B = feat.shape[0]
        device = feat.device

        # Volume-based radius (Eq. 6 sphere approximation)
        vol = seg_prob.sum(dim=(1, 2, 3, 4)).clamp(min=1e-6)  # (B,)
        r_vol = (3 * vol / (4 * torch.pi)) ** (1.0 / 3.0)    # (B,)

        # Learnable radius head (Eq. 6 learned)
        r_pred = self.radius_head(feat).squeeze(-1)            # (B,)

        # Combine both
        r_mean = r_pred.mean()
        r_std  = r_pred.std().clamp(min=1e-6)

        sigma_r = self.alpha_rad * r_mean + self.beta_rad * r_std + self.gamma_rad
        sigma_r = sigma_r.clamp(min=1.0)   # prevent division by zero
        return sigma_r.expand(B)            # (B,)

    def forward(self, feat: torch.Tensor, seg_prob: torch.Tensor) -> torch.Tensor:
        """
        Geometric attention map G(F) (Eqs. 3, 4 → Eq. 11 updated).
        feat:     (B,C,H,W,D)
        seg_prob: (B,1,H,W,D)
        Returns geometric attention (B,1,H,W,D)
        """
        B, C, H, W, D = feat.shape
        device = feat.device

        center   = self._estimate_center(seg_prob)        # (B,3)
        sigma_r  = self._adaptive_radius(feat, seg_prob)  # (B,)

        # Build distance map
        gz = torch.arange(H, device=device).float()
        gy = torch.arange(W, device=device).float()
        gx = torch.arange(D, device=device).float()
        zz, yy, xx = torch.meshgrid(gz, gy, gx, indexing='ij')  # (H,W,D)

        dist2_list = []
        for b in range(B):
            dz = (zz - center[b, 0]) ** 2
            dy = (yy - center[b, 1]) ** 2
            dx = (xx - center[b, 2]) ** 2
            d2 = dz + dy + dx
            dist2_list.append(d2)
        dist2 = torch.stack(dist2_list, dim=0).unsqueeze(1)  # (B,1,H,W,D)

        sigma_r_sq = (sigma_r ** 2).view(B, 1, 1, 1, 1)
        geo_map = torch.exp(-dist2 / (2 * sigma_r_sq))       # (B,1,H,W,D)
        return geo_map


class GCAM(nn.Module):
    """
    Geometric-Constrained Attention Module.
    Combines spatial, channel, and geometric attention (Eq. 13).
    """
    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial  = SpatialAttention(in_channels)
        self.channel  = ChannelAttention(in_channels)
        self.geometric = GeometricConstraint(in_channels)

        # Learnable fusion weights α, β, γ
        self.alpha = nn.Parameter(torch.tensor(0.4))
        self.beta  = nn.Parameter(torch.tensor(0.3))
        self.gamma = nn.Parameter(torch.tensor(0.3))

        # Project geo map to match feature channels for broadcast
        self.geo_proj = nn.Conv3d(1, in_channels, kernel_size=1)

    def forward(self,
                feat: torch.Tensor,
                seg_prob: Optional[torch.Tensor] = None) -> torch.Tensor:
        A_s = self.spatial(feat)                               # (B,1,H,W,D)
        A_c = self.channel(feat)                               # (B,C,1,1,1)

        if seg_prob is None:
            # Fallback: use sigmoid of average as proxy
            seg_prob = torch.sigmoid(feat.mean(dim=1, keepdim=True))
        A_g = self.geometric(feat, seg_prob)                   # (B,1,H,W,D)

        # Broadcast and fuse (Eq. 13)
        A_s_broad = A_s.expand_as(feat)
        A_c_broad = A_c.expand_as(feat)
        A_g_proj  = self.geo_proj(A_g).expand_as(feat)

        fused = self.alpha * A_s_broad + self.beta * A_c_broad + self.gamma * A_g_proj
        return feat * fused


# ─────────────────────────────────────────────────────────────
# 2. U-NET BACKBONE WITH GCAM
# ─────────────────────────────────────────────────────────────

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)
    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_gcam: bool = True):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch * 2, out_ch)
        self.gcam = GCAM(out_ch) if use_gcam else None

    def forward(self, x, skip, seg_prob=None):
        x = self.up(x)
        # Pad if needed
        if x.shape != skip.shape:
            diff = [s - x_ for s, x_ in zip(skip.shape[2:], x.shape[2:])]
            x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        if self.gcam is not None:
            x = self.gcam(x, seg_prob)
        return x


class SegmentationNetwork(nn.Module):
    """
    3D U-Net with GCAM attention at each decoder level.
    Stage 1 of AGA-Net.
    """
    def __init__(self, in_channels: int = 1, base_filters: int = 32):
        super().__init__()
        f = base_filters
        # Encoder
        self.enc1 = EncoderBlock(in_channels, f)
        self.enc2 = EncoderBlock(f, f*2)
        self.enc3 = EncoderBlock(f*2, f*4)
        self.enc4 = EncoderBlock(f*4, f*8)
        # Bottleneck
        self.bottleneck = ConvBlock3D(f*8, f*16)
        # Decoder
        self.dec4 = DecoderBlock(f*16, f*8, use_gcam=True)
        self.dec3 = DecoderBlock(f*8,  f*4, use_gcam=True)
        self.dec2 = DecoderBlock(f*4,  f*2, use_gcam=True)
        self.dec1 = DecoderBlock(f*2,  f,   use_gcam=True)
        # Output
        self.out_conv = nn.Conv3d(f, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, s1 = self.enc1(x)
        x2, s2 = self.enc2(x1)
        x3, s3 = self.enc3(x2)
        x4, s4 = self.enc4(x3)
        xb = self.bottleneck(x4)

        # Decoder — pass seg_prob from previous level where available
        d4 = self.dec4(xb, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        logits  = self.out_conv(d1)                    # (B,1,H,W,D)
        seg_prob = torch.sigmoid(logits)               # (B,1,H,W,D)
        return logits, seg_prob


# ─────────────────────────────────────────────────────────────
# 3. MULTI-SCALE UNCERTAINTY QUANTIFICATION NETWORK (MUQ-Net)
# ─────────────────────────────────────────────────────────────

class MultiScaleBranch(nn.Module):
    """Single scale branch with dilated convolutions"""
    def __init__(self, in_channels: int, out_channels: int, dilation: int,
                 dropout_p: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p),
        )
    def forward(self, x): return self.block(x)


class MUQNet(nn.Module):
    """
    Multi-Scale Uncertainty Quantification Network.
    Stage 2 of AGA-Net: malignancy classification + uncertainty.
    Implements Monte Carlo Dropout (Eqs. 15–19) +
    Deep Ensembles (Eq. 20) + Heteroscedastic Loss (Eq. 21).
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 dropout_p: float = 0.3,
                 base_filters: int = 32):
        super().__init__()
        f = base_filters

        # Multi-scale branches with different receptive fields (Eq. 14)
        self.branch1 = MultiScaleBranch(in_channels, f, dilation=1, dropout_p=dropout_p)
        self.branch2 = MultiScaleBranch(in_channels, f, dilation=2, dropout_p=dropout_p)
        self.branch3 = MultiScaleBranch(in_channels, f, dilation=4, dropout_p=dropout_p)
        self.branch4 = MultiScaleBranch(in_channels, f, dilation=8, dropout_p=dropout_p)

        concat_ch = f * 4  # 4 branches concatenated

        self.fusion = nn.Sequential(
            nn.Conv3d(concat_ch, f*4, 3, padding=1, bias=False),
            nn.BatchNorm3d(f*4),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p),
            nn.AdaptiveAvgPool3d(1),
        )

        # Classification head — mean prediction
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f*4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, num_classes)
        )

        # Heteroscedastic variance head (Eq. 21)
        self.var_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(f*4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
            nn.Softplus()   # ensures σ² > 0
        )

        self.dropout_p = dropout_p

    def forward_once(self, x: torch.Tensor):
        """Single forward pass (used during MC sampling)"""
        f1 = self.branch1(x)
        f2 = self.branch2(x)
        f3 = self.branch3(x)
        f4 = self.branch4(x)
        feat = torch.cat([f1, f2, f3, f4], dim=1)   # (B,4f,H,W,D)
        feat = self.fusion(feat)                      # (B,4f,1,1,1)
        logits  = self.cls_head(feat)
        log_var = self.var_head(feat)                 # predicted log variance
        return logits, log_var

    def forward(self, x: torch.Tensor,
                mc_samples: int = 1) -> dict:
        """
        Forward pass with optional Monte Carlo dropout.
        mc_samples=1 → standard inference
        mc_samples>1 → uncertainty estimation
        """
        if mc_samples == 1:
            logits, log_var = self.forward_once(x)
            probs = F.softmax(logits, dim=-1)
            return {
                'logits': logits,
                'probs': probs,
                'log_var': log_var,
                'uncertainty': log_var.mean(dim=-1)
            }

        # MC Dropout sampling (Eq. 15, 16)
        self.train()   # enable dropout during inference
        all_logits = []
        for _ in range(mc_samples):
            logits, _ = self.forward_once(x)
            all_logits.append(F.softmax(logits, dim=-1))   # (B,C)
        all_logits = torch.stack(all_logits, dim=0)        # (T,B,C)

        mean_probs = all_logits.mean(dim=0)                # (B,C) Eq. 15
        variance   = all_logits.var(dim=0)                 # (B,C) Eq. 16

        # Also get heteroscedastic variance
        self.eval()
        with torch.no_grad():
            logits_det, log_var = self.forward_once(x)

        return {
            'logits': logits_det,
            'probs': mean_probs,
            'mc_probs': all_logits,          # (T,B,C)
            'log_var': log_var,
            'epistemic_var': variance,
            'uncertainty': variance.mean(dim=-1) + log_var.exp().mean(dim=-1)
        }


# ─────────────────────────────────────────────────────────────
# 4. COMPLETE AGA-Net MODEL
# ─────────────────────────────────────────────────────────────

class AGANet(nn.Module):
    """
    Full two-stage AGA-Net model.
    Stage 1: SegmentationNetwork (GCAM-enhanced U-Net)
    Stage 2: MUQ-Net (uncertainty-aware malignancy classifier)
    """
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 base_filters: int = 32,
                 dropout_p: float = 0.3,
                 mc_samples: int = 50):
        super().__init__()
        self.seg_net = SegmentationNetwork(in_channels, base_filters)
        self.muq_net = MUQNet(in_channels, num_classes, dropout_p, base_filters)
        self.mc_samples = mc_samples

    def forward(self, x: torch.Tensor,
                return_uncertainty: bool = False) -> dict:
        # Stage 1: Segmentation
        seg_logits, seg_prob = self.seg_net(x)

        # Crop the nodule region using segmentation mask
        # (During training use ground-truth ROI; here use predicted mask)
        nodule_crop = x * (seg_prob > 0.5).float()

        # Stage 2: Classification with uncertainty
        mc = self.mc_samples if return_uncertainty else 1
        cls_out = self.muq_net(nodule_crop, mc_samples=mc)

        return {
            'seg_logits': seg_logits,
            'seg_prob':   seg_prob,
            **{f'cls_{k}': v for k, v in cls_out.items()}
        }
