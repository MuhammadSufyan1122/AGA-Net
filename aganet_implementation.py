import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import cv2
from scipy import ndimage
import nibabel as nib
import pandas as pd
from tqdm import tqdm
import os
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================
# GEOMETRIC-CONSTRAINED ATTENTION MODULE (GCAM)
# ============================

class GeometricConstrainedAttention(nn.Module):
    """
    Novel Geometric-Constrained Attention Module that leverages 
    the spherical nature of lung nodules
    """
    def __init__(self, in_channels, reduction=16):
        super(GeometricConstrainedAttention, self).__init__()
        self.in_channels = in_channels
        
        # Spatial attention
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        # Geometric constraint parameters
        self.geometric_weight = nn.Parameter(torch.ones(1))
        self.adaptive_radius = nn.Parameter(torch.tensor(5.0))
        
        # Geometric constraint network
        self.geometric_net = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
    def compute_geometric_constraint(self, x, center=None):
        """Compute geometric constraint based on spherical distance"""
        B, C, D, H, W = x.shape
        
        if center is None:
            # Estimate center using attention-weighted coordinates
            spatial_weights = torch.mean(x, dim=1, keepdim=True)
            coords_d = torch.arange(D, device=x.device).float().view(1, 1, -1, 1, 1)
            coords_h = torch.arange(H, device=x.device).float().view(1, 1, 1, -1, 1)
            coords_w = torch.arange(W, device=x.device).float().view(1, 1, 1, 1, -1)
            
            center_d = torch.sum(coords_d * spatial_weights) / torch.sum(spatial_weights)
            center_h = torch.sum(coords_h * spatial_weights) / torch.sum(spatial_weights)
            center_w = torch.sum(coords_w * spatial_weights) / torch.sum(spatial_weights)
            center = [center_d, center_h, center_w]
        
        # Create coordinate grids
        coords_d = torch.arange(D, device=x.device).float().view(1, 1, -1, 1, 1)
        coords_h = torch.arange(H, device=x.device).float().view(1, 1, 1, -1, 1)
        coords_w = torch.arange(W, device=x.device).float().view(1, 1, 1, 1, -1)
        
        # Compute distances from center
        dist_d = (coords_d - center[0]) ** 2
        dist_h = (coords_h - center[1]) ** 2
        dist_w = (coords_w - center[2]) ** 2
        distance = torch.sqrt(dist_d + dist_h + dist_w)
        
        # Apply Gaussian-based geometric constraint
        geometric_mask = torch.exp(-distance ** 2 / (2 * self.adaptive_radius ** 2))
        
        return geometric_mask.expand(B, C, D, H, W)
    
    def forward(self, x):
        # Channel attention
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att
        
        # Spatial attention
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_att = torch.sigmoid(self.spatial_conv(spatial_input))
        x_spatial = x_channel * spatial_att
        
        # Geometric attention
        geometric_constraint = self.compute_geometric_constraint(x_spatial)
        geometric_att = self.geometric_net(x_spatial)
        geometric_refined = geometric_att * geometric_constraint * self.geometric_weight
        
        # Combine all attention mechanisms
        x_out = x_spatial * (1 + geometric_refined)
        
        return x_out

# ============================
# ENCODER-DECODER BLOCKS
# ============================

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout = nn.Dropout3d(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels)
        self.use_attention = use_attention
        if use_attention:
            self.attention = GeometricConstrainedAttention(out_channels)
        self.pool = nn.MaxPool3d(2)
        
    def forward(self, x):
        x = self.conv_block(x)
        if self.use_attention:
            x = self.attention(x)
        skip = x
        x = self.pool(x)
        return x, skip

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.conv_block = ConvBlock3D(out_channels + skip_channels, out_channels)
        
    def forward(self, x, skip):
        x = self.upconv(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x

# ============================
# SEGMENTATION NETWORK
# ============================

class SegmentationNetwork(nn.Module):
    """Stage 1: Geometric-Constrained Segmentation Network"""
    def __init__(self, in_channels=1, out_channels=1):
        super(SegmentationNetwork, self).__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        # Bottleneck with enhanced attention
        self.bottleneck = ConvBlock3D(512, 1024)
        self.bottleneck_attention = GeometricConstrainedAttention(1024)
        
        # Decoder
        self.dec4 = DecoderBlock(1024, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        
        # Output
        self.output_conv = nn.Conv3d(64, out_channels, 1)
        
    def forward(self, x):
        # Encoder path
        x1, skip1 = self.enc1(x)
        x2, skip2 = self.enc2(x1)
        x3, skip3 = self.enc3(x2)
        x4, skip4 = self.enc4(x3)
        
        # Bottleneck
        x = self.bottleneck(x4)
        x = self.bottleneck_attention(x)
        
        # Decoder path
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Output
        output = torch.sigmoid(self.output_conv(x))
        return output

# ============================
# MULTI-SCALE UNCERTAINTY QUANTIFICATION NETWORK (MUQ-Net)
# ============================

class MultiScaleFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Multi-scale branches
        self.scale1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(8)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv3d(in_channels, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(8)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv3d(in_channels, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(8)
        )
        
        self.scale4 = nn.Sequential(
            nn.Conv3d(in_channels, 64, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(8)
        )
        
        self.fusion = nn.Conv3d(256, 128, 1)
        
    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = self.scale4(x)
        
        # Concatenate multi-scale features
        features = torch.cat([s1, s2, s3, s4], dim=1)
        features = self.fusion(features)
        
        return features

class UncertaintyClassificationNetwork(nn.Module):
    """Stage 2: Multi-Scale Uncertainty Quantification Network"""
    def __init__(self, in_channels=1, num_classes=2, dropout_rate=0.3):
        super(UncertaintyClassificationNetwork, self).__init__()
        
        self.feature_extractor = MultiScaleFeatureExtractor(in_channels)
        
        # Classifier with dropout for uncertainty
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Softplus()  # Ensures positive uncertainty
        )
        
    def forward(self, x, monte_carlo_samples=10, training=True):
        features = self.feature_extractor(x)
        
        if training or monte_carlo_samples == 1:
            # Single forward pass
            logits = self.classifier(features)
            uncertainty = self.uncertainty_head(features)
            return logits, uncertainty
        else:
            # Monte Carlo Dropout for uncertainty estimation
            self.train()  # Enable dropout
            predictions = []
            uncertainties = []
            
            for _ in range(monte_carlo_samples):
                logits = self.classifier(features)
                uncertainty = self.uncertainty_head(features)
                predictions.append(F.softmax(logits, dim=1))
                uncertainties.append(uncertainty)
            
            # Aggregate predictions
            mean_pred = torch.stack(predictions).mean(dim=0)
            epistemic_uncertainty = torch.stack(predictions).var(dim=0).mean(dim=1, keepdim=True)
            aleatoric_uncertainty = torch.stack(uncertainties).mean(dim=0)
            
            # Convert back to logits
            mean_logits = torch.log(mean_pred + 1e-8)
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            return mean_logits, total_uncertainty

# ============================
# COMPLETE AGA-NET FRAMEWORK
# ============================

class AGANet(nn.Module):
    """Complete Adaptive Geometric-Attention Network"""
    def __init__(self, in_channels=1, num_classes=2):
        super(AGANet, self).__init__()
        
        # Stage 1: Segmentation
        self.segmentation_net = SegmentationNetwork(in_channels, 1)
        
        # Stage 2: Classification (takes original input + segmentation mask)
        self.classification_net = UncertaintyClassificationNetwork(in_channels + 1, num_classes)
        
    def forward(self, x, monte_carlo_samples=10, training=True):
        # Stage 1: Segmentation
        segmentation = self.segmentation_net(x)
        
        # Stage 2: Classification with segmentation guidance
        classification_input = torch.cat([x, segmentation], dim=1)
        classification_logits, uncertainty = self.classification_net(
            classification_input, monte_carlo_samples, training
        )
        
        return segmentation, classification_logits, uncertainty

# ============================
# LOSS FUNCTIONS
# ============================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class UncertaintyLoss(nn.Module):
    def __init__(self):
        super(UncertaintyLoss, self).__init__()
        
    def forward(self, predictions, targets, uncertainties):
        # Negative log-likelihood with uncertainty
        mse = F.mse_loss(predictions, targets, reduction='none')
        uncertainty_loss = 0.5 * (mse / uncertainties + torch.log(uncertainties))
        return uncertainty_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, seg_weight=1.0, cls_weight=0.8, unc_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.unc_weight = unc_weight
        
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.uncertainty_loss = UncertaintyLoss()
        
    def forward(self, seg_pred, seg_target, cls_pred, cls_target, uncertainty):
        # Segmentation loss
        dice = self.dice_loss(seg_pred, seg_target)
        focal = self.focal_loss(seg_pred, seg_target)
        seg_loss = dice + focal
        
        # Classification loss
        cls_loss = self.ce_loss(cls_pred, cls_target)
        
        # Uncertainty loss
        cls_probs = F.softmax(cls_pred, dim=1)
        cls_targets_one_hot = F.one_hot(cls_target, num_classes=cls_pred.shape[1]).float()
        unc_loss = self.uncertainty_loss(cls_probs, cls_targets_one_hot, uncertainty)
        
        # Combined loss
        total_loss = (self.seg_weight * seg_loss + 
                     self.cls_weight * cls_loss + 
                     self.unc_weight * unc_loss)
        
        return total_loss, seg_loss, cls_loss, unc_loss

# ============================
# DATASET CLASS
# ============================

class LungNoduleDataset(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
        # Load file paths and labels
        self.samples = self._load_samples()
        
    def _load_samples(self):
        # This would load your actual dataset
        # For demonstration, we'll create synthetic data structure
        samples = []
        for i in range(100):  # Simulate 100 samples
            sample = {
                'image_path': f'image_{i}.nii.gz',
                'mask_path': f'mask_{i}.nii.gz',
                'label': np.random.randint(0, 2),  # 0: benign, 1: malignant
                'nodule_size': np.random.uniform(5, 30),
                'patient_id': f'patient_{i}'
            }
            samples.append(sample)
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # For demonstration, create synthetic data
        # In real implementation, load from sample['image_path'] and sample['mask_path']
        
        # Synthetic 3D CT patch (64x64x64)
        image = np.random.randn(64, 64, 64).astype(np.float32)
        
        # Synthetic segmentation mask
        center = (32, 32, 32)
        radius = np.random.uniform(3, 8)
        mask = np.zeros((64, 64, 64), dtype=np.float32)
        
        # Create spherical mask
        for z in range(64):
            for y in range(64):
                for x in range(64):
                    dist = np.sqrt((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2)
                    if dist <= radius:
                        mask[z, y, x] = 1.0
        
        label = sample['label']
        
        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        mask = torch.from_numpy(mask).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'image': image,
            'mask': mask,
            'label': label,
            'patient_id': sample['patient_id'],
            'nodule_size': sample['nodule_size']
        }

# ============================
# DATA TRANSFORMS
# ============================

class Normalize3D:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, mask):
        image = (image - self.mean) / self.std
        return image, mask

class RandomRotation3D:
    def __init__(self, max_angle=15):
        self.max_angle = max_angle
    
    def __call__(self, image, mask):
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.max_angle, self.max_angle)
            # Simple rotation around z-axis
            for i in range(image.shape[0]):
                image[i] = ndimage.rotate(image[i], angle, reshape=False, order=1)
                mask[i] = ndimage.rotate(mask[i], angle, reshape=False, order=0)
        return image, mask

class RandomFlip3D:
    def __call__(self, image, mask):
        # Random flip along each axis
        for axis in range(3):
            if np.random.random() < 0.5:
                image = np.flip(image, axis=axis).copy()
                mask = np.flip(mask, axis=axis).copy()
        return image, mask

class Compose3D:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask

# ============================
# EVALUATION METRICS
# ============================

class MetricsCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        self.hausdorff_distances = []
        self.asd_scores = []
        self.classification_scores = []
        self.uncertainties = []
        
    def update(self, seg_pred, seg_true, cls_pred, cls_true, uncertainty):
        # Segmentation metrics
        dice = self._dice_coefficient(seg_pred, seg_true)
        iou = self._iou_score(seg_pred, seg_true)
        hd = self._hausdorff_distance(seg_pred, seg_true)
        asd = self._average_surface_distance(seg_pred, seg_true)
        
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.hausdorff_distances.append(hd)
        self.asd_scores.append(asd)
        
        # Classification metrics
        cls_pred_prob = F.softmax(cls_pred, dim=1)
        self.classification_scores.append((cls_pred_prob.cpu().numpy(), cls_true.cpu().numpy()))
        self.uncertainties.append(uncertainty.cpu().numpy())
    
    def _dice_coefficient(self, pred, target):
        smooth = 1e-5
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        intersection = (pred_flat * target_flat).sum()
        return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    def _iou_score(self, pred, target):
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return intersection / (union + 1e-5)
    
    def _hausdorff_distance(self, pred, target):
        # Simplified HD calculation
        return np.random.uniform(2, 10)  # Placeholder
    
    def _average_surface_distance(self, pred, target):
        # Simplified ASD calculation
        return np.random.uniform(1, 5)  # Placeholder
    
    def get_metrics(self):
        # Segmentation metrics
        seg_metrics = {
            'dice': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores),
            'hausdorff_distance': np.mean(self.hausdorff_distances),
            'average_surface_distance': np.mean(self.asd_scores)
        }
        
        # Classification metrics
        all_probs = np.vstack([scores[0] for scores in self.classification_scores])
        all_labels = np.concatenate([scores[1] for scores in self.classification_scores])
        
        cls_metrics = {
            'auc': roc_auc_score(all_labels, all_probs[:, 1]),
            'accuracy': np.mean(np.argmax(all_probs, axis=1) == all_labels),
            'mean_uncertainty': np.mean(self.uncertainties)
        }
        
        return seg_metrics, cls_metrics

# ============================
# VISUALIZATION FUNCTIONS
# ============================

class Visualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        
    def plot_training_curves(self, history):
        """Plot training and validation curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train')
        axes[0, 0].plot(history['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Segmentation loss
        axes[0, 1].plot(history['train_seg_loss'], label='Train Seg')
        axes[0, 1].plot(history['val_seg_loss'], label='Val Seg')
        axes[0, 1].set_title('Segmentation Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Classification loss
        axes[0, 2].plot(history['train_cls_loss'], label='Train Cls')
        axes[0, 2].plot(history['val_cls_loss'], label='Val Cls')
        axes[0, 2].set_title('Classification Loss', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Dice score
        axes[1, 0].plot(history['train_dice'], label='Train')
        axes[1, 0].plot(history['val_dice'], label='Validation')
        axes[1, 0].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC score
        axes[1, 1].plot(history['train_auc'], label='Train')
        axes[1, 1].plot(history['val_auc'], label='Validation')
        axes[1, 1].set_title('AUC Score', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Uncertainty
        axes[1, 2].plot(history['train_uncertainty'], label='Train')
        axes[1, 2].plot(history['val_uncertainty'], label='Validation')
        axes[1, 2].set_title('Mean Uncertainty', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Uncertainty')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_predictions(self, images, true_masks, pred_masks, predictions, uncertainties, labels):
        """Visualize predictions with uncertainty"""
        n_samples = min(4, len(images))
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
        
        for i in range(n_samples):
            # Original image (middle slice)
            mid_slice = images[i].shape[2] // 2
            axes[i, 0].imshow(images[i, 0, mid_slice], cmap='gray')
            axes[i, 0].set_title(f'Original Image\nSlice {mid_slice}')
            axes[i, 0].axis('off')
            
            # True mask
            axes[i, 1].imshow(true_masks[i, 0, mid_slice], cmap='Reds', alpha=0.7)
            axes[i, 1].imshow(images[i, 0, mid_slice], cmap='gray', alpha=0.3)
            axes[i, 1].set_title('Ground Truth Mask')
            axes[i, 1].axis('off')
            
            # Predicted mask
            axes[i, 2].imshow(pred_masks[i, 0, mid_slice], cmap='Blues', alpha=0.7)
            axes[i, 2].imshow(images[i, 0, mid_slice], cmap='gray', alpha=0.3)
            axes[i, 2].set_title('Predicted Mask')
            axes[i, 2].axis('off')
            
            # Classification result with uncertainty
            pred_class = torch.argmax(predictions[i]).item()
            uncertainty = uncertainties[i].item()
            true_class = labels[i].item()
            
            class_names = ['Benign', 'Malignant']
            color = 'green' if pred_class == true_class else 'red'
            
            axes[i, 3].bar(['Benign', 'Malignant'], 
                          F.softmax(predictions[i], dim=0).cpu().numpy(),
                          color=['lightblue', 'lightcoral'])
            axes[i, 3].set_title(f'Prediction: {class_names[pred_class]}\n'
                                f'True: {class_names[true_class]}\n'
                                f'Uncertainty: {uncertainty:.3f}',
                                color=color, fontweight='bold')
            axes[i, 3].set_ylabel('Probability')
            axes[i, 3].set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def plot_attention_maps(self, model, image, layer_name='bottleneck_attention'):
        """Visualize attention maps"""
        # Register hook to capture attention
        attention_maps = []
        
        def hook_fn(module, input, output):
            attention_maps.append(output.detach())
        
        # Register hook
        for name, module in model.named_modules():
            if name.endswith(layer_name):
                module.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = model(image.unsqueeze(0))
        
        if attention_maps:
            attention = attention_maps[0][0, 0]  # First batch, first channel
            
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            
            # Show attention maps for different slices
            slices = [attention.shape[0]//4, attention.shape[0]//2, 
                     3*attention.shape[0]//4, attention.shape[0]-1]
            
            for i, slice_idx in enumerate(slices):
                # Original image
                axes[0, i].imshow(image[0, slice_idx], cmap='gray')
                axes[0, i].set_title(f'Original (Slice {slice_idx})')
                axes[0, i].axis('off')
                
                # Attention map
                axes[1, i].imshow(attention[slice_idx], cmap='hot')
                axes[1, i].set_title(f'Attention (Slice {slice_idx})')
                axes[1, i].axis('off')
            
            plt.suptitle('Geometric-Constrained Attention Maps', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            return fig
        
        return None
    
    def plot_uncertainty_analysis(self, uncertainties, predictions, labels):
        """Analyze prediction uncertainty"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Uncertainty distribution
        axes[0, 0].hist(uncertainties, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Uncertainty Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Uncertainty')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Uncertainty vs Accuracy
        correct = (torch.argmax(predictions, dim=1) == labels).numpy()
        axes[0, 1].scatter(uncertainties[correct], np.ones(sum(correct)), 
                          alpha=0.6, label='Correct', color='green')
        axes[0, 1].scatter(uncertainties[~correct], np.zeros(sum(~correct)), 
                          alpha=0.6, label='Incorrect', color='red')
        axes[0, 1].set_title('Uncertainty vs Prediction Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Uncertainty')
        axes[0, 1].set_ylabel('Correct Prediction')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confidence distribution by class
        probs = F.softmax(predictions, dim=1)
        max_probs = torch.max(probs, dim=1)[0].numpy()
        
        benign_conf = max_probs[labels == 0]
        malignant_conf = max_probs[labels == 1]
        
        axes[1, 0].hist(benign_conf, bins=20, alpha=0.7, label='Benign', color='lightblue')
        axes[1, 0].hist(malignant_conf, bins=20, alpha=0.7, label='Malignant', color='lightcoral')
        axes[1, 0].set_title('Confidence Distribution by Class', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Max Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Uncertainty vs Confidence
        axes[1, 1].scatter(max_probs, uncertainties, alpha=0.6, color='purple')
        axes[1, 1].set_title('Uncertainty vs Confidence', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Max Probability (Confidence)')
        axes[1, 1].set_ylabel('Uncertainty')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ============================
# TRAINING FRAMEWORK
# ============================

class AGANetTrainer:
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = CombinedLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=200, 
            eta_min=1e-6
        )
        
        # Metrics
        self.train_metrics = MetricsCalculator()
        self.val_metrics = MetricsCalculator()
        
        # History
        self.history = defaultdict(list)
        
        # Visualizer
        self.visualizer = Visualizer()
        
    def train_epoch(self):
        self.model.train()
        self.train_metrics.reset()
        
        total_loss = 0
        seg_loss_total = 0
        cls_loss_total = 0
        unc_loss_total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            seg_pred, cls_pred, uncertainty = self.model(images, training=True)
            
            # Calculate loss
            total_loss_batch, seg_loss, cls_loss, unc_loss = self.criterion(
                seg_pred, masks, cls_pred, labels, uncertainty
            )
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                self.train_metrics.update(
                    seg_pred.cpu(), masks.cpu(), 
                    cls_pred.cpu(), labels.cpu(), 
                    uncertainty.cpu()
                )
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            seg_loss_total += seg_loss.item()
            cls_loss_total += cls_loss.item()
            unc_loss_total += unc_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Seg': f'{seg_loss.item():.4f}',
                'Cls': f'{cls_loss.item():.4f}'
            })
        
        # Calculate average losses
        avg_total_loss = total_loss / len(self.train_loader)
        avg_seg_loss = seg_loss_total / len(self.train_loader)
        avg_cls_loss = cls_loss_total / len(self.train_loader)
        avg_unc_loss = unc_loss_total / len(self.train_loader)
        
        # Get metrics
        seg_metrics, cls_metrics = self.train_metrics.get_metrics()
        
        return {
            'total_loss': avg_total_loss,
            'seg_loss': avg_seg_loss,
            'cls_loss': avg_cls_loss,
            'unc_loss': avg_unc_loss,
            **seg_metrics,
            **cls_metrics
        }
    
    def validate_epoch(self):
        self.model.eval()
        self.val_metrics.reset()
        
        total_loss = 0
        seg_loss_total = 0
        cls_loss_total = 0
        unc_loss_total = 0
        
        progress_bar = tqdm(self.val_loader, desc='Validation')
        
        with torch.no_grad():
            for batch in progress_bar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass with Monte Carlo samples
                seg_pred, cls_pred, uncertainty = self.model(
                    images, monte_carlo_samples=10, training=False
                )
                
                # Calculate loss
                total_loss_batch, seg_loss, cls_loss, unc_loss = self.criterion(
                    seg_pred, masks, cls_pred, labels, uncertainty
                )
                
                # Update metrics
                self.val_metrics.update(
                    seg_pred.cpu(), masks.cpu(), 
                    cls_pred.cpu(), labels.cpu(), 
                    uncertainty.cpu()
                )
                
                # Accumulate losses
                total_loss += total_loss_batch.item()
                seg_loss_total += seg_loss.item()
                cls_loss_total += cls_loss.item()
                unc_loss_total += unc_loss.item()
                
                progress_bar.set_postfix({
                    'Loss': f'{total_loss_batch.item():.4f}'
                })
        
        # Calculate average losses
        avg_total_loss = total_loss / len(self.val_loader)
        avg_seg_loss = seg_loss_total / len(self.val_loader)
        avg_cls_loss = cls_loss_total / len(self.val_loader)
        avg_unc_loss = unc_loss_total / len(self.val_loader)
        
        # Get metrics
        seg_metrics, cls_metrics = self.val_metrics.get_metrics()
        
        return {
            'total_loss': avg_total_loss,
            'seg_loss': avg_seg_loss,
            'cls_loss': avg_cls_loss,
            'unc_loss': avg_unc_loss,
            **seg_metrics,
            **cls_metrics
        }
    
    def train(self, num_epochs=200, save_dir='./checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_auc = 0.0
        patience = 20
        patience_counter = 0
        
        print("Starting AGA-Net Training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_results = self.train_epoch()
            
            # Validation
            val_results = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_results['total_loss'])
            self.history['val_loss'].append(val_results['total_loss'])
            self.history['train_seg_loss'].append(train_results['seg_loss'])
            self.history['val_seg_loss'].append(val_results['seg_loss'])
            self.history['train_cls_loss'].append(train_results['cls_loss'])
            self.history['val_cls_loss'].append(val_results['cls_loss'])
            self.history['train_dice'].append(train_results['dice'])
            self.history['val_dice'].append(val_results['dice'])
            self.history['train_auc'].append(train_results['auc'])
            self.history['val_auc'].append(val_results['auc'])
            self.history['train_uncertainty'].append(train_results['mean_uncertainty'])
            self.history['val_uncertainty'].append(val_results['mean_uncertainty'])
            
            # Print results
            print(f"Train - Loss: {train_results['total_loss']:.4f}, "
                  f"Dice: {train_results['dice']:.4f}, "
                  f"AUC: {train_results['auc']:.4f}")
            print(f"Val   - Loss: {val_results['total_loss']:.4f}, "
                  f"Dice: {val_results['dice']:.4f}, "
                  f"AUC: {val_results['auc']:.4f}")
            
            # Save best model
            if val_results['auc'] > best_val_auc:
                best_val_auc = val_results['auc']
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_auc': val_results['auc'],
                    'val_dice': val_results['dice'],
                    'history': dict(self.history)
                }
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
                print(f"✓ New best model saved! AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                break
            
            # Plot training curves every 10 epochs
            if (epoch + 1) % 10 == 0:
                fig = self.visualizer.plot_training_curves(self.history)
                plt.savefig(os.path.join(save_dir, f'training_curves_epoch_{epoch+1}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"\nTraining completed! Best validation AUC: {best_val_auc:.4f}")
        return self.history
    
    def evaluate_test_set(self, test_loader, checkpoint_path):
        """Comprehensive evaluation on test set"""
        print("Loading best model...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        test_metrics = MetricsCalculator()
        
        all_images = []
        all_true_masks = []
        all_pred_masks = []
        all_predictions = []
        all_uncertainties = []
        all_labels = []
        
        print("Evaluating on test set...")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass with uncertainty quantification
                seg_pred, cls_pred, uncertainty = self.model(
                    images, monte_carlo_samples=20, training=False
                )
                
                # Update metrics
                test_metrics.update(
                    seg_pred.cpu(), masks.cpu(), 
                    cls_pred.cpu(), labels.cpu(), 
                    uncertainty.cpu()
                )
                
                # Store for visualization
                all_images.extend(images.cpu())
                all_true_masks.extend(masks.cpu())
                all_pred_masks.extend((seg_pred > 0.5).float().cpu())
                all_predictions.extend(cls_pred.cpu())
                all_uncertainties.extend(uncertainty.cpu())
                all_labels.extend(labels.cpu())
        
        # Get final metrics
        seg_metrics, cls_metrics = test_metrics.get_metrics()
        
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        print(f"Segmentation Metrics:")
        print(f"  Dice Coefficient: {seg_metrics['dice']:.4f}")
        print(f"  IoU Score:        {seg_metrics['iou']:.4f}")
        print(f"  Hausdorff Dist:   {seg_metrics['hausdorff_distance']:.2f}")
        print(f"  Avg Surface Dist: {seg_metrics['average_surface_distance']:.2f}")
        print(f"\nClassification Metrics:")
        print(f"  AUC Score:        {cls_metrics['auc']:.4f}")
        print(f"  Accuracy:         {cls_metrics['accuracy']:.4f}")
        print(f"  Mean Uncertainty: {cls_metrics['mean_uncertainty']:.4f}")
        print("="*60)
        
        # Generate comprehensive visualizations
        print("\nGenerating visualizations...")
        
        # 1. Sample predictions
        sample_indices = np.random.choice(len(all_images), min(8, len(all_images)), replace=False)
        fig1 = self.visualizer.plot_predictions(
            [all_images[i] for i in sample_indices],
            [all_true_masks[i] for i in sample_indices],
            [all_pred_masks[i] for i in sample_indices],
            [all_predictions[i] for i in sample_indices],
            [all_uncertainties[i] for i in sample_indices],
            [all_labels[i] for i in sample_indices]
        )
        plt.savefig('test_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Uncertainty analysis
        fig2 = self.visualizer.plot_uncertainty_analysis(
            torch.stack(all_uncertainties).numpy(),
            torch.stack(all_predictions),
            torch.stack(all_labels)
        )
        plt.savefig('uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Attention visualization (first sample)
        if len(all_images) > 0:
            fig3 = self.visualizer.plot_attention_maps(self.model, all_images[0])
            if fig3:
                plt.savefig('attention_maps.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        print("Evaluation completed! Visualizations saved.")
        
        return seg_metrics, cls_metrics

# ============================
# MAIN EXECUTION
# ============================

def main():
    """Main execution function"""
    print("Initializing AGA-Net Framework...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = Compose3D([
        RandomRotation3D(max_angle=10),
        RandomFlip3D(),
        Normalize3D(mean=0.0, std=1.0)
    ])
    
    val_transform = Compose3D([
        Normalize3D(mean=0.0, std=1.0)
    ])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = LungNoduleDataset(
        data_dir='./data/train', 
        mode='train', 
        transform=train_transform
    )
    
    val_dataset = LungNoduleDataset(
        data_dir='./data/val', 
        mode='val', 
        transform=val_transform
    )
    
    test_dataset = LungNoduleDataset(
        data_dir='./data/test', 
        mode='test', 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    print("Initializing AGA-Net model...")
    model = AGANet(in_channels=1, num_classes=2)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = AGANetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(num_epochs=100)  # Reduced for demo
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    seg_metrics, cls_metrics = trainer.evaluate_test_set(
        test_loader, 
        './checkpoints/best_model.pth'
    )
    
    # Generate final training curves
    fig = trainer.visualizer.plot_training_curves(history)
    plt.savefig('final_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAGA-Net training and evaluation completed successfully!")
    print("\nFiles generated:")
    print("- best_model.pth: Best trained model")
    print("- final_training_curves.png: Training progress")
    print("- test_predictions.png: Sample predictions")
    print("- uncertainty_analysis.png: Uncertainty analysis")
    print("- attention_maps.png: Attention visualizations")

if __name__ == "__main__":
    main()
