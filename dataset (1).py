"""
Dataset utilities for AGA-Net federated training.
Supports LUNA16, LIDC-IDRI, NSCLC-Radiomics.
Implements Dirichlet-based non-IID partitioning for federated simulation.
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Dict, Optional
import warnings


# ─────────────────────────────────────────────────────────────
# Base Nodule Dataset
# ─────────────────────────────────────────────────────────────

class NoduleDataset(Dataset):
    """
    Generic 3-D lung nodule dataset.

    Expected structure:
        data_dir/
            images/   *.npy  — CT patches (H,W,D) float32, normalised
            masks/    *.npy  — binary segmentation masks
            labels.json      — {"patch_id": label}  0=benign, 1=malignant

    All patches should already be pre-processed to 64×64×64 voxels.
    If you have raw DICOM/NRRD files, run the preprocessing script first.
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',       # 'train'|'val'|'test'
                 patch_size: Tuple = (64, 64, 64),
                 augment: bool = True,
                 hu_range: Tuple = (-1000, 400),
                 label_file: str = 'labels.json'):
        self.data_dir   = data_dir
        self.split      = split
        self.patch_size = patch_size
        self.augment    = augment and (split == 'train')
        self.hu_min, self.hu_max = hu_range

        # Load file lists
        img_dir   = os.path.join(data_dir, 'images')
        mask_dir  = os.path.join(data_dir, 'masks')
        label_path = os.path.join(data_dir, label_file)

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")

        all_files = sorted([f[:-4] for f in os.listdir(img_dir)
                            if f.endswith('.npy')])

        # Load labels
        if os.path.exists(label_path):
            with open(label_path) as fh:
                self.labels = json.load(fh)
        else:
            warnings.warn(f"No label file found at {label_path}. "
                          "Using zeros as placeholder.")
            self.labels = {f: 0 for f in all_files}

        # Split
        n = len(all_files)
        random.seed(42)
        shuffled = all_files.copy()
        random.shuffle(shuffled)
        tr_end = int(0.70 * n)
        va_end = int(0.90 * n)

        split_map = {
            'train': shuffled[:tr_end],
            'val':   shuffled[tr_end:va_end],
            'test':  shuffled[va_end:]
        }
        self.file_ids = split_map.get(split, all_files)

        self.img_dir  = img_dir
        self.mask_dir = mask_dir

    def __len__(self): return len(self.file_ids)

    def _load_npy(self, fid: str) -> Tuple[np.ndarray, np.ndarray]:
        img  = np.load(os.path.join(self.img_dir,  fid + '.npy')).astype(np.float32)
        mask_path = os.path.join(self.mask_dir, fid + '.npy')
        if os.path.exists(mask_path):
            mask = np.load(mask_path).astype(np.float32)
        else:
            mask = np.zeros_like(img)
        return img, mask

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """HU clipping + z-score normalisation"""
        img = np.clip(img, self.hu_min, self.hu_max)
        mu, sd = img.mean(), img.std() + 1e-6
        return (img - mu) / sd

    def _resize(self, vol: np.ndarray) -> np.ndarray:
        """Resize to target patch_size using simple nearest-neighbour"""
        if vol.shape == self.patch_size:
            return vol
        from scipy.ndimage import zoom
        factors = [t / s for t, s in zip(self.patch_size, vol.shape)]
        return zoom(vol, factors, order=1)

    def _augment(self, img: np.ndarray,
                 mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Random rotation, flip, elastic — applied jointly"""
        # Random flip along each axis
        for ax in range(3):
            if random.random() > 0.5:
                img  = np.flip(img, axis=ax).copy()
                mask = np.flip(mask, axis=ax).copy()

        # Random 90° rotation in axial plane
        k = random.randint(0, 3)
        img  = np.rot90(img,  k, axes=(0, 1)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()

        # Gaussian noise
        if random.random() > 0.5:
            img = img + np.random.normal(0, 0.05, img.shape).astype(np.float32)

        return img, mask

    def __getitem__(self, idx: int) -> dict:
        fid  = self.file_ids[idx]
        img, mask = self._load_npy(fid)
        img  = self._preprocess(img)
        img  = self._resize(img)
        mask = self._resize(mask)
        mask = (mask > 0.5).astype(np.float32)

        if self.augment:
            img, mask = self._augment(img, mask)

        label = int(self.labels.get(fid, 0))

        return {
            'image': torch.from_numpy(img).unsqueeze(0),   # (1,H,W,D)
            'mask':  torch.from_numpy(mask).unsqueeze(0),  # (1,H,W,D)
            'label': torch.tensor(label, dtype=torch.long),
            'id': fid
        }


# ─────────────────────────────────────────────────────────────
# Synthetic Dataset (for testing when no real data available)
# ─────────────────────────────────────────────────────────────

class SyntheticNoduleDataset(Dataset):
    """
    Generates synthetic CT-like nodule patches for unit testing
    and debugging without real patient data.
    """
    def __init__(self, n_samples: int = 200,
                 patch_size: Tuple = (64, 64, 64),
                 num_classes: int = 2,
                 seed: int = 42):
        self.n       = n_samples
        self.ps      = patch_size
        self.nc      = num_classes
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.images  = []
        self.masks   = []
        self.labels  = []

        H, W, D = patch_size
        gz = torch.arange(H).float()
        gy = torch.arange(W).float()
        gx = torch.arange(D).float()
        zz, yy, xx = torch.meshgrid(gz, gy, gx, indexing='ij')

        for i in range(n_samples):
            label = i % num_classes
            # Background noise
            vol = torch.randn(H, W, D) * 0.1 - 0.5

            # Synthetic nodule (sphere)
            r = 5 + np.random.randint(0, 8)
            cz = H // 2 + np.random.randint(-5, 5)
            cy = W // 2 + np.random.randint(-5, 5)
            cx = D // 2 + np.random.randint(-5, 5)
            dist = ((zz - cz)**2 + (yy - cy)**2 + (xx - cx)**2).sqrt()
            nodule_mask = (dist < r).float()
            vol += nodule_mask * (0.5 + 0.3 * label)

            self.images.append(vol.unsqueeze(0))          # (1,H,W,D)
            self.masks.append(nodule_mask.unsqueeze(0))   # (1,H,W,D)
            self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self): return self.n

    def __getitem__(self, idx: int) -> dict:
        return {
            'image': self.images[idx],
            'mask':  self.masks[idx],
            'label': self.labels[idx],
            'id': str(idx)
        }


# ─────────────────────────────────────────────────────────────
# Federated Data Partitioning (Dirichlet non-IID)
# ─────────────────────────────────────────────────────────────

def dirichlet_partition(dataset: Dataset,
                        num_clients: int,
                        alpha: float = 1.0,
                        seed: int = 42) -> List[List[int]]:
    """
    Partition dataset indices across clients using Dirichlet distribution.
    Smaller alpha → more heterogeneous (non-IID) distribution.
    Implements the non-IID strategy described in the paper.

    Returns: list of index lists, one per client.
    """
    np.random.seed(seed)
    labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        labels.append(item['label'].item())
    labels = np.array(labels)

    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        np.random.shuffle(class_idx)

        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(
            np.ones(num_clients) * alpha)

        # Assign indices
        splits = (proportions * len(class_idx)).astype(int)
        splits[-1] = len(class_idx) - splits[:-1].sum()   # fix rounding

        ptr = 0
        for k in range(num_clients):
            client_indices[k].extend(
                class_idx[ptr: ptr + splits[k]].tolist())
            ptr += splits[k]

    # Shuffle each client's data
    for k in range(num_clients):
        random.shuffle(client_indices[k])

    return client_indices


def build_federated_loaders(dataset: Dataset,
                            num_clients: int,
                            alpha_dir: float = 1.0,
                            batch_size: int = 8,
                            num_workers: int = 2,
                            seed: int = 42) -> List[DataLoader]:
    """
    Build one DataLoader per federated client using Dirichlet partitioning.
    """
    partitions = dirichlet_partition(dataset, num_clients, alpha_dir, seed)
    loaders = []
    for k in range(num_clients):
        subset = Subset(dataset, partitions[k])
        loader = DataLoader(subset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False)
        loaders.append(loader)
    return loaders


def build_eval_loader(dataset: Dataset,
                      batch_size: int = 8,
                      num_workers: int = 2) -> DataLoader:
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=num_workers,
                      pin_memory=True)
