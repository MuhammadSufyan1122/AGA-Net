# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:07:55 2026

@author: WIN11
"""

import copy
import os
import random
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

# Reuse the manuscript implementation as the backbone.
# Make sure aganet_implementation.py is in the same directory.
from aganet_implementation import (
    AGANet,
    CombinedLoss,
    LungNoduleDataset,
    Compose3D,
    Normalize3D,
    RandomRotation3D,
    RandomFlip3D,
    MetricsCalculator,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# FEDERATED CONFIGURATION
# ============================================================

@dataclass
class FederatedConfig:
    num_clients: int = 5
    num_rounds: int = 20
    local_epochs: int = 2
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_workers: int = 0
    sample_fraction: float = 1.0  # fraction of clients sampled per round
    fedprox_mu: float = 0.0       # 0.0 = FedAvg, >0 = FedProx
    grad_clip: float = 1.0
    min_clients_per_round: int = 2
    save_dir: str = "./federated_checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# DATA PARTITIONING
# ============================================================

def split_indices_among_clients(
    dataset_len: int,
    num_clients: int,
    seed: int = 42,
    non_iid: bool = False,
    labels: Optional[List[int]] = None,
) -> Dict[int, List[int]]:
    """
    Split dataset indices across clients.

    IID mode: uniform random split.
    Non-IID mode: sort by labels and shard contiguous chunks.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(dataset_len)

    if non_iid and labels is not None:
        paired = list(zip(indices.tolist(), labels))
        paired.sort(key=lambda x: x[1])
        sorted_indices = [idx for idx, _ in paired]
        shards = np.array_split(sorted_indices, num_clients)
        return {cid: shard.tolist() for cid, shard in enumerate(shards)}

    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {cid: split.tolist() for cid, split in enumerate(splits)}


# ============================================================
# MODEL STATE HELPERS
# ============================================================

def get_model_state(model: nn.Module) -> OrderedDict:
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in model.state_dict().items())


def set_model_state(model: nn.Module, state_dict: OrderedDict, device: str) -> None:
    model.load_state_dict(state_dict, strict=True)
    model.to(device)


def fedavg_aggregate(
    client_states: List[OrderedDict],
    client_weights: List[int],
) -> OrderedDict:
    total_weight = float(sum(client_weights))
    aggregated = OrderedDict()

    for key in client_states[0].keys():
        weighted_sum = None
        for state, weight in zip(client_states, client_weights):
            tensor = state[key].float() * (weight / total_weight)
            weighted_sum = tensor if weighted_sum is None else weighted_sum + tensor
        aggregated[key] = weighted_sum

    return aggregated


# ============================================================
# CLIENT
# ============================================================

class FederatedClient:
    def __init__(
        self,
        client_id: int,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: FederatedConfig,
    ) -> None:
        self.client_id = client_id
        self.config = config
        self.device = config.device

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=(self.device == "cuda"),
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=(self.device == "cuda"),
        )

        self.model = AGANet(in_channels=1, num_classes=2).to(self.device)
        self.criterion = CombinedLoss()

    def _proximal_term(self, global_model: nn.Module) -> torch.Tensor:
        if self.config.fedprox_mu <= 0:
            return torch.tensor(0.0, device=self.device)
        prox = torch.tensor(0.0, device=self.device)
        for local_param, global_param in zip(self.model.parameters(), global_model.parameters()):
            prox += torch.norm(local_param - global_param.detach().to(self.device)) ** 2
        return 0.5 * self.config.fedprox_mu * prox

    def train_local(self, global_state: OrderedDict) -> Tuple[OrderedDict, Dict[str, float], int]:
        set_model_state(self.model, global_state, self.device)
        global_reference = copy.deepcopy(self.model).to(self.device)
        global_reference.eval()

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        history = defaultdict(list)
        self.model.train()

        for _ in range(self.config.local_epochs):
            total_loss = 0.0
            total_seg_loss = 0.0
            total_cls_loss = 0.0
            total_unc_loss = 0.0
            num_batches = 0

            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                seg_pred, cls_pred, uncertainty = self.model(images, training=True)
                loss, seg_loss, cls_loss, unc_loss = self.criterion(
                    seg_pred, masks, cls_pred, labels, uncertainty
                )

                loss = loss + self._proximal_term(global_reference)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip)
                optimizer.step()

                total_loss += loss.item()
                total_seg_loss += seg_loss.item()
                total_cls_loss += cls_loss.item()
                total_unc_loss += unc_loss.item()
                num_batches += 1

            history["loss"].append(total_loss / max(num_batches, 1))
            history["seg_loss"].append(total_seg_loss / max(num_batches, 1))
            history["cls_loss"].append(total_cls_loss / max(num_batches, 1))
            history["unc_loss"].append(total_unc_loss / max(num_batches, 1))

        metrics = self.evaluate_local()
        metrics.update({
            "train_loss": float(np.mean(history["loss"])) if history["loss"] else 0.0,
            "train_seg_loss": float(np.mean(history["seg_loss"])) if history["seg_loss"] else 0.0,
            "train_cls_loss": float(np.mean(history["cls_loss"])) if history["cls_loss"] else 0.0,
            "train_unc_loss": float(np.mean(history["unc_loss"])) if history["unc_loss"] else 0.0,
        })

        return get_model_state(self.model), metrics, len(self.train_loader.dataset)

    @torch.no_grad()
    def evaluate_local(self) -> Dict[str, float]:
        self.model.eval()
        metrics_calc = MetricsCalculator()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            seg_pred, cls_pred, uncertainty = self.model(
                images,
                monte_carlo_samples=5,
                training=False,
            )
            loss, _, _, _ = self.criterion(seg_pred, masks, cls_pred, labels, uncertainty)
            metrics_calc.update(seg_pred.cpu(), masks.cpu(), cls_pred.cpu(), labels.cpu(), uncertainty.cpu())
            total_loss += loss.item()
            num_batches += 1

        seg_metrics, cls_metrics = metrics_calc.get_metrics()
        return {
            "val_loss": total_loss / max(num_batches, 1),
            "val_dice": seg_metrics["dice"],
            "val_iou": seg_metrics["iou"],
            "val_auc": cls_metrics["auc"],
            "val_accuracy": cls_metrics["accuracy"],
            "val_mean_uncertainty": cls_metrics["mean_uncertainty"],
        }


# ============================================================
# SERVER
# ============================================================

class FederatedServer:
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.global_model = AGANet(in_channels=1, num_classes=2).to(config.device)
        self.history = defaultdict(list)
        os.makedirs(config.save_dir, exist_ok=True)

    def sample_clients(self, clients: List[FederatedClient], round_idx: int) -> List[FederatedClient]:
        num_total = len(clients)
        num_sampled = max(
            self.config.min_clients_per_round,
            int(round(self.config.sample_fraction * num_total)),
        )
        rng = random.Random(42 + round_idx)
        sampled_indices = rng.sample(range(num_total), min(num_sampled, num_total))
        return [clients[i] for i in sampled_indices]

    def aggregate_round(
        self,
        sampled_clients: List[FederatedClient],
    ) -> Dict[str, float]:
        global_state = get_model_state(self.global_model)
        client_states = []
        client_weights = []
        round_metrics = []

        for client in sampled_clients:
            local_state, local_metrics, sample_count = client.train_local(global_state)
            client_states.append(local_state)
            client_weights.append(sample_count)
            round_metrics.append(local_metrics)

        new_global_state = fedavg_aggregate(client_states, client_weights)
        set_model_state(self.global_model, new_global_state, self.config.device)

        aggregated_metrics = self._aggregate_metrics(round_metrics, client_weights)
        return aggregated_metrics

    @staticmethod
    def _aggregate_metrics(client_metrics: List[Dict[str, float]], client_weights: List[int]) -> Dict[str, float]:
        total = float(sum(client_weights))
        merged = defaultdict(float)
        for metrics, weight in zip(client_metrics, client_weights):
            for key, value in metrics.items():
                merged[key] += float(value) * (weight / total)
        return dict(merged)

    def save_checkpoint(self, round_idx: int, metrics: Dict[str, float]) -> None:
        ckpt = {
            "round": round_idx,
            "global_model_state_dict": self.global_model.state_dict(),
            "metrics": metrics,
            "history": dict(self.history),
            "config": self.config.__dict__,
        }
        path = os.path.join(self.config.save_dir, f"global_round_{round_idx:03d}.pth")
        torch.save(ckpt, path)

    def fit(self, clients: List[FederatedClient]) -> Dict[str, List[float]]:
        best_auc = -float("inf")

        for round_idx in range(1, self.config.num_rounds + 1):
            sampled_clients = self.sample_clients(clients, round_idx)
            metrics = self.aggregate_round(sampled_clients)

            for key, value in metrics.items():
                self.history[key].append(value)

            print(
                f"[Round {round_idx:03d}] "
                f"train_loss={metrics.get('train_loss', 0.0):.4f} | "
                f"val_loss={metrics.get('val_loss', 0.0):.4f} | "
                f"val_dice={metrics.get('val_dice', 0.0):.4f} | "
                f"val_auc={metrics.get('val_auc', 0.0):.4f}"
            )

            self.save_checkpoint(round_idx, metrics)
            if metrics.get("val_auc", -float("inf")) > best_auc:
                best_auc = metrics["val_auc"]
                best_path = os.path.join(self.config.save_dir, "best_global_model.pth")
                torch.save(
                    {
                        "round": round_idx,
                        "global_model_state_dict": self.global_model.state_dict(),
                        "best_val_auc": best_auc,
                        "history": dict(self.history),
                        "config": self.config.__dict__,
                    },
                    best_path,
                )

        return dict(self.history)


# ============================================================
# GLOBAL EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_global_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    model.eval()
    criterion = CombinedLoss()
    metrics_calc = MetricsCalculator()
    total_loss = 0.0
    num_batches = 0

    for batch in test_loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        seg_pred, cls_pred, uncertainty = model(images, monte_carlo_samples=10, training=False)
        loss, _, _, _ = criterion(seg_pred, masks, cls_pred, labels, uncertainty)

        metrics_calc.update(seg_pred.cpu(), masks.cpu(), cls_pred.cpu(), labels.cpu(), uncertainty.cpu())
        total_loss += loss.item()
        num_batches += 1

    seg_metrics, cls_metrics = metrics_calc.get_metrics()
    return {
        "test_loss": total_loss / max(num_batches, 1),
        "test_dice": seg_metrics["dice"],
        "test_iou": seg_metrics["iou"],
        "test_auc": cls_metrics["auc"],
        "test_accuracy": cls_metrics["accuracy"],
        "test_mean_uncertainty": cls_metrics["mean_uncertainty"],
    }


# ============================================================
# DATASET FACTORY
# ============================================================

def make_transforms():
    train_transform = Compose3D([
        RandomRotation3D(max_angle=10),
        RandomFlip3D(),
        Normalize3D(mean=0.0, std=1.0),
    ])
    eval_transform = Compose3D([
        Normalize3D(mean=0.0, std=1.0),
    ])
    return train_transform, eval_transform


def build_datasets(data_root: str):
    train_transform, eval_transform = make_transforms()

    train_dataset = LungNoduleDataset(
        data_dir=os.path.join(data_root, "train"),
        mode="train",
        transform=train_transform,
    )
    val_dataset = LungNoduleDataset(
        data_dir=os.path.join(data_root, "val"),
        mode="val",
        transform=eval_transform,
    )
    test_dataset = LungNoduleDataset(
        data_dir=os.path.join(data_root, "test"),
        mode="test",
        transform=eval_transform,
    )
    return train_dataset, val_dataset, test_dataset


def build_federated_clients(
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: FederatedConfig,
    non_iid: bool = False,
) -> List[FederatedClient]:
    labels = None
    if hasattr(train_dataset, "samples"):
        labels = [sample["label"] for sample in train_dataset.samples]

    train_map = split_indices_among_clients(
        len(train_dataset),
        config.num_clients,
        seed=42,
        non_iid=non_iid,
        labels=labels,
    )
    val_map = split_indices_among_clients(
        len(val_dataset),
        config.num_clients,
        seed=123,
        non_iid=False,
        labels=None,
    )

    clients = []
    for client_id in range(config.num_clients):
        client_train = Subset(train_dataset, train_map[client_id])
        client_val = Subset(val_dataset, val_map[client_id])
        clients.append(FederatedClient(client_id, client_train, client_val, config))
    return clients


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(42)

    config = FederatedConfig(
        num_clients=5,
        num_rounds=10,
        local_epochs=1,
        batch_size=2,
        lr=1e-4,
        sample_fraction=0.6,
        fedprox_mu=1e-3,  # set to 0.0 for pure FedAvg
        save_dir="./federated_checkpoints",
    )

    print("Initializing federated AGA-Net training...")
    print(f"Device: {config.device}")
    print(f"Clients: {config.num_clients}, Rounds: {config.num_rounds}")

    data_root = "./data"
    train_dataset, val_dataset, test_dataset = build_datasets(data_root)

    clients = build_federated_clients(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        non_iid=True,
    )

    server = FederatedServer(config)
    history = server.fit(clients)

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )
    test_metrics = evaluate_global_model(server.global_model, test_loader, config.device)

    print("\nFinal Global Test Metrics")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nFederated training complete.")
    print(f"Checkpoints saved to: {config.save_dir}")
    print(f"Tracked metrics: {list(history.keys())}")


if __name__ == "__main__":
    main()
