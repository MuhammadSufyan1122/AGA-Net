# -*- coding: utf-8 -*-
"""
Federated AGA-Net aligned to Table 3-3 detailed federated learning parameters
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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

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
# Table 3-3 alignment
# ============================================================

@dataclass
class FederatedConfig:
    # Core FL parameters
    num_clients: int = 10                  # K: 8-10 -> use 10
    num_rounds: int = 100                  # R: 100
    local_epochs: int = 5                  # E: 5
    sample_fraction: float = 0.5           # C: 0.5
    min_clients_per_round: int = 4         # K_min: 4

    # Learning rates
    global_lr: float = 1e-4                # eta_global: 1e-4
    local_lr: float = 5e-3                 # eta_local: 5e-3
    momentum: float = 0.9                  # mu_sgd: 0.9
    weight_decay: float = 1e-5             # lambda_wd: 1e-5

    # Privacy-related parameters
    privacy_epsilon: float = 8.0           # epsilon: 8.0
    privacy_delta: float = 1e-5            # delta: 1e-5
    dp_noise_multiplier: float = 0.19      # sigma_dp: 0.19
    grad_clip: float = 2.0                 # C_clip: 2.0

    # Federated regularization / comms
    fed_reg_weight: float = 0.1            # mu_fed: 0.1
    compression_rho: float = 0.1           # rho: 0.1

    # Reliability / convergence
    client_dropout_rate: float = 0.1       # p_dropout: 0.1
    convergence_tolerance: float = 1e-6    # tau: 1e-6

    # Runtime
    batch_size: int = 2
    num_workers: int = 0
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


def state_to_update(local_state: OrderedDict, global_state: OrderedDict) -> OrderedDict:
    update = OrderedDict()
    for k in global_state.keys():
        update[k] = local_state[k].float() - global_state[k].float()
    return update


def update_to_state(global_state: OrderedDict, update: OrderedDict) -> OrderedDict:
    new_state = OrderedDict()
    for k in global_state.keys():
        new_state[k] = global_state[k].float() + update[k].float()
    return new_state


def compress_update(update: OrderedDict, rho: float) -> OrderedDict:
    """
    Simple top-k magnitude sparsification.
    rho=0.1 means 10% compression ratio dropped? Here we keep (1-rho) of elements.
    """
    if rho <= 0.0:
        return update

    compressed = OrderedDict()
    keep_ratio = max(0.0, min(1.0, 1.0 - rho))

    for k, v in update.items():
        flat = v.view(-1)
        numel = flat.numel()
        keep_k = max(1, int(numel * keep_ratio))

        if keep_k >= numel:
            compressed[k] = v
            continue

        _, idx = torch.topk(flat.abs(), keep_k)
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        compressed[k] = (flat * mask).view_as(v)

    return compressed


def add_dp_noise(update: OrderedDict, noise_multiplier: float, clip_bound: float) -> OrderedDict:
    """
    Basic Gaussian noise addition to clipped updates.
    Note: this is not a full formal DP accountant.
    """
    if noise_multiplier <= 0:
        return update

    noised = OrderedDict()
    for k, v in update.items():
        noise = torch.randn_like(v) * (noise_multiplier * clip_bound)
        noised[k] = v + noise
    return noised


def fedavg_aggregate_updates(
    client_updates: List[OrderedDict],
    client_weights: List[int],
) -> OrderedDict:
    total_weight = float(sum(client_weights))
    aggregated = OrderedDict()

    for key in client_updates[0].keys():
        weighted_sum = None
        for upd, weight in zip(client_updates, client_weights):
            tensor = upd[key].float() * (weight / total_weight)
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
        if self.config.fed_reg_weight <= 0:
            return torch.tensor(0.0, device=self.device)

        prox = torch.tensor(0.0, device=self.device)
        for local_param, global_param in zip(self.model.parameters(), global_model.parameters()):
            prox += torch.norm(local_param - global_param.detach().to(self.device)) ** 2
        return 0.5 * self.config.fed_reg_weight * prox

    def train_local(self, global_state: OrderedDict) -> Tuple[OrderedDict, Dict[str, float], int]:
        set_model_state(self.model, global_state, self.device)
        global_reference = copy.deepcopy(self.model).to(self.device)
        global_reference.eval()

        # Table says local learning rate = 5e-3 and momentum = 0.9
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config.local_lr,
            momentum=self.config.momentum,
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

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.grad_clip
                )
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

        local_state = get_model_state(self.model)
        local_update = state_to_update(local_state, global_state)

        # Communication compression
        local_update = compress_update(local_update, self.config.compression_rho)

        # Basic DP-style noise injection on update
        local_update = add_dp_noise(
            local_update,
            noise_multiplier=self.config.dp_noise_multiplier,
            clip_bound=self.config.grad_clip,
        )

        return local_update, metrics, len(self.train_loader.dataset)

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
            metrics_calc.update(
                seg_pred.cpu(), masks.cpu(), cls_pred.cpu(), labels.cpu(), uncertainty.cpu()
            )
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
        # Simulate client dropout first
        rng = random.Random(42 + round_idx)
        active_clients = [
            c for c in clients
            if rng.random() > self.config.client_dropout_rate
        ]

        if len(active_clients) < self.config.min_clients_per_round:
            active_clients = clients[:]

        num_total = len(active_clients)
        num_sampled = max(
            self.config.min_clients_per_round,
            int(round(self.config.sample_fraction * len(clients))),
        )
        num_sampled = min(num_sampled, num_total)

        sampled_indices = rng.sample(range(num_total), num_sampled)
        return [active_clients[i] for i in sampled_indices]

    def aggregate_round(self, sampled_clients: List[FederatedClient]) -> Dict[str, float]:
        global_state = get_model_state(self.global_model)
        client_updates = []
        client_weights = []
        round_metrics = []

        for client in sampled_clients:
            local_update, local_metrics, sample_count = client.train_local(global_state)
            client_updates.append(local_update)
            client_weights.append(sample_count)
            round_metrics.append(local_metrics)

        aggregated_update = fedavg_aggregate_updates(client_updates, client_weights)

        # Apply server/global learning rate
        scaled_update = OrderedDict()
        for k, v in aggregated_update.items():
            scaled_update[k] = self.config.global_lr * v

        new_global_state = update_to_state(global_state, scaled_update)
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
        prev_val_loss = None

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

            # Convergence tolerance tau = 1e-6
            current_val_loss = metrics.get("val_loss", None)
            if prev_val_loss is not None and current_val_loss is not None:
                if abs(prev_val_loss - current_val_loss) < self.config.convergence_tolerance:
                    print(
                        f"Early stopping at round {round_idx} "
                        f"(convergence tolerance reached: {self.config.convergence_tolerance})"
                    )
                    break
            prev_val_loss = current_val_loss

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

    # Use table-aligned values directly
    config = FederatedConfig(
        num_clients=10,
        num_rounds=100,
        local_epochs=5,
        batch_size=2,
        global_lr=1e-4,
        local_lr=5e-3,
        momentum=0.9,
        weight_decay=1e-5,
        sample_fraction=0.5,
        min_clients_per_round=4,
        privacy_epsilon=8.0,
        privacy_delta=1e-5,
        dp_noise_multiplier=0.19,
        grad_clip=2.0,
        fed_reg_weight=0.1,
        compression_rho=0.1,
        client_dropout_rate=0.1,
        convergence_tolerance=1e-6,
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
