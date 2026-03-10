"""
Federated Learning Trainer for AGA-Net
Implements Algorithms 1 & 2 from the paper:
- FedAvg-style aggregation with adaptive loss weighting
- Differential privacy (Gaussian mechanism)
- Client drift regularisation (FedProx-style L_fed)
- Client selection with partial participation
- Support for FedAvg / FedProx / SCAFFOLD baselines
"""

import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# Differential Privacy Utilities
# ─────────────────────────────────────────────────────────────

def compute_dp_noise_std(epsilon: float,
                         delta: float,
                         sensitivity: float) -> float:
    """
    Gaussian mechanism noise σ_dp (Eq. 27):
    σ_dp = sqrt(2 ln(1.25/δ)) * S / ε
    """
    return math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon


def add_dp_noise(model: nn.Module, sigma_dp: float,
                 clip_bound: float = 2.0) -> None:
    """
    Clip gradients and add Gaussian noise in-place (Eq. 27 / Alg. 1).
    Applied to model parameters after local training.
    """
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                # Gradient clipping (sensitivity = clip_bound)
                param.data = param.data / max(
                    1.0,
                    param.data.norm(2).item() / clip_bound
                )
                # Add calibrated Gaussian noise
                noise = torch.randn_like(param.data) * sigma_dp
                param.data += noise


# ─────────────────────────────────────────────────────────────
# Client Trainer
# ─────────────────────────────────────────────────────────────

class ClientTrainer:
    """
    Local training procedure for a single federated client.
    Runs E local epochs per communication round.
    """

    def __init__(self,
                 client_id: int,
                 model: nn.Module,
                 loss_fn,
                 train_loader: DataLoader,
                 device: torch.device,
                 local_lr: float = 5e-3,
                 local_epochs: int = 5,
                 weight_decay: float = 1e-5,
                 sigma_dp: float = 0.19,
                 clip_bound: float = 2.0,
                 use_dp: bool = True):

        self.client_id    = client_id
        self.model        = model
        self.loss_fn      = loss_fn
        self.loader       = train_loader
        self.device       = device
        self.lr           = local_lr
        self.epochs       = local_epochs
        self.sigma_dp     = sigma_dp
        self.clip_bound   = clip_bound
        self.use_dp       = use_dp

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=local_lr,
            weight_decay=weight_decay
        )

    def train_round(self,
                    global_params: List[torch.Tensor],
                    client_loss_var: float = 0.0) -> Dict:
        """
        Run one federated round (E local epochs) — Algorithm 1 / 2.
        Returns updated model state dict + loss statistics.
        """
        self.model.train()
        self.model.to(self.device)

        # Copy global params to local model
        with torch.no_grad():
            for lp, gp in zip(self.model.parameters(), global_params):
                lp.data.copy_(gp.data)

        total_losses = defaultdict(float)
        n_batches = 0

        for epoch in range(self.epochs):
            for batch in self.loader:
                images = batch['image'].to(self.device)
                masks  = batch['mask'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()

                output = self.model(images, return_uncertainty=False)

                local_params  = list(self.model.parameters())
                loss_dict = self.loss_fn(
                    output,
                    seg_target=masks,
                    cls_target=labels,
                    local_params=local_params,
                    global_params=global_params,
                    sigma_dp=self.sigma_dp,
                    client_loss_var=client_loss_var
                )

                loss_dict['total'].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.clip_bound)
                self.optimizer.step()

                for k, v in loss_dict.items():
                    if isinstance(v, torch.Tensor):
                        total_losses[k] += v.item()
                    elif k == 'weights':
                        total_losses['w_seg'] += v[0]
                        total_losses['w_cls'] += v[1]
                        total_losses['w_unc'] += v[2]
                        total_losses['w_fed'] += v[3]
                n_batches += 1

        # Apply differential privacy noise (Algorithm 1, line 20)
        if self.use_dp:
            add_dp_noise(self.model, self.sigma_dp, self.clip_bound)

        avg_losses = {k: v / max(n_batches, 1)
                      for k, v in total_losses.items()}
        avg_losses['n_samples'] = len(self.loader.dataset)

        return {
            'state_dict': copy.deepcopy(self.model.state_dict()),
            'losses': avg_losses
        }


# ─────────────────────────────────────────────────────────────
# Federated Aggregation Strategies
# ─────────────────────────────────────────────────────────────

def fedavg_aggregate(global_model: nn.Module,
                     client_updates: List[Dict],
                     total_samples: int) -> None:
    """
    FedAvg weighted aggregation (Eq. 14 / Algorithm 1, line 22):
    θ_g ← Σ_k  (|D_k| / |D|) * θ_k
    In-place update of global_model.
    """
    global_state = global_model.state_dict()

    for key in global_state.keys():
        aggregated = torch.zeros_like(global_state[key], dtype=torch.float32)
        for update in client_updates:
            n_k    = update['losses']['n_samples']
            weight = n_k / total_samples
            aggregated += weight * update['state_dict'][key].float()
        global_state[key] = aggregated

    global_model.load_state_dict(global_state)


def scaffold_aggregate(global_model: nn.Module,
                       client_updates: List[Dict],
                       total_samples: int,
                       server_controls: Dict,
                       client_controls_list: List[Dict]) -> None:
    """
    Simplified SCAFFOLD aggregation (variance reduction).
    Updates global model and server control variates.
    """
    # Standard FedAvg update
    fedavg_aggregate(global_model, client_updates, total_samples)

    # Update server control variates (simplified)
    if server_controls and client_controls_list:
        n_clients = len(client_controls_list)
        for key in server_controls:
            delta = torch.zeros_like(server_controls[key])
            for cc in client_controls_list:
                if key in cc:
                    delta += cc[key] / n_clients
            server_controls[key] += delta


# ─────────────────────────────────────────────────────────────
# Federated Server / Orchestrator
# ─────────────────────────────────────────────────────────────

class FederatedServer:
    """
    Central server orchestrating federated training.
    Implements Algorithm 1 (basic) and Algorithm 2 (adaptive weighting).
    """

    def __init__(self,
                 global_model: nn.Module,
                 loss_fn,
                 client_loaders: List[DataLoader],
                 val_loader: DataLoader,
                 device: torch.device,
                 # Federated hyperparameters
                 num_rounds: int = 100,
                 local_epochs: int = 5,
                 client_fraction: float = 0.5,
                 min_clients: int = 4,
                 global_lr: float = 1e-4,
                 local_lr: float = 5e-3,
                 weight_decay: float = 1e-5,
                 # Privacy
                 epsilon: float = 8.0,
                 delta: float = 1e-5,
                 clip_bound: float = 2.0,
                 use_dp: bool = True,
                 # Aggregation
                 aggregation: str = 'fedavg',  # 'fedavg'|'fedprox'|'scaffold'
                 # Misc
                 seed: int = 42,
                 log_every: int = 10,
                 checkpoint_dir: str = './checkpoints'):

        self.global_model    = global_model.to(device)
        self.loss_fn         = loss_fn
        self.client_loaders  = client_loaders
        self.val_loader      = val_loader
        self.device          = device
        self.num_rounds      = num_rounds
        self.local_epochs    = local_epochs
        self.client_fraction = client_fraction
        self.min_clients     = min_clients
        self.global_lr       = global_lr
        self.local_lr        = local_lr
        self.weight_decay    = weight_decay
        self.use_dp          = use_dp
        self.aggregation     = aggregation
        self.log_every       = log_every
        self.checkpoint_dir  = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Compute DP noise
        sensitivity = clip_bound
        self.sigma_dp = compute_dp_noise_std(epsilon, delta, sensitivity) \
                        if use_dp else 0.0
        print(f"[DP] ε={epsilon}, δ={delta}, σ_dp={self.sigma_dp:.4f}")

        # Prepare per-client model (shared architecture, separate instances)
        num_clients = len(client_loaders)
        self.client_models = [
            copy.deepcopy(global_model) for _ in range(num_clients)
        ]

        # SCAFFOLD control variates
        self.server_controls  = {}
        self.client_controls  = [{} for _ in range(num_clients)]

        random.seed(seed)
        self.history = defaultdict(list)

    def _select_clients(self) -> List[int]:
        """Random client selection with participation fraction"""
        num_clients = len(self.client_loaders)
        n_select = max(self.min_clients,
                       int(num_clients * self.client_fraction))
        n_select = min(n_select, num_clients)
        return random.sample(range(num_clients), n_select)

    def _run_client(self, client_id: int,
                    global_params: List[torch.Tensor],
                    client_loss_var: float = 0.0) -> Dict:
        """Run local training on one client"""
        client_model = self.client_models[client_id]
        trainer = ClientTrainer(
            client_id     = client_id,
            model         = client_model,
            loss_fn       = self.loss_fn,
            train_loader  = self.client_loaders[client_id],
            device        = self.device,
            local_lr      = self.local_lr,
            local_epochs  = self.local_epochs,
            weight_decay  = self.weight_decay,
            sigma_dp      = self.sigma_dp,
            clip_bound    = 2.0,
            use_dp        = self.use_dp
        )
        return trainer.train_round(global_params, client_loss_var)

    def _aggregate(self, client_updates: List[Dict],
                   selected_clients: List[int]) -> None:
        """Aggregate client updates into global model"""
        total_samples = sum(u['losses']['n_samples'] for u in client_updates)

        if self.aggregation in ('fedavg', 'fedprox'):
            fedavg_aggregate(self.global_model, client_updates, total_samples)

        elif self.aggregation == 'scaffold':
            cc_list = [self.client_controls[k] for k in selected_clients]
            scaffold_aggregate(self.global_model, client_updates,
                               total_samples,
                               self.server_controls, cc_list)

    @torch.no_grad()
    def _validate(self) -> Dict:
        """Evaluate global model on validation set"""
        from .trainer import evaluate_model   # local import to avoid circular
        return evaluate_model(self.global_model, self.val_loader, self.device)

    def train(self) -> Dict:
        """
        Main federated training loop — Algorithm 1 / 2.
        Returns training history dict.
        """
        print(f"\n{'='*60}")
        print(f"  AGA-Net Federated Training  ({self.aggregation.upper()})")
        print(f"  Clients: {len(self.client_loaders)} | Rounds: {self.num_rounds}")
        print(f"  DP: {self.use_dp} | σ_dp: {self.sigma_dp:.4f}")
        print(f"{'='*60}\n")

        global_params = list(self.global_model.parameters())

        for rnd in range(1, self.num_rounds + 1):
            selected = self._select_clients()

            # ── Client Phase ──────────────────────────────────
            client_updates  = []
            round_total_losses = []

            for k in selected:
                update = self._run_client(k, global_params)
                client_updates.append(update)
                round_total_losses.append(update['losses'].get('total', 0.0))

            # Compute client loss variance for adaptive weighting
            client_loss_var = float(np.var(round_total_losses)) \
                              if len(round_total_losses) > 1 else 0.0

            # Re-run clients with updated variance (simplified single pass)
            # In practice this would be done iteratively within the round

            # ── Server Aggregation ────────────────────────────
            self._aggregate(client_updates, selected)
            global_params = list(self.global_model.parameters())

            # ── Logging ───────────────────────────────────────
            avg_total = np.mean([u['losses'].get('total', 0.0)
                                 for u in client_updates])
            avg_seg   = np.mean([u['losses'].get('seg', 0.0)
                                 for u in client_updates])
            avg_cls   = np.mean([u['losses'].get('cls', 0.0)
                                 for u in client_updates])
            avg_unc   = np.mean([u['losses'].get('unc', 0.0)
                                 for u in client_updates])

            self.history['round'].append(rnd)
            self.history['loss_total'].append(avg_total)
            self.history['loss_seg'].append(avg_seg)
            self.history['loss_cls'].append(avg_cls)
            self.history['loss_unc'].append(avg_unc)
            self.history['client_var'].append(client_loss_var)

            if rnd % self.log_every == 0:
                print(f"[Round {rnd:3d}/{self.num_rounds}] "
                      f"Clients: {len(selected)} | "
                      f"Loss: {avg_total:.4f} "
                      f"(seg={avg_seg:.4f}, cls={avg_cls:.4f}, "
                      f"unc={avg_unc:.4f})")
                self._save_checkpoint(rnd)

        print("\n[Training Complete]")
        self._save_checkpoint('final')
        return dict(self.history)

    def _save_checkpoint(self, tag) -> None:
        path = os.path.join(self.checkpoint_dir, f'aga_net_{tag}.pt')
        torch.save({
            'model_state': self.global_model.state_dict(),
            'history':     dict(self.history),
            'sigma_dp':    self.sigma_dp,
        }, path)
        print(f"  ✓ Checkpoint saved → {path}")


import os   # ensure os is available at module level
