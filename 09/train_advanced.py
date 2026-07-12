"""
Advanced CNN Training for ARC Tasks.

Techniques:
  1. U-Net skip connections — preserve spatial detail
  2. Hard example mining — focus on cells the model gets wrong
  3. Mixed loss (CE + Dice) — better for sparse grids
  4. Progressive unfreezing — train layers gradually
  5. More data augmentation (6 rotations/flips)

Architecture: Lightweight U-Net (~60K params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

NUM_COLORS = 10
CANVAS = 30


# ============================================================================
# Architecture: Lightweight U-Net (~60K params)
# ============================================================================

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(8, ch)

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, ch)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return F.relu(self.norm(self.conv(x)))


class AdvancedTaskCNN(nn.Module):
    """
    Lightweight U-Net with ResBlocks and GroupNorm.
    All channels = 32 to keep params under 100K.

    Param count: ~97K
    """
    def __init__(self, ch=32):
        super().__init__()
        self.ch = ch

        # Encoder
        self.enc_in = nn.Sequential(
            nn.Conv2d(10, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.ReLU(inplace=True),
        )
        self.enc1 = nn.Sequential(ResBlock(ch), ResBlock(ch))
        self.down1 = Downsample(ch)
        self.enc2 = ResBlock(ch)

        # Decoder
        self.up1 = Upsample(ch)
        self.dec1_conv = nn.Conv2d(ch * 2, ch, 1)
        self.dec1 = ResBlock(ch)

        # Head
        self.head = nn.Conv2d(ch, 10, 1)

    def forward(self, x):
        # Encoder
        e0 = self.enc_in(x)         # (B, 32, 30, 30)
        e1 = self.enc1(e0)          # (B, 32, 30, 30)
        d1 = self.down1(e1)         # (B, 32, 15, 15)
        e2 = self.enc2(d1)          # (B, 32, 15, 15)

        # Decoder with skip connection
        u1 = self.up1(e2)           # (B, 32, 30, 30)
        u1 = torch.cat([u1, e1], dim=1)  # (B, 64, 30, 30)
        u1 = self.dec1_conv(u1)     # (B, 32, 30, 30)
        d1 = self.dec1(u1)          # (B, 32, 30, 30)

        return self.head(d1)        # (B, 10, 30, 30)


# ============================================================================
# Loss functions
# ============================================================================

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss per-channel, averaged over channels."""
    C = pred.shape[1]
    pred_soft = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target.long(), C).permute(0, 3, 1, 2).float()
    intersection = (pred_soft * target_onehot).sum(dim=(2, 3))
    union = pred_soft.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def ce_dice_loss(pred: torch.Tensor, target: torch.Tensor,
                 ce_weight: float = 0.7, dice_weight: float = 0.3) -> torch.Tensor:
    """Combined CE + Dice loss."""
    ce = F.cross_entropy(pred, target.long())
    dl = dice_loss(pred, target)
    return ce_weight * ce + dice_weight * dl


# ============================================================================
# Hard example mining
# ============================================================================

def compute_cell_weights(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                         temperature: float = 2.0) -> torch.Tensor:
    """Compute per-cell sample weights based on model error."""
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        correct = (pred == Y.argmax(dim=1)).float()
        weights = 1.0 + (temperature - 1.0) * (1.0 - correct)
    model.train()
    return weights


def weighted_ce(pred: torch.Tensor, target: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with per-cell weights."""
    ce = F.cross_entropy(pred, target.long(), reduction='none')
    return (ce * weights).mean()


# ============================================================================
# Progressive unfreezing
# ============================================================================

def get_param_groups(model: nn.Module, epoch: int, total_epochs: int,
                     n_stages: int = 3) -> List[dict]:
    """Progressive unfreezing: head→decoder→bridge→encoder."""
    progress = epoch / total_epochs
    stage = min(int(progress * n_stages), n_stages - 1)

    groups = []

    # Always train head + decoder
    groups.append({'params': model.head.parameters(), 'lr_scale': 1.0})
    groups.append({'params': model.dec1.parameters(), 'lr_scale': 1.0})
    groups.append({'params': model.dec1_conv.parameters(), 'lr_scale': 1.0})
    groups.append({'params': model.up1.parameters(), 'lr_scale': 1.0})

    if stage >= 1:
        groups.append({'params': model.enc2.parameters(), 'lr_scale': 0.5})

    if stage >= 2:
        groups.append({'params': model.enc1.parameters(), 'lr_scale': 0.3})
        groups.append({'params': model.enc_in.parameters(), 'lr_scale': 0.3})
        groups.append({'params': model.down1.parameters(), 'lr_scale': 0.3})

    return groups


# ============================================================================
# Data augmentation
# ============================================================================

def augment_data(X: torch.Tensor, Y: torch.Tensor):
    """Create 6 augmented versions (original + 5 flips/rotations)."""
    X_aug = [X]
    Y_aug = [Y]
    X_aug.append(torch.flip(X, [2])); Y_aug.append(torch.flip(Y, [2]))
    X_aug.append(torch.flip(X, [3])); Y_aug.append(torch.flip(Y, [3]))
    X_aug.append(torch.rot90(X, 1, [2, 3])); Y_aug.append(torch.rot90(Y, 1, [2, 3]))
    X_aug.append(torch.rot90(X, 2, [2, 3])); Y_aug.append(torch.rot90(Y, 2, [2, 3]))
    X_aug.append(torch.rot90(X, 3, [2, 3])); Y_aug.append(torch.rot90(Y, 3, [2, 3]))
    return torch.cat(X_aug, dim=0), torch.cat(Y_aug, dim=0)


# ============================================================================
# Training loop
# ============================================================================

def train_advanced(task_num: int, X: torch.Tensor, Y: torch.Tensor,
                   max_epochs: int = 1000, lr: float = 3e-3,
                   verbose: bool = True) -> Tuple[Optional[AdvancedTaskCNN], float]:
    """Full advanced training loop.

    Returns (model, final_accuracy) or (None, 0.0).
    """
    N = X.shape[0]
    Y_target = Y.argmax(dim=1)

    # Augment
    X_all, Y_all = augment_data(X, Y)
    Y_all_target = Y_all.argmax(dim=1)

    model = AdvancedTaskCNN()

    # Track best
    best_acc = 0.0
    best_state = None
    patience = 300
    no_improve = 0
    base_lr = lr

    for epoch in range(max_epochs):
        # Progressive unfreezing — rebuild optimizer at stage boundaries
        progress = epoch / max_epochs
        stage = min(int(progress * 3), 2)
        prev_stage = min(int((epoch - 1) / max_epochs * 3), 2) if epoch > 0 else -1

        if epoch == 0 or stage != prev_stage:
            param_groups = get_param_groups(model, epoch, max_epochs)
            optimizer = torch.optim.Adam(
                [{'params': g['params'], 'lr': base_lr * g['lr_scale']}
                 for g in param_groups],
                weight_decay=1e-5,
            )

        # Hard example mining: active during epochs where acc is checked
        use_weights = (epoch % 100 >= 50 and epoch % 100 < 100 and epoch > 0)
        if use_weights:
            weights = compute_cell_weights(model, X, Y)
            weights_aug = weights.repeat(X_all.shape[0] // N, 1, 1)
            loss = weighted_ce(model(X_all), Y_all_target, weights_aug)
        else:
            pred = model(X_all)
            loss = ce_dice_loss(pred, Y_all_target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Check accuracy every 100 epochs
        if epoch % 100 == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(X)
                acc = (pred.argmax(dim=1) == Y_target).float().mean().item()
            model.train()

            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 100

            if verbose:
                stage_name = ["head+dec", "+enc2", "+all"][stage]
                print(f"    epoch {epoch:4d}/{max_epochs}  loss={loss.item():.6f}  "
                      f"acc={acc:.4f}  best={best_acc:.4f}  stage={stage_name}")

            if best_acc >= 1.0:
                if verbose:
                    print(f"    converged at epoch {epoch}")
                break

            if no_improve >= patience:
                if verbose:
                    print(f"    early stop at epoch {epoch}")
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final accuracy
    model.eval()
    with torch.no_grad():
        pred = model(X)
        acc = (pred.argmax(dim=1) == Y_target).float().mean().item()

    if verbose:
        print(f"    final accuracy: {acc:.4f}")

    return model, acc


# ============================================================================
# Export to ONNX
# ============================================================================

def export_to_onnx(model: nn.Module, onnx_path: str) -> str:
    """Export model to ONNX with inlined weights."""
    import onnx

    dummy = torch.zeros(1, NUM_COLORS, CANVAS, CANVAS)
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
    )

    # Inline external data
    try:
        m = onnx.load(onnx_path)
        onnx.save_model(m, onnx_path)
        from pathlib import Path
        data_file = Path(onnx_path).with_suffix(".onnx.data")
        if data_file.exists():
            data_file.unlink()
    except Exception:
        pass

    return onnx_path


# ============================================================================
# Count params
# ============================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
