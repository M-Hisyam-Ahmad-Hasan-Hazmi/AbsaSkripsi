"""
train_optimized.py
==================
Script training optimized dengan:
  - Layer-wise Learning Rate Decay (LLRD)
  - Label Smoothing
  - Advanced class weighting
  - Gradient accumulation support
  - Early stopping + learning rate scheduling
  - Better logging & checkpointing
  - Validation pada berbagai metrics

Proyek  : ABSA Shopee Mykonos
Penulis : M. Hisyam Ahmad Hasan Hazmi (20220040239)
Kampus  : Universitas Nusa Putra Sukabumi

Jalankan:
  python train_optimized.py --csv data/dataset_ulasan_mykonos_final.csv --epochs 15 --batch_size 32
"""

import os
import time
import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Local imports
from model_optimized import (
    ABSAIndoBERTOptimized, 
    ASPECT_NAMES, SENTIMENT_NAMES,
    NUM_ASPECTS, NUM_SENTIMENTS,
    LabelSmoothingLoss,
    compute_multi_aspect_loss,
)
from dataset import ReviewDataset, get_tokenizer, ASPECT_LABEL_COLS
from preprocessing import build_labeled_dataset


# ─────────────────────────────────────────────────────────────
# Hyperparameter Optimized
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Path
    "raw_csv"        : "data/dataset_ulasan_mykonos_final.csv",
    "labeled_csv"    : "data/dataset_final.csv",
    "checkpoint_dir" : "checkpoints",
    "best_model_path": "checkpoints/best_model_optimized.pt",
    "logs_path"      : "logs/training_log.json",

    # Hyperparameter training (optimized)
    "epochs"         : 15,              # ↑ Lebih lama
    "batch_size"     : 32,              # ↑ Lebih besar
    "learning_rate"  : 3e-5,            # ↑ Sedikit lebih tinggi
    "weight_decay"   : 0.01,            # L2 regularization
    "max_length"     : 128,
    
    # Advanced optimization
    "warmup_ratio"   : 0.15,            # ↑ Lebih lama warm-up
    "gradient_accumulation_steps": 1,   # Gunakan >1 untuk small batch
    "label_smoothing": 0.1,
    "dropout_rate"   : 0.15,            # ↑ Lebih tinggi untuk regularization
    
    # Split rasio
    "train_ratio"    : 0.70,
    "val_ratio"      : 0.15,
    
    # Early stopping
    "patience"       : 4,               # ↑ Lebih lenient
    "min_delta"      : 0.001,           # Minimum improvement threshold
    
    # Layer-wise LR decay
    "use_llrd"       : True,
    "llrd_factor"    : 0.95,            # Decay factor per layer
    
    # Reproducibility
    "seed"           : 42,
    "num_workers"    : 0,
}


# ─────────────────────────────────────────────────────────────
# Utilitas
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Set seed untuk reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Detect best device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Device] CUDA Version: {torch.version.cuda}")
    print(f"[Device] Using: {device}")
    return device


class Logger:
    """Simple JSON logger untuk tracking metrics."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.logs = []
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def add(self, entry: dict) -> None:
        """Add log entry."""
        self.logs.append(entry)
    
    def save(self) -> None:
        """Save logs ke JSON."""
        with open(self.filepath, 'w') as f:
            json.dump(self.logs, f, indent=2)


# ─────────────────────────────────────────────────────────────
# Data Preparation
# ─────────────────────────────────────────────────────────────
def prepare_dataset(cfg: dict) -> pd.DataFrame:
    """Prepare labeled dataset."""
    labeled_path = cfg["labeled_csv"]
    
    if Path(labeled_path).exists():
        print(f"[Data] Loading labeled dataset: {labeled_path}")
        df = pd.read_csv(labeled_path, encoding="utf-8")
    else:
        print(f"[Data] Creating labeled dataset from: {cfg['raw_csv']}")
        if not Path(cfg["raw_csv"]).exists():
            raise FileNotFoundError(f"Dataset not found: {cfg['raw_csv']}")
        
        df = build_labeled_dataset(
            input_csv  = cfg["raw_csv"],
            output_csv = labeled_path,
            sep        = "\t",
            encoding   = "cp1252",
        )
    
    # Validate
    for col in ASPECT_LABEL_COLS:
        assert col in df.columns, f"Column '{col}' not found"
    
    print(f"[Data] Total samples: {len(df)}")
    return df


def split_dataset(
    df         : pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio  : float = 0.15,
    seed       : int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data → Train / Val / Test."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n       = len(df)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    
    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train : n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val:].copy()
    
    print(f"[Data] Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")
    return train_df, val_df, test_df


def compute_class_weights(
    df        : pd.DataFrame,
    label_cols: List[str],
    device    : torch.device,
) -> torch.Tensor:
    """Compute class weights untuk imbalanced dataset."""
    n_classes = NUM_SENTIMENTS
    weights_all = []
    
    for col in label_cols:
        counts = df[col].value_counts().sort_index()
        for c in range(n_classes):
            if c not in counts.index:
                counts[c] = 1
        counts = counts.sort_index()
        total  = counts.sum()
        w      = [total / (n_classes * counts[c]) for c in range(n_classes)]
        weights_all.append(w)
    
    weights_mean = np.mean(weights_all, axis=0)
    weights_norm = weights_mean / weights_mean.sum() * n_classes
    
    print(f"[Training] Class weights:")
    for i, (name, w) in enumerate(zip(SENTIMENT_NAMES, weights_norm)):
        print(f"  {name}: {w:.3f}")
    
    return torch.tensor(weights_norm, dtype=torch.float).to(device)


def create_dataloaders(
    train_df   : pd.DataFrame,
    val_df     : pd.DataFrame,
    test_df    : pd.DataFrame,
    tokenizer,
    batch_size : int = 32,
    max_length : int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders."""
    kwargs = dict(tokenizer=tokenizer, max_length=max_length)
    
    train_ds = ReviewDataset(train_df, **kwargs)
    val_ds   = ReviewDataset(val_df,   **kwargs)
    test_ds  = ReviewDataset(test_df,  **kwargs)
    
    pin = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# Optimizer dengan Layer-wise Learning Rate Decay
# ─────────────────────────────────────────────────────────────
def create_optimizer_with_llrd(
    model: ABSAIndoBERTOptimized,
    base_lr: float = 3e-5,
    weight_decay: float = 0.01,
    llrd_factor: float = 0.95,
) -> AdamW:
    """
    Create optimizer dengan Layer-wise Learning Rate Decay.
    
    Earlier layers (representation) mendapat LR lebih rendah.
    Later layers (task-specific) mendapat LR lebih tinggi.
    """
    
    layer_groups = model.get_layer_groups()
    optimizer_grouped_parameters = []
    
    # BERT layers (earlier layers): lower LR
    for group_idx, layer_group in enumerate(layer_groups[:-3]):  # BERT layers
        lr = base_lr * (llrd_factor ** (len(layer_groups) - 1 - group_idx))
        optimizer_grouped_parameters.append({
            'params': layer_group,
            'lr': lr,
            'weight_decay': weight_decay,
        })
    
    # Custom layers (later): higher LR
    for layer_group in layer_groups[-3:]:  # Last 3 groups (custom layers)
        optimizer_grouped_parameters.append({
            'params': layer_group,
            'lr': base_lr,
            'weight_decay': weight_decay,
        })
    
    print("[Optimizer] Layer-wise Learning Rate Decay enabled:")
    for i, group in enumerate(optimizer_grouped_parameters):
        print(f"  Group {i+1}: LR={group['lr']:.2e}")
    
    return AdamW(optimizer_grouped_parameters, eps=1e-8)


# ─────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────
def train_one_epoch(
    model    : ABSAIndoBERTOptimized,
    loader   : DataLoader,
    optimizer: AdamW,
    scheduler,
    criterion: nn.Module,
    device   : torch.device,
    accumulation_steps: int = 1,
) -> float:
    """Train satu epoch."""
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)
        
        logits_list = model(input_ids, attention_mask, token_type_ids)
        loss        = compute_multi_aspect_loss(logits_list, labels, criterion)
        
        # Gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        
        if (step + 1) % max(1, n_batches // 5) == 0:
            print(f"    Step [{step+1:>3}/{n_batches}]  Loss: {loss.item():.4f}")
    
    return total_loss / n_batches


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model    : ABSAIndoBERTOptimized,
    loader   : DataLoader,
    criterion: nn.Module,
    device   : torch.device,
    phase    : str = "Val",
) -> Tuple[float, dict]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds  = [[] for _ in range(NUM_ASPECTS)]
    all_labels = [[] for _ in range(NUM_ASPECTS)]
    
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)
        
        logits_list = model(input_ids, attention_mask, token_type_ids)
        loss        = compute_multi_aspect_loss(logits_list, labels, criterion)
        total_loss += loss.item()
        
        for i, logit in enumerate(logits_list):
            all_preds[i].extend(torch.argmax(logit, dim=-1).cpu().tolist())
            all_labels[i].extend(labels[:, i].cpu().tolist())
    
    avg_loss = total_loss / len(loader)
    
    # Compute metrics
    metrics = {}
    acc_list, f1_list = [], []
    
    for i, name in enumerate(ASPECT_NAMES):
        acc = accuracy_score(all_labels[i], all_preds[i])
        f1  = f1_score(all_labels[i], all_preds[i], average='weighted', zero_division=0)
        metrics[name] = {'accuracy': round(acc, 4), 'f1': round(f1, 4)}
        acc_list.append(acc)
        f1_list.append(f1)
    
    metrics['avg_accuracy'] = round(float(np.mean(acc_list)), 4)
    metrics['avg_f1']       = round(float(np.mean(f1_list)), 4)
    
    # Print
    print(f"\n  [{phase}] Loss={avg_loss:.4f}  "
          f"Avg Acc={metrics['avg_accuracy']:.4f}  "
          f"Avg F1={metrics['avg_f1']:.4f}")
    for name in ASPECT_NAMES:
        m = metrics[name]
        print(f"    {name:<28} Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
    
    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────────────────────
def train(config: dict = None) -> None:
    """Orchestrate training end-to-end."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])
    device = get_device()
    
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(cfg["logs_path"]), exist_ok=True)
    
    logger = Logger(cfg["logs_path"])
    
    # ── 1. Dataset ──
    print("\n" + "="*70)
    print(" PREPARING DATA")
    print("="*70)
    df_all = prepare_dataset(cfg)
    train_df, val_df, test_df = split_dataset(
        df_all,
        train_ratio = cfg["train_ratio"],
        val_ratio   = cfg["val_ratio"],
        seed        = cfg["seed"],
    )
    
    # ── 2. Tokenizer & DataLoader ──
    print("\n" + "="*70)
    print(" CREATING DATALOADERS")
    print("="*70)
    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        tokenizer   = tokenizer,
        batch_size  = cfg["batch_size"],
        max_length  = cfg["max_length"],
        num_workers = cfg["num_workers"],
    )
    
    # ── 3. Class Weights ──
    class_weights = compute_class_weights(train_df, ASPECT_LABEL_COLS, device)
    
    # ── 4. Model ──
    print("\n" + "="*70)
    print(" INITIALIZING MODEL")
    print("="*70)
    model = ABSAIndoBERTOptimized().to(device)
    print(f"[Model] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Model] Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ── 5. Optimizer dengan LLRD ──
    print("\n" + "="*70)
    print(" SETTING UP OPTIMIZER")
    print("="*70)
    if cfg["use_llrd"]:
        optimizer = create_optimizer_with_llrd(
            model,
            base_lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            llrd_factor=cfg["llrd_factor"],
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr           = cfg["learning_rate"],
            weight_decay = cfg["weight_decay"],
            eps          = 1e-8,
        )
    
    # ── 6. Scheduler ──
    total_steps  = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    
    print(f"[Scheduler] Total steps: {total_steps}")
    print(f"[Scheduler] Warmup steps: {warmup_steps}")
    
    # ── 7. Loss Function ──
    if cfg["label_smoothing"] > 0:
        criterion = LabelSmoothingLoss(
            num_classes=NUM_SENTIMENTS,
            smoothing=cfg["label_smoothing"],
        )
        print(f"[Loss] Label Smoothing enabled: {cfg['label_smoothing']}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # ── 8. Training Loop ──
    print("\n" + "="*70)
    print(f" TRAINING — {cfg['epochs']} Epochs")
    print("="*70 + "\n")
    
    best_val_f1   = -1.0
    no_improve    = 0
    history       = []
    
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        print(f"── Epoch {epoch}/{cfg['epochs']} ──────────────────────────")
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device,
            accumulation_steps=cfg["gradient_accumulation_steps"],
        )
        
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, phase="Val"
        )
        
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s  |  Train Loss: {train_loss:.4f}")
        
        # Log
        log_entry = {
            'epoch': epoch,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            'val_avg_acc': val_metrics['avg_accuracy'],
            'val_avg_f1': val_metrics['avg_f1'],
        }
        history.append(log_entry)
        logger.add(log_entry)
        
        # Save best checkpoint
        improvement = val_metrics['avg_f1'] - best_val_f1
        if improvement > cfg["min_delta"]:
            best_val_f1 = val_metrics['avg_f1']
            no_improve  = 0
            torch.save({
                'epoch'       : epoch,
                'model_state' : model.state_dict(),
                'optimizer'   : optimizer.state_dict(),
                'val_f1'      : best_val_f1,
                'config'      : cfg,
            }, cfg["best_model_path"])
            print(f"  ✓ Checkpoint saved (Val F1={best_val_f1:.4f})")
        else:
            no_improve += 1
            print(f"  ✗ No improvement ({no_improve}/{cfg['patience']})")
        
        # Early stopping
        if no_improve >= cfg["patience"]:
            print(f"\n[Early Stopping] Stopped at epoch {epoch}")
            break
        
        print()
    
    # ── 9. Final Evaluation ──
    print("\n" + "="*70)
    print(" FINAL EVALUATION (TEST SET)")
    print("="*70)
    
    ckpt = torch.load(cfg["best_model_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    
    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, device, phase="Test"
    )
    
    print(f"\n[Results] Test Set Performance:")
    print(f"  Avg Accuracy : {test_metrics['avg_accuracy']:.4f}")
    print(f"  Avg F1       : {test_metrics['avg_f1']:.4f}")
    for name in ASPECT_NAMES:
        m = test_metrics[name]
        print(f"  {name:<28} Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
    
    # ── 10. Save history ──
    hist_path = os.path.join(cfg["checkpoint_dir"], "training_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)
    logger.save()
    
    print(f"\n[Saved]")
    print(f"  Training history: {hist_path}")
    print(f"  Best model: {cfg['best_model_path']}")
    print(f"  Logs: {cfg['logs_path']}")
    print(f"  Best Val F1: {best_val_f1:.4f}")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized training for ABSA IndoBERT")
    parser.add_argument("--csv", type=str, default=DEFAULT_CONFIG["raw_csv"],
                       help="Path to raw CSV")
    parser.add_argument("--labeled", type=str, default=DEFAULT_CONFIG["labeled_csv"],
                       help="Path to labeled CSV")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CONFIG["best_model_path"])
    parser.add_argument("--no-llrd", action="store_true", help="Disable LLRD")
    args = parser.parse_args()
    
    train(config={
        "raw_csv"        : args.csv,
        "labeled_csv"    : args.labeled,
        "epochs"         : args.epochs,
        "batch_size"     : args.batch_size,
        "learning_rate"  : args.lr,
        "patience"       : args.patience,
        "best_model_path": args.checkpoint,
        "use_llrd"       : not args.no_llrd,
    })
