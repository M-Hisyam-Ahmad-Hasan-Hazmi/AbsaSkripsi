"""
train.py
========
Script fine-tuning model ABSAIndoBERTGated pada dataset ulasan Shopee Mykonos.

Perbaikan dari versi sebelumnya:
    - Path dataset TIDAK lagi hardcode (gunakan argumen CLI atau DEFAULT_CONFIG)
    - Class weights otomatis untuk menangani dataset imbalanced
    - Early stopping agar tidak overfit
    - Preprocessing otomatis jika CSV berlabel belum ada
    - Progress bar yang benar (enumerate, bukan iterrows index)
    - Evaluasi lengkap: Accuracy + F1 + Classification Report

Alur:
    1. Cek/buat dataset berlabel (via preprocessing.py)
    2. Split Train / Val / Test
    3. DataLoader + class weights
    4. Fine-tune IndoBERT + GatedAttention
    5. Evaluasi akhir → simpan checkpoint terbaik

Jalankan dengan:
    python train.py --csv data/dataset_ulasan_mykonos_final.csv

Proyek  : ABSA Shopee Mykonos
Penulis : M. Hisyam Ahmad Hasan Hazmi (20220040239)
Kampus  : Universitas Nusa Putra Sukabumi
"""

import os
import time
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Modul lokal
from model        import (
    ABSAIndoBERTGated, ASPECT_NAMES, SENTIMENT_NAMES,
    NUM_ASPECTS, NUM_SENTIMENTS,
)
from dataset      import ReviewDataset, get_tokenizer, ASPECT_LABEL_COLS
from preprocessing import build_labeled_dataset


# ─────────────────────────────────────────────────────────────
# Hyperparameter default
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Path — ubah sesuai lokasi file CSV hasil scraping
    "raw_csv"        : "D:\COBA\data\dataset_ulasan_mykonos_final.csv",
    "labeled_csv"    : "D:\COBA\data\dataset_final.csv",
    "checkpoint_dir" : "checkpoints",
    "best_model_path": "checkpoints/best_model.pt",

    # Hyperparameter training
    "epochs"         : 10,
    "batch_size"     : 16,
    "learning_rate"  : 2e-5,
    "weight_decay"   : 1e-2,
    "max_length"     : 128,

    # Split rasio
    "train_ratio"    : 0.70,
    "val_ratio"      : 0.15,
    # test_ratio      = 1 - train_ratio - val_ratio

    # Scheduler
    "warmup_ratio"   : 0.1,

    # Early stopping
    "patience"       : 3,     # Berhenti jika Val F1 tidak membaik N epoch

    # Reproducibility
    "seed"           : 42,
    "num_workers"    : 0,     # Set > 0 hanya di Linux
}


# ─────────────────────────────────────────────────────────────
# Utilitas
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = 42) -> None:
    """Seed untuk reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Deteksi device terbaik."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Menggunakan: {device}")
    return device


# ─────────────────────────────────────────────────────────────
# Persiapan Data
# ─────────────────────────────────────────────────────────────
def prepare_dataset(cfg: dict) -> pd.DataFrame:
    """
    Pastikan dataset berlabel tersedia.
    Jika belum ada, jalankan auto-labeling dari CSV mentah.
    """
    labeled_path = cfg["labeled_csv"]

    if Path(labeled_path).exists():
        print(f"[Data] Dataset berlabel ditemukan: {labeled_path}")
        df = pd.read_csv(labeled_path, encoding="utf-8")
    else:
        print(f"[Data] Dataset berlabel tidak ditemukan.")
        print(f"[Data] Menjalankan auto-labeling dari: {cfg['raw_csv']}")

        if not Path(cfg["raw_csv"]).exists():
            raise FileNotFoundError(
                f"File CSV tidak ditemukan: {cfg['raw_csv']}\n"
                f"Pastikan file CSV hasil scraping ada di path tersebut, "
                f"atau gunakan argumen --csv untuk menentukan path yang benar."
            )

        # Deteksi separator dan encoding
        df = build_labeled_dataset(
            input_csv  = cfg["raw_csv"],
            output_csv = labeled_path,
            sep        = "\t",
            encoding   = "cp1252",
        )

    # Validasi kolom
    for col in ASPECT_LABEL_COLS:
        assert col in df.columns, (
            f"Kolom '{col}' tidak ditemukan. "
            "Hapus file labeled_csv dan jalankan ulang untuk regenerasi."
        )

    print(f"[Data] Total sampel: {len(df)}")
    return df


def split_dataset(
    df         : pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio  : float = 0.15,
    seed       : int   = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split DataFrame → Train / Val / Test."""
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
    """
    Hitung class weights untuk CrossEntropyLoss agar dataset imbalanced
    ditangani dengan baik.

    Returns
    -------
    Tensor [num_sentiments] — bobot per kelas, dirata-rata dari semua aspek
    """
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

    print(f"[Training] Class weights: "
          f"Positif={weights_norm[0]:.3f} | "
          f"Negatif={weights_norm[1]:.3f} | "
          f"Netral={weights_norm[2]:.3f}")

    return torch.tensor(weights_norm, dtype=torch.float).to(device)


def create_dataloaders(
    train_df   : pd.DataFrame,
    val_df     : pd.DataFrame,
    test_df    : pd.DataFrame,
    tokenizer,
    batch_size : int = 16,
    max_length : int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Buat DataLoader untuk Train / Val / Test."""
    kwargs = dict(tokenizer=tokenizer, max_length=max_length)

    train_ds = ReviewDataset(train_df, **kwargs)
    val_ds   = ReviewDataset(val_df,   **kwargs)
    test_ds  = ReviewDataset(test_df,  **kwargs)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# Loss: multi-aspect cross-entropy
# ─────────────────────────────────────────────────────────────
def compute_multi_aspect_loss(
    logits_list: List[torch.Tensor],
    labels     : torch.Tensor,
    criterion  : nn.CrossEntropyLoss,
) -> torch.Tensor:
    """
    Rata-rata Cross-Entropy di semua aspek.

    logits_list : List[[B, C]]   (len = num_aspects)
    labels      : [B, num_aspects]
    """
    total = torch.tensor(0.0, device=labels.device, requires_grad=True)
    for i, logit in enumerate(logits_list):
        total = total + criterion(logit, labels[:, i])
    return total / len(logits_list)


# ─────────────────────────────────────────────────────────────
# Metrik evaluasi
# ─────────────────────────────────────────────────────────────
def compute_metrics(
    all_preds : List[List[int]],
    all_labels: List[List[int]],
) -> dict:
    """
    Accuracy + weighted F1 untuk setiap aspek + rata-rata.

    Returns
    -------
    dict dengan kunci per nama aspek dan 'avg_accuracy', 'avg_f1'
    """
    results  = {}
    acc_list, f1_list = [], []

    for i, name in enumerate(ASPECT_NAMES):
        preds  = all_preds[i]
        labels = all_labels[i]
        acc    = accuracy_score(labels, preds)
        f1     = f1_score(labels, preds, average="weighted", zero_division=0)
        results[name] = {"accuracy": round(acc, 4), "f1": round(f1, 4)}
        acc_list.append(acc)
        f1_list.append(f1)

    results["avg_accuracy"] = round(float(np.mean(acc_list)), 4)
    results["avg_f1"]       = round(float(np.mean(f1_list)),  4)
    return results


# ─────────────────────────────────────────────────────────────
# Satu epoch pelatihan
# ─────────────────────────────────────────────────────────────
def train_one_epoch(
    model    : ABSAIndoBERTGated,
    loader   : DataLoader,
    optimizer: AdamW,
    scheduler,
    criterion: nn.CrossEntropyLoss,
    device   : torch.device,
) -> float:
    """Training satu epoch; kembalikan rata-rata loss."""
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()

        logits_list = model(input_ids, attention_mask, token_type_ids)
        loss        = compute_multi_aspect_loss(logits_list, labels, criterion)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        if (step + 1) % max(1, n_batches // 5) == 0:
            print(f"    Step [{step+1:>3}/{n_batches}]  Loss: {loss.item():.4f}")

    return total_loss / n_batches


# ─────────────────────────────────────────────────────────────
# Evaluasi
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model    : ABSAIndoBERTGated,
    loader   : DataLoader,
    criterion: nn.CrossEntropyLoss,
    device   : torch.device,
    phase    : str = "Val",
    verbose  : bool = True,
) -> Tuple[float, dict]:
    """Evaluasi model; kembalikan (avg_loss, metrics_dict)."""
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
    metrics  = compute_metrics(all_preds, all_labels)

    if verbose:
        print(f"\n  [{phase}] Loss={avg_loss:.4f}  "
              f"Avg Acc={metrics['avg_accuracy']:.4f}  "
              f"Avg F1={metrics['avg_f1']:.4f}")
        for name in ASPECT_NAMES:
            m = metrics[name]
            print(f"    {name:<28} Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")

    return avg_loss, metrics


# ─────────────────────────────────────────────────────────────
# Fungsi utama
# ─────────────────────────────────────────────────────────────
def train(config: dict = None) -> None:
    """
    Orkestrasi pelatihan end-to-end dengan early stopping.

    Parameter
    ----------
    config : dict (opsional) — override DEFAULT_CONFIG
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    set_seed(cfg["seed"])
    device = get_device()

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ── 1. Dataset ──────────────────────────────────────────
    df_all = prepare_dataset(cfg)
    train_df, val_df, test_df = split_dataset(
        df_all,
        train_ratio = cfg["train_ratio"],
        val_ratio   = cfg["val_ratio"],
        seed        = cfg["seed"],
    )

    # ── 2. Tokenizer & DataLoader ───────────────────────────
    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        tokenizer   = tokenizer,
        batch_size  = cfg["batch_size"],
        max_length  = cfg["max_length"],
        num_workers = cfg["num_workers"],
    )

    # ── 3. Class Weights ────────────────────────────────────
    class_weights = compute_class_weights(train_df, ASPECT_LABEL_COLS, device)

    # ── 4. Model ─────────────────────────────────────────────
    model = ABSAIndoBERTGated().to(device)

    # ── 5. Optimizer & Scheduler ────────────────────────────
    optimizer = AdamW(
        model.parameters(),
        lr           = cfg["learning_rate"],
        weight_decay = cfg["weight_decay"],
        eps          = 1e-8,
    )
    total_steps  = len(train_loader) * cfg["epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )

    # ── 6. Loss Function ─────────────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── 7. Training Loop ─────────────────────────────────────
    best_val_f1   = -1.0
    no_improve    = 0
    history       = []

    print(f"\n{'='*60}")
    print(f" Mulai Pelatihan — {cfg['epochs']} Epoch  |  "
          f"LR={cfg['learning_rate']}  |  BS={cfg['batch_size']}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        print(f"── Epoch {epoch}/{cfg['epochs']} ──────────────────────────")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, phase="Val"
        )

        elapsed = time.time() - t0
        print(f"  Durasi: {elapsed:.1f}s  |  Train Loss: {train_loss:.4f}")

        history.append({
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 4),
            "val_loss"   : round(val_loss, 4),
            "val_avg_acc": val_metrics["avg_accuracy"],
            "val_avg_f1" : val_metrics["avg_f1"],
        })

        # Simpan checkpoint terbaik
        if val_metrics["avg_f1"] > best_val_f1:
            best_val_f1 = val_metrics["avg_f1"]
            no_improve  = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_f1"      : best_val_f1,
                "config"      : cfg,
            }, cfg["best_model_path"])
            print(f"  ✓ Checkpoint disimpan  (Val F1={best_val_f1:.4f})")
        else:
            no_improve += 1
            print(f"  ✗ Tidak ada peningkatan ({no_improve}/{cfg['patience']})")

        # Early stopping
        if no_improve >= cfg["patience"]:
            print(f"\n  [Early Stopping] Berhenti di epoch {epoch}.")
            break

    # ── 8. Evaluasi akhir (Test set) ─────────────────────────
    print(f"\n{'='*60}")
    print(" Evaluasi pada Set Test (best checkpoint)")
    print(f"{'='*60}")

    ckpt = torch.load(cfg["best_model_path"], map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, device, phase="Test"
    )

    print(f"\n  Hasil Akhir Test Set:")
    print(f"  Avg Accuracy : {test_metrics['avg_accuracy']:.4f}")
    print(f"  Avg F1       : {test_metrics['avg_f1']:.4f}")
    for name in ASPECT_NAMES:
        m = test_metrics[name]
        print(f"  {name:<28} Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")

    # ── 9. Simpan riwayat ─────────────────────────────────────
    hist_path = os.path.join(cfg["checkpoint_dir"], "training_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"\n[Info] Riwayat pelatihan : {hist_path}")
    print(f"[Info] Best model        : {cfg['best_model_path']}")
    print(f"[Info] Best Val F1       : {best_val_f1:.4f}")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning ABSA IndoBERT-Gated Shopee Mykonos"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CONFIG["raw_csv"],
        help="Path CSV mentah hasil scraping (default: data/dataset_ulasan_mykonos_final.csv)",
    )
    parser.add_argument(
        "--labeled",
        type=str,
        default=DEFAULT_CONFIG["labeled_csv"],
        help="Path dataset berlabel output preprocessing (default: data/dataset_labeled.csv)",
    )
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["learning_rate"])
    parser.add_argument("--patience",   type=int,   default=DEFAULT_CONFIG["patience"],
                        help="Early stopping patience (default: 3)")
    parser.add_argument("--checkpoint", type=str,   default=DEFAULT_CONFIG["best_model_path"])
    args = parser.parse_args()

    train(config={
        "raw_csv"        : args.csv,
        "labeled_csv"    : args.labeled,
        "epochs"         : args.epochs,
        "batch_size"     : args.batch_size,
        "learning_rate"  : args.lr,
        "patience"       : args.patience,
        "best_model_path": args.checkpoint,
    })
