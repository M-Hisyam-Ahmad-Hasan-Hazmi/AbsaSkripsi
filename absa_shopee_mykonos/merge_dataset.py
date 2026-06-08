"""
merge_dataset.py
================
Menggabungkan dataset asli (auto-labeled) dengan dataset sintetis,
lalu menghasilkan dataset final yang seimbang untuk training.

Output: data/dataset_final.csv

Proyek : ABSA Shopee Mykonos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from generate_synthetic_data import generate_synthetic_dataset

LABEL_COLS = [
    "label_kualitas_produk",
    "label_harga",
    "label_kualitas_pengiriman",
    "label_kepuasan_pelanggan",
]
ASPECT_NAMES = [
    "Kualitas Produk", "Harga",
    "Kualitas Pengiriman", "Kepuasan Pelanggan",
]
LBL = {0: "Positif", 1: "Negatif", 2: "Netral"}


def print_distribution(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{'='*55}")
        print(f" {title}")
        print(f"{'='*55}")
    print(f" Total baris: {len(df)}")
    for col, name in zip(LABEL_COLS, ASPECT_NAMES):
        vc  = df[col].value_counts().sort_index()
        row = " | ".join(
            f"{LBL[k]}={v} ({v/len(df)*100:.1f}%)"
            for k, v in vc.items()
        )
        print(f"  {name:<28}: {row}")


def merge_and_balance(
    original_path : str = "data/dataset_labeled.csv",
    synthetic_path: str = "data/dataset_synthetic.csv",
    output_path   : str = "data/dataset_final.csv",
    seed          : int = 42,
) -> pd.DataFrame:
    """
    Gabungkan data asli + sintetis, shuffle, simpan.
    """
    # ── Muat data asli ─────────────────────────────────────
    print(f"\n[Merge] Memuat data asli: {original_path}")
    df_orig = pd.read_csv(original_path, encoding="utf-8")
    if "source" not in df_orig.columns:
        df_orig["source"] = "original"

    print_distribution(df_orig, "Distribusi Data ASLI")

    # ── Muat / buat data sintetis ──────────────────────────
    if not Path(synthetic_path).exists():
        print(f"\n[Merge] Membuat dataset sintetis...")
        df_synth = generate_synthetic_dataset(synthetic_path)
    else:
        print(f"\n[Merge] Memuat data sintetis: {synthetic_path}")
        df_synth = pd.read_csv(synthetic_path, encoding="utf-8")

    print_distribution(df_synth, "Distribusi Data SINTETIS")

    # ── Pastikan kolom seragam ──────────────────────────────
    keep_cols = ["review_text"] + LABEL_COLS + ["source"]

    for df in [df_orig, df_synth]:
        if "source" not in df.columns:
            df["source"] = "unknown"

    df_orig  = df_orig[keep_cols]
    df_synth = df_synth[keep_cols]

    # ── Gabungkan ──────────────────────────────────────────
    df_combined = pd.concat([df_orig, df_synth], ignore_index=True)

    # Hapus duplikat teks
    before = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=["review_text"]).copy()
    after  = len(df_combined)
    if before != after:
        print(f"\n[Merge] Hapus {before - after} duplikat.")

    # Shuffle
    df_combined = df_combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    print_distribution(df_combined, "Distribusi GABUNGAN (Final)")

    # ── Validasi label ─────────────────────────────────────
    for col in LABEL_COLS:
        assert df_combined[col].isin([0, 1, 2]).all(), \
            f"Label tidak valid di kolom {col}"

    # ── Simpan ────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[Merge] Dataset final disimpan: {output_path}")
    print(f"[Merge] Total sampel           : {len(df_combined)}")

    # Ringkasan sumber data
    print(f"\n[Merge] Komposisi sumber:")
    for src, cnt in df_combined["source"].value_counts().items():
        print(f"  {src:<12}: {cnt} ({cnt/len(df_combined)*100:.1f}%)")

    return df_combined


if __name__ == "__main__":
    merge_and_balance()
