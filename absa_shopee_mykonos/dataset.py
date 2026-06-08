"""
dataset.py
==========
Custom PyTorch Dataset untuk ulasan produk Shopee Mykonos.
Menangani tokenisasi teks berbahasa Indonesia menggunakan tokenizer IndoBERT.

Proyek  : ABSA Shopee Mykonos
Penulis : M. Hisyam Ahmad Hasan Hazmi (20220040239)
Kampus  : Universitas Nusa Putra Sukabumi

Format DataFrame yang diharapkan
---------------------------------
Kolom wajib (output dari preprocessing.py):
    review_text               : str  - teks ulasan yang sudah dibersihkan
    label_kualitas_produk     : int  - 0=Positif, 1=Negatif, 2=Netral
    label_harga               : int
    label_kualitas_pengiriman : int
    label_kepuasan_pelanggan  : int

Catatan: Gunakan preprocessing.py terlebih dahulu untuk menghasilkan
         file berlabel dari CSV scraping mentah.
"""

from typing import Optional, List

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer

from preprocessing import clean_text_for_model


# ─────────────────────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────────────────────
INDOBERT_MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LENGTH          = 128     # Sesuai proposal

ASPECT_LABEL_COLS = [
    "label_kualitas_produk",
    "label_harga",
    "label_kualitas_pengiriman",
    "label_kepuasan_pelanggan",
]

TEXT_COL = "review_text"


# ─────────────────────────────────────────────────────────────
# Kelas Dataset
# ─────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    """
    Dataset ulasan produk untuk task ABSA multi-aspek.

    Parameter
    ----------
    dataframe    : pd.DataFrame
        DataFrame yang mengandung kolom teks dan (opsional) kolom label aspek.
    tokenizer    : AutoTokenizer atau None
        Tokenizer IndoBERT. Jika None, dimuat otomatis.
    max_length   : int
        Panjang maksimum token setelah tokenisasi (default 128).
    label_cols   : list[str]
        Nama kolom label per aspek dalam DataFrame.
    text_col     : str
        Nama kolom yang berisi teks ulasan.
    is_inference : bool
        Jika True, DataFrame tidak perlu memiliki kolom label.
    """

    def __init__(
        self,
        dataframe    : pd.DataFrame,
        tokenizer               = None,
        max_length   : int      = MAX_LENGTH,
        label_cols   : list     = None,
        text_col     : str      = TEXT_COL,
        is_inference : bool     = False,
    ):
        self.df           = dataframe.reset_index(drop=True)
        self.max_length   = max_length
        self.label_cols   = label_cols or ASPECT_LABEL_COLS
        self.text_col     = text_col
        self.is_inference = is_inference

        # ── Tokenizer ─────────────────────────────────────────
        if tokenizer is None:
            print(f"[ReviewDataset] Memuat tokenizer: {INDOBERT_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(INDOBERT_MODEL_NAME)
        else:
            self.tokenizer = tokenizer

        # ── Validasi kolom ────────────────────────────────────
        assert text_col in self.df.columns, (
            f"Kolom teks '{text_col}' tidak ditemukan. "
            f"Kolom tersedia: {self.df.columns.tolist()}"
        )
        if not is_inference:
            missing = [c for c in self.label_cols if c not in self.df.columns]
            assert not missing, (
                f"Kolom label berikut tidak ditemukan: {missing}. "
                "Pastikan sudah menjalankan preprocessing.py terlebih dahulu."
            )

    # ──────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.df)

    # ──────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> dict:
        """
        Mengembalikan satu sampel berupa dict tensor siap masuk model.

        Returns
        -------
        dict:
            input_ids       : LongTensor [max_length]
            attention_mask  : LongTensor [max_length]
            token_type_ids  : LongTensor [max_length]
            labels          : LongTensor [num_aspects]  (hanya jika bukan inference)
        """
        row  = self.df.iloc[idx]
        text = clean_text_for_model(str(row[self.text_col]))

        # Pastikan teks tidak kosong
        if not text.strip():
            text = "ulasan kosong"

        # ── Tokenisasi ────────────────────────────────────────
        encoding = self.tokenizer(
            text,
            max_length            = self.max_length,
            padding               = "max_length",
            truncation            = True,
            return_tensors        = "pt",
            return_token_type_ids = True,
        )

        sample = {
            "input_ids"     : encoding["input_ids"].squeeze(0),       # [L]
            "attention_mask": encoding["attention_mask"].squeeze(0),  # [L]
            "token_type_ids": encoding["token_type_ids"].squeeze(0),  # [L]
        }

        # ── Label (tidak dibutuhkan saat inference) ───────────
        if not self.is_inference:
            labels = torch.tensor(
                [int(row[col]) for col in self.label_cols],
                dtype=torch.long,
            )
            sample["labels"] = labels   # [num_aspects]

        return sample


# ─────────────────────────────────────────────────────────────
# Utilitas
# ─────────────────────────────────────────────────────────────
def get_tokenizer(model_name: str = INDOBERT_MODEL_NAME) -> AutoTokenizer:
    """
    Muat dan kembalikan tokenizer IndoBERT.
    Disarankan dipakai bersama (shared instance) agar tidak dimuat ulang.
    """
    print(f"[get_tokenizer] Memuat: {model_name}")
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_single(
    text      : str,
    tokenizer : AutoTokenizer,
    max_length: int = MAX_LENGTH,
    device    : str = "cpu",
) -> dict:
    """
    Tokenisasi satu kalimat; kembalikan dict tensor siap inference.

    Parameter
    ----------
    text       : teks ulasan mentah
    tokenizer  : AutoTokenizer
    max_length : panjang maksimum token
    device     : 'cpu' atau 'cuda'

    Returns
    -------
    dict { input_ids [1,L], attention_mask [1,L], token_type_ids [1,L] }
    """
    text = clean_text_for_model(text)
    if not text.strip():
        text = "ulasan kosong"

    encoding = tokenizer(
        text,
        max_length            = max_length,
        padding               = "max_length",
        truncation            = True,
        return_tensors        = "pt",
        return_token_type_ids = True,
    )
    return {k: v.to(device) for k, v in encoding.items()}
