"""
preprocessing.py
================
Modul preprocessing dan auto-labeling dataset ulasan Shopee Mykonos.

Fungsi:
    1. Bersihkan teks (cleaning + normalisasi slang Indonesia)
    2. Label otomatis 4 aspek via keyword-based rules + rating fallback
    3. Ekspor dataset berlabel ke CSV siap training

Label : 0 = Positif | 1 = Negatif | 2 = Netral
Aspek : Kualitas Produk | Harga | Kualitas Pengiriman | Kepuasan Pelanggan

Proyek : ABSA Shopee Mykonos
Penulis: M. Hisyam Ahmad Hasan Hazmi (20220040239)
Kampus : Universitas Nusa Putra Sukabumi

Cara pakai:
    python preprocessing.py --input data/dataset_ulasan_mykonos_final.csv \
                             --output data/dataset_labeled.csv
"""

import re
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

# ============================================================
# Kamus Normalisasi Bahasa Indonesia Informal
# ============================================================
SLANG_DICT: Dict[str, str] = {
    "gak": "tidak", "ga": "tidak", "ngga": "tidak", "nggak": "tidak",
    "gapapa": "tidak apa apa", "gabisa": "tidak bisa", "gaada": "tidak ada",
    "bgt": "banget", "bngt": "banget", "bngat": "banget",
    "yg": "yang", "dg": "dengan", "dgn": "dengan",
    "utk": "untuk", "tuk": "untuk",
    "krn": "karena", "karna": "karena",
    "tp": "tapi", "tpi": "tapi",
    "sy": "saya", "aq": "aku",
    "jg": "juga", "jga": "juga",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "blm": "belum", "blom": "belum",
    "sm": "sama",
    "bs": "bisa", "bsa": "bisa",
    "br": "baru",
    "lg": "lagi",
    "lbh": "lebih",
    "krng": "kurang",
    "hrs": "harus",
    "emg": "memang", "emang": "memang",
    "sbnernya": "sebenarnya", "sebenernya": "sebenarnya",
    "bnr": "benar", "bner": "benar",
    "pake": "pakai", "make": "memakai",
    "mantep": "mantap",
    "kyk": "kayak", "kyak": "kayak",
    "trs": "terus", "truss": "terus",
    "jd": "jadi",
    "mevvah": "mewah", "mewwaahh": "mewah", "mewahh": "mewah",
    "baguss": "bagus", "bagusss": "bagus",
    "cantikk": "cantik", "cakeepp": "cantik",
    "wangiii": "wangi", "wangii": "wangi", "wangiiii": "wangi",
    "ok": "oke", "oks": "oke",
    "nyesel": "menyesal",
    "kapok": "kapok",
    "puas": "puas",
    "recommend": "rekomendasikan",
    "recommended": "direkomendasikan",
    "worth": "sepadan",
    "best": "terbaik",
    "top": "terbaik",
    "zonk": "mengecewakan",
    "ngecewain": "mengecewakan",
    "kecewain": "mengecewakan",
}

# ============================================================
# Kamus Kata Kunci per Aspek dan Sentimen
# ============================================================
ASPECT_KEYWORDS: Dict[str, Dict[str, list]] = {
    "kualitas_produk": {
        "positif": [
            "wangi", "harum", "bagus", "enak", "tahan lama", "awet",
            "keren", "mewah", "cantik", "premium", "memuaskan",
            "mantap", "terbaik", "original", "fresh", "segar", "lembut",
            "sempurna", "berkualitas", "luar biasa", "asli",
            "botolnya bagus", "kemasan bagus", "kualitas bagus",
            "tidak mengecewakan", "wanginya", "aromanya",
            "packaging bagus", "packaging keren", "packaging mewah",
            "kualitas produk", "produk bagus", "parfumnya bagus",
            "parfum bagus", "tahan banget", "awet banget",
            "wangi banget", "harum banget", "kualitas baik",
            "tidak rusak", "tidak bocor", "kondisi baik",
            "direkomendasikan", "rekomendasikan",
        ],
        "negatif": [
            "tidak tahan", "tidak harum", "bau", "jelek", "murahan",
            "mengecewakan", "palsu", "tidak sesuai", "rusak", "cacat",
            "bocor", "kurang wangi", "tidak wangi", "kualitas jelek",
            "kemasan rusak", "botol rusak", "kualitas buruk",
            "parfumnya jelek", "produk jelek", "tidak bagus",
            "tidak original", "cepat habis", "tidak awet",
            "kualitas buruk", "tidak berkualitas", "produk cacat",
        ],
        "netral": [
            "biasa aja", "standar", "lumayan", "cukup oke",
            "biasa saja", "tidak terlalu", "agak kurang",
        ],
    },
    "harga": {
        "positif": [
            "murah", "terjangkau", "sepadan", "tidak mahal",
            "harga pas", "harga sesuai", "ekonomis", "ramah di kantong",
            "relatif murah", "harganya", "harga terjangkau", "ga mahal",
            "tidak terlalu mahal", "harga oke", "harga bagus",
            "harga murah", "harganya sesuai", "sesuai sama harganya",
            "sesuai kualitas", "200 ribuan", "200rb", "budget",
            "cukup murah", "terjangkau banget", "harga bersahabat",
        ],
        "negatif": [
            "mahal", "kemahalan", "tidak worth", "overprice",
            "terlalu mahal", "tidak sesuai harga", "harga ga sesuai",
            "harga tidak sesuai", "harga mahal", "kurang worth",
            "ga worth", "harusnya lebih murah", "terlalu tinggi",
            "kemahalan banget",
        ],
        "netral": [
            "harga lumayan", "harga standar", "harga biasa",
            "harga normal", "segitu", "harga sedang",
        ],
    },
    "kualitas_pengiriman": {
        "positif": [
            "cepat", "aman", "rapih", "rapi", "terlindungi",
            "tepat waktu", "selamat", "tidak rusak dalam pengiriman",
            "mantap pengiriman", "pengiriman bagus", "dikemas rapih",
            "aman sampai", "pengiriman kilat", "pengiriman mantap",
            "packing aman", "packing rapih", "pengiriman cepat",
            "cepat sampai", "packaging aman", "ga rusak",
            "sampai dengan aman", "packaging rapih", "packing bagus",
            "tidak ada kerusakan", "packaging oke", "pengemasan rapi",
            "kurir cepat", "ekspedisi cepat", "sampai cepet",
        ],
        "negatif": [
            "lambat", "lecet", "tidak aman", "hancur",
            "lama pengiriman", "packaging jelek", "tidak rapih",
            "pengiriman lelet", "kelamaan", "paket rusak",
            "packing buruk", "pengiriman payah", "lama banget",
            "pengiriman lama", "sangat lama", "lambat banget",
            "terlalu lama", "rusak saat tiba", "rusak sampai",
            "kemasan rusak", "isi tumpah", "pengiriman buruk",
            "kurir lambat", "lama sekali", "telat sampai",
        ],
        "netral": [
            "biasa pengiriman", "standar pengiriman",
            "sesuai estimasi", "pengiriman normal",
        ],
    },
    "kepuasan_pelanggan": {
        "positif": [
            "puas", "senang", "bahagia", "beli lagi",
            "tidak menyesal", "ga nyesel", "gak nyesel",
            "suka banget", "happy", "akan beli lagi",
            "tidak kecewa", "ga kecewa", "gak kecewa",
            "next order", "mau beli lagi", "puas banget",
            "sangat puas", "beli terus", "demen", "sukses",
            "5 bintang", "bintang 5", "top banget",
            "tidak nyesal", "ga menyesal", "repeat order",
            "cocok banget", "suka",
        ],
        "negatif": [
            "kecewa", "tidak puas", "tidak beli lagi",
            "menyesal", "kapok", "kecewa banget", "kapok beli",
            "tidak cocok", "tidak lagi", "nyesel beli",
            "ga balik lagi", "tidak akan beli",
            "sangat kecewa", "menyesal beli", "tidak mau beli",
        ],
        "netral": [
            "lumayan puas", "cukup ok", "standar lah",
            "biasa aja", "ok sih", "ya lumayan",
        ],
    },
}

# Label encoding — konsisten dengan model.py & dataset.py
LABEL_COLS = [
    "label_kualitas_produk",
    "label_harga",
    "label_kualitas_pengiriman",
    "label_kepuasan_pelanggan",
]

_ASPECT_MAP = {
    "label_kualitas_produk"     : "kualitas_produk",
    "label_harga"               : "harga",
    "label_kualitas_pengiriman" : "kualitas_pengiriman",
    "label_kepuasan_pelanggan"  : "kepuasan_pelanggan",
}


# ============================================================
# Utilitas Teks
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalisasi ringan: lowercase, hapus emoji, normalisasi karakter berulang,
    ganti slang. Digunakan untuk keyword matching saja (bukan untuk model).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Hapus karakter non-ASCII (emoji, simbol khusus)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Hapus URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Kurangi karakter berulang > 2 (wangiii → wangii)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Hapus karakter selain huruf, angka, spasi
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalisasi whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Ganti kata slang
    tokens = text.split()
    tokens = [SLANG_DICT.get(t, t) for t in tokens]
    text   = " ".join(t for t in tokens if t)

    return text


def clean_text_for_model(text: str) -> str:
    """
    Pembersihan untuk input ke tokenizer IndoBERT.
    Mempertahankan huruf Indonesia dan tanda baca penting,
    hanya membuang karakter kontrol dan URL.
    """
    if not isinstance(text, str):
        return ""

    # Hapus karakter kontrol
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)

    # Hapus URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Normalisasi whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Auto-Labeling
# ============================================================

def _score_aspect(text_norm: str, aspect_key: str) -> int:
    """
    Hitung skor sentimen untuk satu aspek dari teks yang sudah dinormalisasi.

    Returns
    -------
    int:
        0 = Positif, 1 = Negatif, 2 = Netral, -1 = tidak terdeteksi
    """
    kw = ASPECT_KEYWORDS[aspect_key]
    pos = sum(1 for w in kw["positif"] if w in text_norm)
    neg = sum(1 for w in kw["negatif"] if w in text_norm)
    net = sum(1 for w in kw["netral"]  if w in text_norm)

    if pos == 0 and neg == 0 and net == 0:
        return -1  # Aspek tidak terdeteksi

    if neg > pos and neg >= net:
        return 1   # Negatif
    if net > pos and net > neg:
        return 2   # Netral
    return 0       # Positif


def _rating_fallback(rating: int) -> int:
    """Fallback label berdasarkan rating bintang."""
    if rating >= 4:
        return 0   # Positif
    if rating == 3:
        return 2   # Netral
    return 1       # Negatif


def label_review(text: str, rating: int) -> dict:
    """
    Beri label 4 aspek untuk satu ulasan.

    Strategi (sesuai metodologi proposal):
        1. Cari kata kunci aspek-spesifik → gunakan hasilnya
        2. Jika aspek tidak disebutkan → fallback ke rating global
    """
    text_norm = normalize_text(text)
    fallback  = _rating_fallback(int(rating))

    return {
        col: (s if (s := _score_aspect(text_norm, key)) != -1 else fallback)
        for col, key in _ASPECT_MAP.items()
    }


# ============================================================
# Pipeline Utama
# ============================================================

def build_labeled_dataset(
    input_csv : str,
    output_csv: str = None,
    sep       : str = "\t",
    encoding  : str = "cp1252",
) -> pd.DataFrame:
    """
    Baca CSV mentah → bersihkan → label otomatis → simpan.

    Parameter
    ----------
    input_csv  : path CSV hasil scraping Shopee
    output_csv : path output; None = tidak disimpan ke disk
    sep        : separator CSV (default '\\t' untuk tab)
    encoding   : encoding file (default 'cp1252' untuk Windows)

    Returns
    -------
    pd.DataFrame berlabel siap training
    """
    print(f"\n{'='*55}")
    print(f" [Preprocessing] Memuat : {input_csv}")
    print(f"{'='*55}")

    df = pd.read_csv(input_csv, sep=sep, encoding=encoding,
                     on_bad_lines="skip")

    print(f" Total baris dimuat     : {len(df)}")
    print(f" Kolom                  : {df.columns.tolist()}")

    # ── Deteksi kolom teks & rating ──────────────────────────
    text_col   = next((c for c in df.columns
                       if any(k in c.lower() for k in ["teks", "ulasan", "review", "text"])),
                      df.columns[0])
    rating_col = next((c for c in df.columns
                       if any(k in c.lower() for k in ["rating", "bintang", "star"])),
                      None)

    print(f" Kolom teks             : {text_col}")
    print(f" Kolom rating           : {rating_col}")

    # ── Bersihkan teks ───────────────────────────────────────
    df["review_text"] = df[text_col].apply(clean_text_for_model)
    df = df[df["review_text"].str.strip().str.len() > 5].copy()
    df = df.drop_duplicates(subset=["review_text"]).copy()

    # ── Parsing rating ───────────────────────────────────────
    if rating_col and rating_col in df.columns:
        df["_rating"] = pd.to_numeric(df[rating_col], errors="coerce").fillna(5).astype(int)
    else:
        df["_rating"] = 5

    df = df.reset_index(drop=True)

    # ── Auto-labeling ────────────────────────────────────────
    print("\n [Auto-labeling] Memproses ulasan...")
    label_rows = [label_review(row["review_text"], row["_rating"])
                  for _, row in df.iterrows()]
    label_df   = pd.DataFrame(label_rows)
    df         = pd.concat([df[["review_text"]], label_df], axis=1)

    # ── Statistik distribusi label ───────────────────────────
    lbl_names  = {0: "Positif", 1: "Negatif", 2: "Netral"}
    asp_names  = ["Kualitas Produk", "Harga",
                  "Kualitas Pengiriman", "Kepuasan Pelanggan"]

    print("\n [Distribusi Label]")
    for col, asp in zip(LABEL_COLS, asp_names):
        vc  = df[col].value_counts().sort_index()
        row = " | ".join(f"{lbl_names.get(k, k)}={v}" for k, v in vc.items())
        print(f"  {asp:<28}: {row}")

    print(f"\n Total data siap training: {len(df)}")

    # ── Simpan ke CSV ─────────────────────────────────────────
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f" Dataset berlabel disimpan: {output_csv}")

    return df


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-labeling dataset ABSA Shopee Mykonos"
    )
    parser.add_argument(
        "--input",    type=str, required=True,
        help="Path CSV hasil scraping (Nama/Tanggal/Rating/Teks)"
    )
    parser.add_argument(
        "--output",   type=str, default="data/dataset_labeled.csv",
        help="Path output dataset berlabel (default: data/dataset_labeled.csv)"
    )
    parser.add_argument(
        "--sep",      type=str, default="\t",
        help="Separator CSV (default: tab)"
    )
    parser.add_argument(
        "--encoding", type=str, default="cp1252",
        help="Encoding file CSV (default: cp1252)"
    )
    args = parser.parse_args()

    build_labeled_dataset(
        input_csv  = args.input,
        output_csv = args.output,
        sep        = args.sep,
        encoding   = args.encoding,
    )
