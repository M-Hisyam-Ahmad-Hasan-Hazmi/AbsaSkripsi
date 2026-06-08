"""
generate_synthetic_data.py
==========================
Generator dataset sintetis ulasan produk parfum/kosmetik Shopee Indonesia.

Strategi:
    - Template-based generation dengan variasi kalimat nyata
    - Mencakup 16 kombinasi aspek × sentimen
    - Fokus perbanyak kelas Negatif & Netral yang sangat kurang

Target distribusi akhir (setelah digabung dengan data asli):
    Positif: ~65%   Negatif: ~25%   Netral: ~10%

Output: data/dataset_synthetic.csv
"""

import random
import pandas as pd
from pathlib import Path
from itertools import product as iterproduct

random.seed(42)

# ============================================================
# Kamus Template Kalimat per Aspek × Sentimen
# ============================================================

TEMPLATES = {

    # ── KUALITAS PRODUK ─────────────────────────────────────
    "kualitas_produk": {
        "positif": [
            "Parfumnya wangi banget, tahan lama seharian",
            "Aromanya mewah dan enak banget, beneran suka",
            "Wanginya harum, botolnya keren dan premium banget",
            "Kualitas parfumnya bagus, wangi sepanjang hari",
            "Produknya berkualitas tinggi, wanginya lembut dan elegan",
            "Parfum ini beneran enak banget wanginya, tahan lama",
            "Kemasan mewah, kualitas produk tidak mengecewakan sama sekali",
            "Wanginya original, tidak murahan, sangat memuaskan",
            "Kualitas bagus banget, beda sama parfum abal-abal",
            "Aroma parfumnya premium, seperti parfum mahal padahal terjangkau",
            "Produknya oke banget, wangi segar dan tahan sampai malam",
            "Botol cantik dan elegan, parfumnya enak banget dipakai",
            "Kualitas produknya sangat baik, wangi awet dan tidak bikin mual",
            "Parfumnya bagus, packaging mewah, cocok buat hadiah",
            "Wangi parfumnya juara banget, tidak mengecewakan",
            "Aromanya benar-benar enak, kualitas tidak perlu diragukan",
            "Produknya mantap banget, wanginya fresh dan tahan lama",
            "Kualitas kelas atas, wanginya soft dan feminin",
            "Parfumnya top markotop, aroma berkelas",
            "Beneran worth it banget, kualitas parfum sangat bagus",
        ],
        "negatif": [
            "Kualitas parfumnya mengecewakan, wanginya cepat hilang",
            "Tidak sesuai deskripsi, baunya kurang enak",
            "Parfumnya palsu kayaknya, wanginya beda jauh dari asli",
            "Kualitas jelek banget, langsung hilang wanginya dalam 1 jam",
            "Wanginya tidak tahan lama sama sekali, kecewa berat",
            "Kualitas produk buruk, kemasan juga rusak",
            "Bau parfumnya aneh, tidak seperti gambar",
            "Kualitas sangat mengecewakan, tidak layak harga segitu",
            "Produk rusak saat diterima, kualitas kontrol buruk",
            "Wanginya terlalu menyengat dan bikin pusing",
            "Tidak seperti yang diiklankan, kualitas jauh di bawah ekspektasi",
            "Parfumnya sudah expired kayaknya, baunya tidak enak",
            "Botol bocor saat tiba, produk sudah tumpah separuh",
            "Kualitas sangat buruk, wangi hilang kurang dari 30 menit",
            "Aroma tidak sesuai foto, ternyata beda varian",
            "Produk tidak original, kayak tiruan murahan",
            "Parfum tidak enak dipakai, aroma kimiawi terlalu kuat",
            "Sangat kecewa dengan kualitas produknya, tidak akan beli lagi",
            "Warnanya berbeda dari foto, kemungkinan produk salah kirim",
            "Parfumnya kurang bagus, tidak sebanding dengan harganya",
        ],
        "netral": [
            "Kualitas parfumnya biasa aja sih, standar",
            "Wanginya lumayan, tidak terlalu istimewa tapi tidak jelek",
            "Kualitas produk cukup oke, sesuai harga",
            "Parfumnya lumayan, bisa dipakai sehari-hari",
            "Wangi cukup enak, tapi tidak sampai wow",
            "Produk standar, tidak ada yang spesial",
            "Kualitas biasa saja, sesuai dengan harga yang dibayar",
            "Wanginya netral, cocok untuk semua suasana",
            "Parfumnya oke lah, cukup untuk pemakaian harian",
            "Tidak terlalu wangi, tidak terlalu kurang, pas aja",
        ],
    },

    # ── HARGA ────────────────────────────────────────────────
    "harga": {
        "positif": [
            "Harganya terjangkau banget untuk kualitas segitu",
            "Worth it banget, harga murah tapi kualitas oke",
            "Harga sangat bersahabat, ramah di kantong",
            "Murah banget untuk parfum sekelas ini",
            "Harga sesuai kualitas, tidak mengecewakan",
            "Sangat terjangkau, bisa beli beberapa varian sekaligus",
            "Harganya murah, kualitas tidak murahan",
            "Budget friendly banget, rekomendasikan untuk yang hemat",
            "Harga promo sangat membantu, bisa beli lebih",
            "Harga 200 ribuan tapi kualitas premium, mantap",
            "Harganya masuk akal banget untuk parfum lokal berkualitas",
            "Ekonomis banget, tidak menguras kantong",
            "Harga normal tapi kualitas di atas rata-rata",
            "Sangat worth it, harga tidak tipu",
            "Harganya pas banget, tidak terlalu mahal tidak terlalu murah",
            "Dapet diskon lumayan, jadi tambah worth it",
            "Harga kompetitif dibanding merek lain yang serupa",
            "Murah tapi tidak murahan, senang dapat harga segini",
            "Harga promo Shopee bikin lebih hemat",
            "Cost-effective banget, cocok untuk pemakaian rutin",
        ],
        "negatif": [
            "Harganya mahal banget untuk kualitas yang biasa aja",
            "Terlalu kemahalan, tidak worth sama sekali",
            "Harga tidak sesuai dengan kualitas yang didapat",
            "Overpriced banget, ada yang lebih murah dengan kualitas sama",
            "Harganya kemahalan, harusnya bisa lebih murah",
            "Tidak worth it, harga segitu harusnya dapat yang lebih bagus",
            "Harga naik tapi kualitas tetap sama, kurang fair",
            "Mahal tapi mengecewakan, rugi beli ini",
            "Harga tidak sebanding dengan kualitas produk",
            "Kemahalan banget, mending beli merek lain",
            "Harga tinggi tapi parfumnya biasa aja, tidak worth",
            "Kalau harga segitu harusnya kualitasnya lebih baik",
            "Terlalu mahal untuk ukuran parfum lokal",
            "Tidak cocok di kantong, harganya nggak masuk akal",
            "Harga tidak bersahabat, susah beli rutin",
        ],
        "netral": [
            "Harganya standar, tidak murah tidak mahal",
            "Harga lumayan, sesuai pasaran",
            "Harga normal aja, tidak ada yang spesial",
            "Harganya segitu ya, masih bisa diterima",
            "Harga biasa saja untuk parfum lokal",
            "Harga wajar, standar pasaran",
            "Harganya oke, tidak terlalu mahal",
            "Harga cukup, tidak berlebihan",
        ],
    },

    # ── KUALITAS PENGIRIMAN ──────────────────────────────────
    "kualitas_pengiriman": {
        "positif": [
            "Pengiriman super cepat, kurang dari 24 jam sudah sampai",
            "Packing aman banget, botol tidak pecah sama sekali",
            "Pengiriman cepat dan produk sampai dalam kondisi sempurna",
            "Packaging rapih dan aman, tidak ada kerusakan",
            "Kurir cepat, packing double bubble wrap sangat aman",
            "Tiba lebih cepat dari estimasi, packing sangat baik",
            "Pengiriman kilat, sampai dalam kondisi baik",
            "Packing super aman, dikemas dengan bubble wrap tebal",
            "Pengiriman cepat, produk aman sampai tujuan",
            "Kemas dengan baik, pengiriman tidak mengecewakan",
            "Sampai dalam kondisi mulus, pengiriman tepat waktu",
            "Pengiriman cepat sekali, hari ini order besok sudah sampai",
            "Packaging bagus, isinya aman tidak bocor",
            "Pengiriman mantap, produk terlindungi dengan baik",
            "Cepat sampai, packaging rapih dan solid",
            "Tidak perlu khawatir soal pengiriman, sangat aman",
            "Pengiriman memuaskan, kurir responsif",
            "Packing berlapis, botol parfum aman tidak retak",
            "Pengiriman sangat cepat, packaging standar tinggi",
            "Kurir cepat dan packaging oke banget",
        ],
        "negatif": [
            "Pengiriman sangat lambat, sampai seminggu baru tiba",
            "Packing buruk, botol parfum retak saat diterima",
            "Pengiriman lama banget, sudah expired estimasinya",
            "Barang datang dalam kondisi rusak, packaging tidak aman",
            "Pengiriman molor jauh dari estimasi, mengecewakan",
            "Botol bocor karena packing tidak memadai",
            "Pengiriman lambat banget, tidak sesuai estimasi sama sekali",
            "Packaging buruk, isi parfum sudah tumpah sebagian",
            "Kurir tidak amanah, barang ditaruh sembarangan",
            "Pengiriman sangat lama, hampir 2 minggu baru sampai",
            "Packing jelek banget, parfum sampai dalam kondisi pecah",
            "Pengiriman mengecewakan, tracking tidak update berhari-hari",
            "Barang rusak karena pengiriman tidak hati-hati",
            "Lama banget sampainya, padahal jarak tidak terlalu jauh",
            "Pengiriman buruk, packing tidak standar",
            "Parfum bocor karena packing tidak rapat",
            "Kurir lambat dan tidak responsif, kecewa",
            "Pengiriman terlalu lama, sudah tidak sabar menunggu",
            "Barang lecet dan penyok karena packing tidak baik",
            "Pengiriman tidak memuaskan, perlu perbaikan",
        ],
        "netral": [
            "Pengiriman standar, sesuai estimasi",
            "Packing biasa, cukup aman",
            "Pengiriman normal, tidak terlalu cepat tidak terlalu lambat",
            "Lumayan cepat, packaging oke lah",
            "Pengiriman biasa saja, tidak ada yang istimewa",
            "Sampai tepat waktu, packing cukup memadai",
            "Pengiriman sesuai ekspektasi, biasa saja",
            "Packing standar, produk sampai dengan baik",
        ],
    },

    # ── KEPUASAN PELANGGAN ───────────────────────────────────
    "kepuasan_pelanggan": {
        "positif": [
            "Sangat puas dengan pembelian ini, akan beli lagi",
            "Happy banget, tidak menyesal beli produk ini",
            "Puas banget, langsung repeat order",
            "Beli lagi pasti, produk ini tidak mengecewakan",
            "Senang sekali, sesuai ekspektasi bahkan lebih",
            "Sangat puas, rekomendasikan ke semua teman",
            "Tidak nyesel beli, akan jadi langganan tetap",
            "Puas banget dengan semua aspek pembelian ini",
            "Bahagia banget dapet produk ini, sukses buat Mykonos",
            "Puas dan akan kembali berbelanja di sini",
            "Tidak kecewa, malah lebih dari yang diharapkan",
            "Sangat satisfied, produk terbaik pilihanku",
            "Happy dan puas, tidak sabar coba varian lain",
            "Beli lagi lagi dan lagi, selalu puas",
            "Kepuasan 100%, tidak ada yang perlu dikeluhkan",
            "Senang banget, beli ini keputusan terbaik",
            "Puas dan akan jadi repeat buyer setia",
            "Sangat memuaskan, produk recommended banget",
            "Happy dan tidak menyesal, top banget",
            "Puas banget, sudah beli 3 kali dan tetap puas",
        ],
        "negatif": [
            "Sangat kecewa, tidak akan beli lagi",
            "Menyesal banget beli produk ini",
            "Kapok beli di sini, total mengecewakan",
            "Tidak puas sama sekali, rugi waktu dan uang",
            "Kecewa berat, tidak sesuai ekspektasi",
            "Tidak akan rekomendasikan ke siapapun",
            "Menyesal banget, uang terbuang sia-sia",
            "Kecewa parah, tidak beli lagi di sini",
            "Pengalaman belanja terburuk, tidak puas",
            "Sangat tidak puas, komplain tapi tidak direspons",
            "Nyesel banget beli ini, buang-buang uang",
            "Kecewa total, tidak ada niat beli lagi",
            "Tidak merekomendasikan, banyak yang lebih baik",
            "Kecewa banget, tidak worth it sama sekali",
            "Pembeli kecewa, tidak akan kembali berbelanja",
        ],
        "netral": [
            "Biasa saja sih, tidak terlalu puas tidak terlalu kecewa",
            "Lumayan lah, masih bisa diterima",
            "Oke-oke aja, standar",
            "Cukup puas, tidak ada komplain berarti",
            "Biasa aja, mungkin beli lagi mungkin tidak",
            "Netral aja perasaannya, tidak istimewa",
            "Cukup memuaskan, tidak lebih tidak kurang",
            "Standar aja, bisa beli lagi kalau ada diskon",
        ],
    },
}

# ============================================================
# Kata penghubung dan variasi kalimat
# ============================================================

CONNECTORS = [
    ". ", ". Dan ", ". Tapi ", ". Untuk ", ". Soal ",
    ". Dari segi ", ", ", ". Sedangkan ", ". Sementara ",
    ". Oh ya, ", ". Btw ", ". FYI, ",
]

OPENERS = [
    "", "Update: ", "Review jujur: ", "Overall ", "Jujur aja, ",
    "Harus dibilang, ", "Terus terang, ", "Kalau kata saya, ",
    "Pengalaman saya: ", "Menurut saya, ", "Setelah dicoba: ",
    "Honest review: ", "Real review: ", "", "",
]

CLOSERS = [
    "", " Semoga membantu.", " GBU!", " Semoga berguna.",
    " Sekian review dari saya.", " Terima kasih.",
    " Semoga review ini bermanfaat!", "",
    " 5 bintang layak!", " Recommended!",
    " Tidak direkomendasikan.", " Kecewa.",
    " Mantap pokoknya!", "", "",
]


def random_review(kp_label: int, h_label: int, pg_label: int, ks_label: int) -> str:
    """
    Buat satu ulasan dari kombinasi label 4 aspek.

    Label: 0=Positif, 1=Negatif, 2=Netral
    """
    label_map = {0: "positif", 1: "negatif", 2: "netral"}

    kp_sent = random.choice(TEMPLATES["kualitas_produk"][label_map[kp_label]])
    h_sent  = random.choice(TEMPLATES["harga"][label_map[h_label]])
    pg_sent = random.choice(TEMPLATES["kualitas_pengiriman"][label_map[pg_label]])
    ks_sent = random.choice(TEMPLATES["kepuasan_pelanggan"][label_map[ks_label]])

    # Pilih urutan aspek secara acak
    parts = [kp_sent, h_sent, pg_sent, ks_sent]
    random.shuffle(parts)

    # Pilih 2-4 aspek untuk digabung (tidak selalu semua 4)
    n_parts = random.choices([2, 3, 4], weights=[0.3, 0.4, 0.3])[0]
    selected = parts[:n_parts]

    conn  = random.choice(CONNECTORS)
    text  = conn.join(selected)

    opener = random.choice(OPENERS)
    closer = random.choice(CLOSERS)
    text   = f"{opener}{text}{closer}".strip()

    # Variasi kapitalisasi
    if random.random() < 0.1:
        text = text.lower()
    elif random.random() < 0.05:
        text = text.upper()

    return text


# ============================================================
# Definisi jumlah sampel per kombinasi
# ============================================================

# Format: (label_kp, label_h, label_pg, label_ks): jumlah
COMBINATION_COUNTS = {
    # ── SEMUA POSITIF ──────────────────────────────────────
    (0, 0, 0, 0): 200,   # Semua positif (tambahan)

    # ── SEMUA NEGATIF ──────────────────────────────────────
    (1, 1, 1, 1): 250,   # Semua negatif

    # ── SEMUA NETRAL ───────────────────────────────────────
    (2, 2, 2, 2): 150,   # Semua netral

    # ── PRODUK BURUK, LAIN POSITIF ─────────────────────────
    (1, 0, 0, 1): 100,   # Produk&kepuasan negatif, harga&pengiriman positif
    (1, 0, 0, 0): 80,    # Produk negatif, lain positif
    (1, 1, 0, 1): 90,    # Produk&harga&kepuasan negatif
    (1, 0, 1, 1): 80,    # Produk&pengiriman&kepuasan negatif

    # ── HARGA MAHAL, LAIN BERVARIASI ───────────────────────
    (0, 1, 0, 0): 80,    # Harga negatif, lain positif
    (0, 1, 0, 2): 60,    # Harga negatif, kepuasan netral
    (0, 1, 1, 1): 80,    # Harga&pengiriman&kepuasan negatif
    (2, 1, 0, 2): 60,    # Produk netral, harga negatif

    # ── PENGIRIMAN BURUK, LAIN BERVARIASI ──────────────────
    (0, 0, 1, 0): 80,    # Pengiriman buruk, lain positif
    (0, 0, 1, 1): 100,   # Pengiriman&kepuasan buruk
    (0, 2, 1, 2): 60,    # Pengiriman negatif, lain netral
    (1, 0, 1, 1): 70,    # Produk&pengiriman&kepuasan buruk

    # ── KEPUASAN BURUK ─────────────────────────────────────
    (0, 0, 0, 1): 80,    # Kepuasan buruk, lain positif
    (2, 2, 2, 1): 60,    # Kepuasan buruk, lain netral
    (1, 1, 2, 1): 70,    # Produk&harga&kepuasan buruk

    # ── KOMBINASI NETRAL CAMPURAN ──────────────────────────
    (2, 0, 0, 0): 60,    # Produk netral, lain positif
    (0, 2, 0, 0): 60,    # Harga netral, lain positif
    (0, 0, 2, 0): 60,    # Pengiriman netral, lain positif
    (0, 0, 0, 2): 60,    # Kepuasan netral, lain positif
    (2, 2, 0, 0): 50,    # Produk&harga netral
    (0, 0, 2, 2): 50,    # Pengiriman&kepuasan netral
    (2, 1, 2, 1): 60,    # Produk&pengiriman netral, harga&kepuasan negatif
    (1, 2, 1, 2): 60,    # Produk&pengiriman negatif, harga&kepuasan netral

    # ── POSITIF SEBAGIAN ───────────────────────────────────
    (0, 0, 1, 2): 50,    # Pengiriman buruk, kepuasan netral
    (0, 1, 0, 2): 50,    # Harga mahal, kepuasan netral
    (1, 0, 0, 2): 50,    # Produk buruk, kepuasan netral
    (0, 0, 2, 1): 50,    # Pengiriman netral, kepuasan buruk
}


# ============================================================
# Generator utama
# ============================================================

def generate_synthetic_dataset(output_path: str = "data/dataset_synthetic.csv") -> pd.DataFrame:
    """
    Generate dataset sintetis dan simpan ke CSV.
    """
    print("=" * 55)
    print(" [Synthetic] Membangun dataset sintetis...")
    print("=" * 55)

    rows = []
    total_target = sum(COMBINATION_COUNTS.values())
    print(f" Target total sampel: {total_target}")
    print()

    for (kp, h, pg, ks), n in COMBINATION_COUNTS.items():
        for _ in range(n):
            text = random_review(kp, h, pg, ks)
            rows.append({
                "review_text"              : text,
                "label_kualitas_produk"    : kp,
                "label_harga"              : h,
                "label_kualitas_pengiriman": pg,
                "label_kepuasan_pelanggan" : ks,
                "source"                   : "synthetic",
            })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    lbl = {0: "Positif", 1: "Negatif", 2: "Netral"}
    cols_map = {
        "label_kualitas_produk"    : "Kualitas Produk",
        "label_harga"              : "Harga",
        "label_kualitas_pengiriman": "Kualitas Pengiriman",
        "label_kepuasan_pelanggan" : "Kepuasan Pelanggan",
    }
    print(" [Distribusi Sintetis]")
    for col, name in cols_map.items():
        vc  = df[col].value_counts().sort_index()
        row = " | ".join(
            f"{lbl[k]}={v} ({v/len(df)*100:.1f}%)"
            for k, v in vc.items()
        )
        print(f"  {name:<28}: {row}")

    print(f"\n Total sampel sintetis: {len(df)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f" Disimpan: {output_path}")

    return df


if __name__ == "__main__":
    generate_synthetic_dataset()
