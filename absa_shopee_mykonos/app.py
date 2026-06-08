"""
app.py
======
Antarmuka Streamlit untuk Sistem ABSA Shopee Mykonos.

Menu:
    1. Analisis Teks Tunggal  — masukkan teks ulasan secara langsung.
    2. Upload CSV             — analisis batch dari file CSV.

Proyek  : ABSA Shopee Mykonos
Penulis : M. Hisyam Ahmad Hasan Hazmi (20220040239)
Kampus  : Universitas Nusa Putra Sukabumi

Jalankan dengan:
    streamlit run app.py
"""

import os
from typing import Optional

import torch
import pandas as pd
import streamlit as st

from model        import ABSAIndoBERTGated, ASPECT_NAMES, SENTIMENT_NAMES
from dataset      import get_tokenizer, tokenize_single
from preprocessing import clean_text_for_model


# ─────────────────────────────────────────────────────────────
# Konfigurasi halaman
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ABSA Shopee Mykonos",
    page_icon  = "🛍️",
    layout     = "wide",
)

# ─────────────────────────────────────────────────────────────
# Konstanta tampilan
# ─────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = "checkpoints/best_model.pt"

SENTIMENT_STYLE = {
    "Positif": "success",
    "Negatif": "error",
    "Netral" : "warning",
}

ASPECT_ICONS = {
    "Kualitas Produk"     : "📦",
    "Harga"               : "💰",
    "Kualitas Pengiriman" : "🚚",
    "Kepuasan Pelanggan"  : "😊",
}


# ─────────────────────────────────────────────────────────────
# Caching model & tokenizer
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model IndoBERT…")
def load_model_and_tokenizer(checkpoint_path: str):
    """
    Muat tokenizer dan model dari checkpoint yang sudah dilatih.

    Returns (model, tokenizer, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ABSAIndoBERTGated()

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        st.sidebar.success(f"✅ Model dimuat: `{checkpoint_path}`")

        if "val_f1" in ckpt:
            st.sidebar.info(f"Val F1 saat training: `{ckpt['val_f1']:.4f}`")
    else:
        st.sidebar.warning(
            f"⚠️ Checkpoint tidak ditemukan di `{checkpoint_path}`.  \n"
            "Gunakan bobot awal — **hasil tidak valid**.  \n"
            "Jalankan `python train.py` terlebih dahulu."
        )

    model.to(device).eval()
    tokenizer = get_tokenizer()
    return model, tokenizer, device


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────
def run_inference(
    text     : str,
    model    : ABSAIndoBERTGated,
    tokenizer,
    device   : torch.device,
) -> dict:
    """Inference satu teks ulasan → dict hasil prediksi."""
    # Bersihkan teks sebelum tokenisasi
    text     = clean_text_for_model(text)
    encoding = tokenize_single(text, tokenizer, device=str(device))
    return model.predict(
        input_ids      = encoding["input_ids"],
        attention_mask = encoding["attention_mask"],
        token_type_ids = encoding.get("token_type_ids"),
    )


# ─────────────────────────────────────────────────────────────
# Komponen UI: tampilkan hasil satu sampel
# ─────────────────────────────────────────────────────────────
def display_result(result: dict, review_text: Optional[str] = None) -> None:
    """Tampilkan hasil prediksi aspek & sentimen secara visual."""
    if review_text:
        st.markdown(f"> *{review_text[:300]}{'...' if len(review_text)>300 else ''}*")
        st.markdown("---")

    cols = st.columns(len(ASPECT_NAMES))

    for i, (aspect, col) in enumerate(zip(ASPECT_NAMES, cols)):
        sentiment  = result["sentiments"][i]
        confidence = result["confidences"][i]
        probs      = result["probabilities"][i]
        icon       = ASPECT_ICONS.get(aspect, "🔍")
        style_fn   = getattr(col, SENTIMENT_STYLE.get(sentiment, "info"))

        with col:
            st.markdown(f"**{icon} {aspect}**")
            style_fn(f"**{sentiment}**")
            st.markdown(f"Confidence: `{confidence * 100:.1f}%`")
            st.progress(float(confidence))

            with st.expander("Detail probabilitas"):
                for j, sname in enumerate(SENTIMENT_NAMES):
                    pct = probs[j] * 100
                    st.markdown(f"- **{sname}**: {pct:.1f}%")
                    st.progress(float(probs[j]))


# ─────────────────────────────────────────────────────────────
# Inference batch (CSV)
# ─────────────────────────────────────────────────────────────
def run_batch_inference(
    df       : pd.DataFrame,
    text_col : str,
    model    : ABSAIndoBERTGated,
    tokenizer,
    device   : torch.device,
) -> pd.DataFrame:
    """
    Inference seluruh baris DataFrame dan tambahkan kolom hasil.
    Menggunakan enumerate untuk progress bar yang benar.
    """
    results_list = []
    n            = len(df)
    progress_bar = st.progress(0.0, text="Memproses ulasan…")

    for step, (_, row) in enumerate(df.iterrows()):
        text   = str(row[text_col])
        result = run_inference(text, model, tokenizer, device)
        results_list.append(result)
        progress_bar.progress((step + 1) / n,
                               text=f"Proses {step+1}/{n}…")

    progress_bar.empty()

    df = df.copy()
    for i, aspect in enumerate(ASPECT_NAMES):
        safe = aspect.lower().replace(" ", "_")
        df[f"pred_sentimen_{safe}"]   = [r["sentiments"][i]  for r in results_list]
        df[f"pred_confidence_{safe}"] = [r["confidences"][i] for r in results_list]

    return df


# ─────────────────────────────────────────────────────────────
# Halaman: Analisis Teks Tunggal
# ─────────────────────────────────────────────────────────────
def page_single_text(model, tokenizer, device) -> None:
    st.subheader("✍️ Analisis Teks Tunggal")
    st.markdown(
        "Masukkan teks ulasan produk Shopee Mykonos. "
        "Model akan menganalisis **4 aspek** sekaligus."
    )

    review_text = st.text_area(
        label       = "Teks Ulasan",
        placeholder = "Contoh: Parfumnya wangi banget, tahan lama, tapi pengirimannya agak lama.",
        height      = 120,
    )

    with st.expander("💡 Coba contoh teks"):
        examples = [
            "Parfumnya bagus banget, wanginya enak dan tahan lama. Harganya sesuai kualitas.",
            "Pengiriman sangat lambat, produk lecet saat tiba. Sangat kecewa!",
            "Aroma standar, tidak terlalu kuat. Harga lumayan. Pengiriman tepat waktu.",
            "Wanginya mewah banget, tapi agak mahal. Packaging rapih dan aman.",
        ]
        for ex in examples:
            if st.button(f"📋 {ex[:55]}…", key=ex[:20]):
                st.session_state["example_text"] = ex

    if "example_text" in st.session_state:
        review_text = st.session_state["example_text"]

    if st.button("🔍 Analisis Sekarang", type="primary",
                 disabled=not (review_text or "").strip()):
        if not review_text.strip():
            st.warning("Harap masukkan teks ulasan terlebih dahulu.")
            return

        with st.spinner("Menganalisis…"):
            result = run_inference(review_text, model, tokenizer, device)

        st.markdown("### 📊 Hasil Analisis Sentimen")
        display_result(result, review_text=review_text)


# ─────────────────────────────────────────────────────────────
# Halaman: Upload CSV
# ─────────────────────────────────────────────────────────────
def page_upload_csv(model, tokenizer, device) -> None:
    st.subheader("📂 Upload File CSV")
    st.markdown(
        "Upload file `.csv` yang memiliki kolom berisi teks ulasan. "
        "Sistem akan menganalisis setiap baris dan menambahkan kolom prediksi."
    )

    uploaded_file = st.file_uploader(label="Pilih file CSV", type=["csv"])

    if uploaded_file is None:
        st.info("Belum ada file yang diupload.")
        return

    # Coba baca dengan beberapa encoding
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, sep="\t", encoding="cp1252",
                         on_bad_lines="skip")

    st.markdown(f"**Total baris:** {len(df)} | **Kolom:** {list(df.columns)}")
    st.dataframe(df.head(5), use_container_width=True)

    text_col = st.selectbox(
        "Pilih kolom teks ulasan:",
        options = df.columns.tolist(),
        index   = 0,
    )

    if st.button("⚡ Mulai Analisis Batch", type="primary"):
        with st.spinner("Proses inference…"):
            result_df = run_batch_inference(
                df.copy(), text_col, model, tokenizer, device
            )

        st.success(f"✅ Selesai! {len(result_df)} ulasan dianalisis.")
        st.markdown("### 📋 Hasil Analisis")
        st.dataframe(result_df, use_container_width=True)

        csv_result = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "💾 Unduh Hasil CSV",
            data      = csv_result,
            file_name = "hasil_absa.csv",
            mime      = "text/csv",
        )

        st.markdown("### 📈 Distribusi Sentimen per Aspek")
        summary_cols = st.columns(len(ASPECT_NAMES))
        for i, aspect in enumerate(ASPECT_NAMES):
            safe     = aspect.lower().replace(" ", "_")
            col_name = f"pred_sentimen_{safe}"
            if col_name in result_df.columns:
                with summary_cols[i]:
                    st.markdown(f"**{ASPECT_ICONS.get(aspect,'')} {aspect}**")
                    counts = result_df[col_name].value_counts()
                    st.bar_chart(counts)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    st.title("🛍️ Sistem ABSA Shopee Mykonos")
    st.markdown(
        "**Aspect-Based Sentiment Analysis** menggunakan "
        "**IndoBERT + Gated-Attention**  \n"
        "Analisis sentimen pada 4 aspek: "
        "Kualitas Produk · Harga · Kualitas Pengiriman · Kepuasan Pelanggan"
    )
    st.markdown("---")

    # Sidebar
    st.sidebar.title("⚙️ Pengaturan")
    st.sidebar.markdown("**ABSA Shopee Mykonos**")
    st.sidebar.markdown("*IndoBERT + Gated-Attention*")
    st.sidebar.markdown("---")

    menu = st.sidebar.radio(
        "📌 Pilih Menu",
        options=["Analisis Teks Tunggal", "Upload CSV"],
    )

    checkpoint_path = st.sidebar.text_input(
        "📁 Path Checkpoint Model",
        value=DEFAULT_CHECKPOINT,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Aspek Dianalisis:**")
    for aspect in ASPECT_NAMES:
        st.sidebar.markdown(f"- {ASPECT_ICONS.get(aspect,'•')} {aspect}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Label Sentimen:**")
    st.sidebar.markdown("🟢 Positif &nbsp; 🔴 Negatif &nbsp; 🟡 Netral")

    model, tokenizer, device = load_model_and_tokenizer(checkpoint_path)

    if menu == "Analisis Teks Tunggal":
        page_single_text(model, tokenizer, device)
    else:
        page_upload_csv(model, tokenizer, device)

    st.markdown("---")
    st.caption(
        "© 2026 M. Hisyam Ahmad Hasan Hazmi · Universitas Nusa Putra Sukabumi · "
        "Powered by IndoBERT + Gated-Attention"
    )


if __name__ == "__main__":
    main()
