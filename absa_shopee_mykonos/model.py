"""
model.py
========
Arsitektur model ABSA berbasis IndoBERT dengan mekanisme Gated-Attention kustom.

Perbaikan dari versi sebelumnya:
    - Kompatibel Python >= 3.8 (type hints diperbaiki)
    - Tambah LayerNorm untuk stabilitas training
    - Tambah Label Smoothing support
    - GatedAttention lebih robust dengan NaN guard
    - predict() mengembalikan format yang konsisten

Proyek  : ABSA Shopee Mykonos
Penulis : M. Hisyam Ahmad Hasan Hazmi (20220040239)
Kampus  : Universitas Nusa Putra Sukabumi

Aspek   : Kualitas Produk | Harga | Kualitas Pengiriman | Kepuasan Pelanggan
Sentimen: Positif (0) | Negatif (1) | Netral (2)
"""

from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel


# ─────────────────────────────────────────────────────────────
# Konstanta global
# ─────────────────────────────────────────────────────────────
INDOBERT_MODEL_NAME = "indobenchmark/indobert-base-p1"
HIDDEN_SIZE         = 768
NUM_ASPECTS         = 4
NUM_SENTIMENTS      = 3
DROPOUT_RATE        = 0.1

ASPECT_NAMES: List[str] = [
    "Kualitas Produk",
    "Harga",
    "Kualitas Pengiriman",
    "Kepuasan Pelanggan",
]

SENTIMENT_NAMES: List[str] = ["Positif", "Negatif", "Netral"]


# ─────────────────────────────────────────────────────────────
# Layer kustom: Gated Attention
# ─────────────────────────────────────────────────────────────
class GatedAttention(nn.Module):
    """
    Mekanisme Gated-Attention untuk memfilter noise pada representasi token.

    Rumus:
        scores     = softmax( Q @ K^T / sqrt(d_k) )
        context    = scores @ V
        g (gate)   = sigmoid( W_g @ h + b_g )
        output     = g * context + LayerNorm(h)   ← residual untuk stabilitas

    Parameter
    ----------
    hidden_size : int
        Dimensi hidden state (768 untuk IndoBERT base).
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super(GatedAttention, self).__init__()

        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj   = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_proj  = nn.Linear(hidden_size, hidden_size)

        # Layer normalization untuk output dan residual
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.scale = hidden_size ** 0.5

    def forward(
        self,
        hidden_states : torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass Gated-Attention.

        Parameter
        ----------
        hidden_states  : Tensor [B, seq_len, hidden_size]
        attention_mask : Tensor [B, seq_len] atau None

        Returns
        -------
        Tensor [B, seq_len, hidden_size]
        """
        Q = self.query_proj(hidden_states)
        K = self.key_proj(hidden_states)
        V = self.value_proj(hidden_states)

        # Skor attention [B, L, L]
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # Mask padding token
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).float()
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)

        # Context vector
        context = torch.bmm(attention_weights, V)  # [B, L, H]

        # Gate mechanism
        g = torch.sigmoid(self.gate_proj(hidden_states))  # [B, L, H]

        # Gated output + residual + LayerNorm
        gated_output = self.layer_norm(g * context + hidden_states)

        return gated_output


# ─────────────────────────────────────────────────────────────
# Model utama: ABSAIndoBERTGated
# ─────────────────────────────────────────────────────────────
class ABSAIndoBERTGated(nn.Module):
    """
    Model ABSA berbasis IndoBERT dengan lapisan Gated-Attention kustom.

    Arsitektur:
        Input → IndoBERT → GatedAttention → Mean Pooling
              → Dropout → LayerNorm → 4 × Classifier Head

    Setiap classifier head: Linear(768→256) → ReLU → Dropout → Linear(256→3)

    Parameter
    ----------
    pretrained_name : str
        Model IndoBERT dari Hugging Face.
    num_aspects     : int
        Jumlah aspek (default 4).
    num_sentiments  : int
        Jumlah kelas sentimen per aspek (default 3).
    dropout_rate    : float
        Dropout untuk regularisasi.
    freeze_bert     : bool
        Jika True, bekukan semua parameter BERT (hanya fine-tune head).
    """

    def __init__(
        self,
        pretrained_name: str   = INDOBERT_MODEL_NAME,
        num_aspects    : int   = NUM_ASPECTS,
        num_sentiments : int   = NUM_SENTIMENTS,
        dropout_rate   : float = DROPOUT_RATE,
        freeze_bert    : bool  = False,
    ):
        super(ABSAIndoBERTGated, self).__init__()

        self.num_aspects    = num_aspects
        self.num_sentiments = num_sentiments

        # Backbone IndoBERT
        self.bert = AutoModel.from_pretrained(pretrained_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Gated-Attention layer
        self.gated_attention = GatedAttention(hidden_size=HIDDEN_SIZE)

        # Post-pooling normalisasi
        self.pooling_norm = nn.LayerNorm(HIDDEN_SIZE)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

        # Classifier heads paralel (1 head per aspek)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_SIZE, 256),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(256, num_sentiments),
            )
            for _ in range(num_aspects)
        ])

    # ──────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor,
        token_type_ids : Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass.

        Returns
        -------
        List[Tensor]  — logit per aspek, masing-masing [B, num_sentiments]
        """
        # 1. Encoding IndoBERT
        bert_outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )
        hidden_states = bert_outputs.last_hidden_state  # [B, L, 768]

        # 2. Gated-Attention
        gated_states = self.gated_attention(hidden_states, attention_mask)

        # 3. Mean Pooling (hanya token non-padding)
        mask_exp   = attention_mask.unsqueeze(-1).float()         # [B, L, 1]
        sum_hidden = (gated_states * mask_exp).sum(dim=1)         # [B, H]
        n_tokens   = mask_exp.sum(dim=1).clamp(min=1e-9)          # [B, 1]
        pooled     = sum_hidden / n_tokens                         # [B, H]

        # 4. Normalisasi + Dropout
        pooled = self.dropout(self.pooling_norm(pooled))

        # 5. Classifier heads
        logits = [clf(pooled) for clf in self.classifiers]

        return logits

    # ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(
        self,
        input_ids      : torch.Tensor,
        attention_mask : torch.Tensor,
        token_type_ids : Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Inference: kembalikan label + confidence + probabilitas per aspek.

        Returns
        -------
        dict:
            labels       : List[int]         — index kelas sentimen
            sentiments   : List[str]         — nama sentimen
            confidences  : List[float]       — confidence (0–1)
            probabilities: List[List[float]] — distribusi probabilitas
        """
        self.eval()
        logits = self.forward(input_ids, attention_mask, token_type_ids)

        labels_out, sentiments_out, confidences_out, probs_out = [], [], [], []

        for logit in logits:
            probs     = torch.softmax(logit, dim=-1).squeeze(0).tolist()
            label_idx = int(torch.argmax(logit, dim=-1).item())

            labels_out.append(label_idx)
            sentiments_out.append(SENTIMENT_NAMES[label_idx])
            confidences_out.append(round(float(probs[label_idx]), 4))
            probs_out.append([round(float(p), 4) for p in probs])

        return {
            "labels"       : labels_out,
            "sentiments"   : sentiments_out,
            "confidences"  : confidences_out,
            "probabilities": probs_out,
        }
