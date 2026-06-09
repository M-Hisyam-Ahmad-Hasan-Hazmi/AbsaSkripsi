# 📚 Dokumentasi Lengkap: ABSA dengan IndoBERT & Gated Attention

**Proyek:** ABSA Shopee Mykonos  
**Institusi:** Universitas Nusa Putra Sukabumi  
**Penulis:** M. Hisyam Ahmad Hasan Hazmi (20220040239)

---

## 📖 Daftar Isi

1. [Pengenalan ABSA](#pengenalan-absa)
2. [IndoBERT Architecture](#indobert-architecture)
3. [Gated Attention Mechanism (Detail)](#gated-attention-mechanism-detail)
4. [Model Architecture](#model-architecture)
5. [Training & Optimization](#training--optimization)
6. [Production Deployment](#production-deployment)
7. [Best Practices](#best-practices)

---

## 🎯 Pengenalan ABSA

### Apa itu ABSA?

**Aspect-Based Sentiment Analysis (ABSA)** adalah teknik NLP yang menganalisis sentimen terhadap **aspek-aspek spesifik** dalam teks, bukan sentimen keseluruhan.

#### Contoh:
```
Text: "Produk berkualitas baik, tapi harganya mahal dan pengiriman lambat"

Analisis Keseluruhan (Traditional SA):
  Sentimen: Netral (campur positif & negatif)

Analisis Berbasis Aspek (ABSA):
  Kualitas Produk:        Positif ✓
  Harga:                  Negatif ✗
  Kualitas Pengiriman:    Negatif ✗
  Kepuasan Pelanggan:     Netral ○
```

### Mengapa ABSA Penting?

| Aspek | Keuntungan |
|-------|-----------|
| **Business Insight** | Tahu area mana yang perlu perbaikan |
| **Customer Feedback** | Identifikasi masalah spesifik |
| **Product Development** | Prioritas pengembangan yang tepat |
| **Market Analysis** | Analisis kompetitor lebih detail |
| **Brand Management** | Kelola reputasi per kategori |

### Task dalam ABSA

ABSA biasanya memiliki 3 subtask:

```
1. Aspect Extraction
   Input: "Produk berkualitas, harganya mahal"
   Output: [Produk, Harga]

2. Opinion Extraction
   Input: "Produk berkualitas, harganya mahal"
   Output: [berkualitas, mahal]

3. Aspect Sentiment Classification (yang kami lakukan)
   Input: "Kualitas Produk: berkualitas"
   Output: Positif ✓
```

---

## 🏗️ IndoBERT Architecture

### Apa itu IndoBERT?

**IndoBERT** adalah model BERT yang di-fine-tune khusus untuk **Bahasa Indonesia** menggunakan dataset besar (Wikipedia, Common Crawl, BookCorpus).

```
BERT (English)
    ↓
IndoBERT Tokenizer (SentencePiece - support Bahasa Indonesia)
    ↓
Pre-training pada corpus Indonesia
    ↓
IndoBERT Model (indobenchmark/indobert-base-p1)
```

### Spesifikasi IndoBERT Base P1

```
Model Size:        Base (tidak Large)
Hidden Size:       768
Num Layers:        12 (Transformer blocks)
Num Attention Heads: 12
Intermediate Size: 3072
Vocabulary Size:   30,000
Parameters:        ~110M
Pre-training Task: MLM + NSP

MLM (Masked Language Model):
  Input:  "Produk [MASK] dan harganya mahal"
  Target: [MASK] → "berkualitas"

NSP (Next Sentence Prediction):
  Sentence A: "Produk berkualitas."
  Sentence B: "Harganya mahal."
  Target: IsNext → True
```

### Why IndoBERT vs Multilingual BERT?

```
┌─────────────────────────────────────────────────┐
│ Multilingual BERT                               │
├─────────────────────────────────────────────────┤
│ ✓ Support 104+ bahasa                           │
│ ✗ Vocabulary dibagi untuk semua bahasa          │
│ ✗ Kurang optimal untuk bahasa tertentu          │
│ ✗ Lebih besar (ukuran model)                    │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ IndoBERT                                        │
├─────────────────────────────────────────────────┤
│ ✓ Optimized untuk Bahasa Indonesia              │
│ ✓ Vocabulary lebih fokus & efisien              │
│ ✓ Pre-trained pada corpus Indonesia besar       │
│ ✓ Lebih ringan & cepat                          │
│ ✓ Better performance untuk Indonesian tasks     │
└─────────────────────────────────────────────────┘
```

### Transfer Learning dengan IndoBERT

```
Pre-trained IndoBERT (frozen)
    ↓
Fine-tune pada task ABSA
    ↓
Add Custom Head untuk ABSA
    ↓
Train dengan labeled data (Shopee reviews)
```

---

## 🔥 Gated Attention Mechanism (Detail)

### Problem yang Ingin Diselesaikan

Dalam ABSA, banyak token yang **tidak relevan** dengan aspek tertentu:

```
Text: "Produk berkualitas baik, tapi harganya mahal dan pengiriman lambat"

Aspek: "Harga"
Token penting:     [harganya, mahal]
Token noise:       [Produk, berkualitas, pengiriman, lambat]
                    ↓
                    Attention bisa ter-distraksi oleh noise
```

**Solusi:** Gated Attention — gunakan "gate" untuk filter mana token yang penting.

---

### 1️⃣ Standard Attention (Baseline)

#### Rumus:

```
Attention(Q, K, V) = softmax( Q @ K^T / √d_k ) @ V
```

#### Step-by-step:

```python
# 1. Project to Q, K, V
Q = input @ W_Q    # [B, L, d]
K = input @ W_K    # [B, L, d]
V = input @ W_V    # [B, L, d]

# 2. Compute attention scores
scores = (Q @ K^T) / √d    # [B, L, L]
        
# 3. Softmax untuk normalisasi
weights = softmax(scores)   # [B, L, L]

# 4. Apply attention weights to values
output = weights @ V        # [B, L, d]
```

#### Visualisasi:

```
Input tokens:  [<CLS>, Produk, berkualitas, harganya, mahal, ...]
                   ↓         ↓              ↓         ↓
                   ┌─────────────────────────────────┐
                   │   Self-Attention Mechanism      │
                   │                                 │
                   │  Compute Q, K, V                │
                   │  → Scores = Q @ K^T / √d        │
                   │  → Softmax                      │
                   │  → Output = Weights @ V         │
                   └─────────────────────────────────┘
                   ↓         ↓              ↓         ↓
Output:       [<CLS>, v1,    v2,           v3,       v4,    ...]
```

---

### 2️⃣ Gated Attention (Kami Gunakan)

#### Problem dengan Standard Attention:
- ❌ Semua token mendapat bobot attention (termasuk noise)
- ❌ Gate mechanism tidak ada untuk filter informatif vs non-informatif
- ❌ Pada ABSA, kita perlu fokus ke token yang relevan dengan aspek

#### Solusi: Tambahkan Gate Mechanism

#### Rumus:

```
Q = input @ W_Q
K = input @ W_K
V = input @ W_V

scores = softmax( Q @ K^T / √d_k )
context = scores @ V                    ← Context dari attention

g = sigmoid( W_g @ input + b_g )        ← Gate: input-dependent
                                           sigmoid → output [0, 1]

gated_output = g ⊙ context + input      ⊙ = element-wise multiplication
                                          + residual connection

output = LayerNorm(gated_output)
```

#### Step-by-step Gated Attention:

```python
# 1. Standard Attention
Q = self.query_proj(hidden_states)           # [B, L, H]
K = self.key_proj(hidden_states)             # [B, L, H]
V = self.value_proj(hidden_states)           # [B, L, H]

# 2. Scaled Dot-Product Attention
scores = torch.bmm(Q, K.transpose(1, 2)) / sqrt(H)  # [B, L, L]

# 3. Apply attention mask untuk padding
if attention_mask is not None:
    scores = scores.masked_fill(attention_mask == 0, -1e9)

# 4. Softmax
attention_weights = torch.softmax(scores, dim=-1)  # [B, L, L]

# 5. Apply attention to values
context = torch.bmm(attention_weights, V)   # [B, L, H]

# ─── GATING MECHANISM ───

# 6. Compute gate
g = torch.sigmoid(self.gate_proj(hidden_states))  # [B, L, H]
                                                   # Output [0, 1]

# 7. Apply gate to context
gated = g * context                          # [B, L, H]
                                            # Hanya ambil 
                                            # informasi penting

# 8. Residual + LayerNorm
output = self.layer_norm(gated + hidden_states)  # [B, L, H]
```

---

### 3️⃣ Visualisasi Gated Attention

```
Input: "Produk berkualitas, harganya mahal"
        [p1,   p2,       p3, p4,       p5,  ...]

─── STEP 1: Attention ───
        ┌──────────────────────────────┐
        │ Self-Attention (Q, K, V)     │
        │ attention_weights @ V        │
        └──────────────────────────────┘
                    ↓
        context: [c1, c2, c3, c4, c5, ...]

─── STEP 2: Gate Computation ───
        g = sigmoid(W_g @ hidden_states)
        g: [0.2, 0.9, 0.1, 0.8, 0.3, ...]
           ↑ low    ↑ high  ↑ low
        Token "Produk" → gate=0.2 (kurang penting untuk "Harga")
        Token "harganya" → gate=0.9 (sangat penting untuk "Harga")

─── STEP 3: Apply Gate ───
        gated = g ⊙ context
        
        gated[1] = 0.2 × c1  ← Diminimalkan
        gated[3] = 0.9 × c3  ← Diperkuat
        
─── STEP 4: Residual + LayerNorm ───
        output = LayerNorm(gated + hidden_states)
```

---

### 4️⃣ Keuntungan Gated Attention

```
┌─────────────────────────────────────────┐
│ Standard Attention                      │
├─────────────────────────────────────────┤
│ ✓ Flexible, learn any token relevance   │
│ ✗ Semua token dapat bobot (termasuk     │
│   noise)                                │
│ ✗ Untuk ABSA, bisa ter-distraksi       │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Gated Attention                         │
├─────────────────────────────────────────┤
│ ✓ Soft filtering via sigmoid gate       │
│ ✓ Mengurangi noise dari token           │
│   irrelevant                            │
│ ✓ Better untuk ABSA (fokus ke aspek)    │
│ ✓ Residual connection → stabilitas      │
│ ✓ LayerNorm → normalisasi output        │
└─────────────────────────────────────────┘
```

---

### 5️⃣ Mathematical Intuition

#### Gate sebagai Soft Mask:

```
Tanpa Gate (standard attention):
output = attention_output
        ↓ Semua informasi diambil

Dengan Gate:
output = g ⊙ attention_output
g ∈ [0, 1]
        ↓ Hanya ambil informasi penting (g ≈ 1)
        ↓ Filter noise (g ≈ 0)

Analogi:
g = volume control
  - g=0   → mute (silent)
  - g=0.5 → medium volume
  - g=1.0 → full volume
```

#### Sigmoid vs ReLU untuk Gate:

```
sigmoid(x) = 1 / (1 + e^(-x))
Output range: [0, 1]
             ↑ Smooth, differentiable

g = sigmoid(W_g @ h) ← Kami gunakan ini
- Bounded output [0, 1]
- Smooth gradient untuk backprop
- Interpretable: 0=mute, 1=full

vs

ReLU(x) = max(0, x)
Output range: [0, ∞)
- Unbounded, bisa explode
- Tidak cocok untuk gating
```

---

## 🏛️ Model Architecture

### Alur Lengkap Model ABSA

```
INPUT TEXT
   ↓
"Produk berkualitas, harganya mahal, pengiriman lambat"
   ↓
[TOKENIZATION] (IndoBERT Tokenizer - SentencePiece)
   ↓
Token IDs: [101, 5067, 15403, 1006, 10405, 2088, ...]
           [<CLS>, Produk, berkualitas, harganya, mahal, ...]
   ↓
[EMBEDDING]
   ↓
Embedding vectors: [768-dim each]
   ↓
[INDOBERT ENCODER] (12 layers × 12 attention heads)
   ↓
hidden_states: [B=1, L=10, H=768]
   ↓
[GATED ATTENTION] (Custom Layer)
   ↓
gated_states: [B=1, L=10, H=768]
   ↓
[MEAN POOLING] (Average across sequence)
   ↓
pooled: [B=1, H=768]
   ↓
[DROPOUT + LAYER NORM]
   ↓
normalized: [B=1, 768]
   ↓
┌──────────────────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────────────────┐
│                              │                              │                              │                              │
↓                              ↓                              ↓                              ↓
[CLASSIFIER HEAD 1]      [CLASSIFIER HEAD 2]      [CLASSIFIER HEAD 3]      [CLASSIFIER HEAD 4]
Kualitas Produk          Harga                    Pengiriman               Kepuasan
[768→256→3]              [768→256→3]              [768→256→3]              [768→256→3]
↓                        ↓                        ↓                        ↓
logits1: [1, 3]          logits2: [1, 3]          logits3: [1, 3]          logits4: [1, 3]
         softmax                 softmax                    softmax                  softmax
         ↓                        ↓                         ↓                        ↓
pred1: Positif           pred2: Negatif           pred3: Negatif           pred4: Netral
conf1: 0.92              conf2: 0.87              conf3: 0.79              conf4: 0.65

OUTPUT
{
  "Kualitas Produk": {
    "sentiment": "Positif",
    "confidence": 0.92,
    "probabilities": [0.92, 0.05, 0.03]
  },
  "Harga": {
    "sentiment": "Negatif",
    "confidence": 0.87,
    ...
  },
  ...
}
```

### Code Implementation

```python
class ABSAIndoBERTGated(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Backbone: IndoBERT
        self.bert = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
        
        # 2. Custom Gated Attention Layer
        self.gated_attention = GatedAttention(hidden_size=768)
        
        # 3. Post-pooling normalization
        self.pooling_norm = nn.LayerNorm(768)
        
        # 4. Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 5. Classifier heads (1 per aspect)
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 3),  # 3 sentiments
            )
            for _ in range(4)  # 4 aspects
        ])
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # 1. IndoBERT encoding
        bert_out = self.bert(input_ids, attention_mask, token_type_ids)
        hidden = bert_out.last_hidden_state  # [B, L, 768]
        
        # 2. Gated Attention
        gated = self.gated_attention(hidden, attention_mask)
        
        # 3. Mean Pooling
        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled = (gated * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)
        
        # 4. Normalization + Dropout
        pooled = self.dropout(self.pooling_norm(pooled))
        
        # 5. Multi-head classification
        logits = [clf(pooled) for clf in self.classifiers]
        
        return logits
```

---

## 🚀 Training & Optimization

### Hyperparameter Tuning

```python
# Current (Baseline)
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 10
WARMUP_RATIO = 0.1

# Optimized (Recommended)
LEARNING_RATE = 3e-5        # Sedikit lebih tinggi untuk convergence
BATCH_SIZE = 32             # Lebih besar untuk stable gradients
EPOCHS = 15                 # Lebih lama untuk fine-tuning
WARMUP_RATIO = 0.15         # Lebih lama warm-up
WEIGHT_DECAY = 0.01         # L2 regularization
DROPOUT = 0.15              # Sedikit lebih tinggi untuk regularization
```

### Strategi Optimasi

#### 1. Early Stopping + Learning Rate Warmup
```
Epoch 1-2 (Warmup):   LR 0% → 15%  (gradual increase)
Epoch 3-12 (Main):    LR 15% → 100% (linear warm-up)
Epoch 13-15 (Decay):  LR 100% → 0%  (linear decay)

Benefit:
- Smooth convergence di awal
- Menghindari divergence
- Better generalization
```

#### 2. Gradient Accumulation (untuk dataset kecil)
```python
accumulation_steps = 2
optimizer.zero_grad()
for step, batch in enumerate(loader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (step + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 3. Label Smoothing
```python
# Mengurangi overfitting dengan smooth target distribution
label_smoothing = 0.1

# Hard target:  [1, 0, 0]
# Smooth:       [0.9, 0.05, 0.05]

criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
```

#### 4. Class Weighting (untuk imbalanced dataset)
```python
# Hitung bobot per kelas
class_counts = [100, 50, 30]  # Negatif | Positif | Netral
total = sum(class_counts)
weights = [total / (3 * c) for c in class_counts]
       # [0.33, 0.67, 1.0]

criterion = nn.CrossEntropyLoss(weight=weights)
```

#### 5. Layer-wise Learning Rate Decay (LLRD)
```python
# Different LR untuk berbagai layer BERT
# Later layers (closer to output): higher LR
# Earlier layers (representation): lower LR

lr_backbone = 1e-5
lr_head = 5e-5

group1 = [BERT layers]     → lr=1e-5
group2 = [Gated Attention] → lr=2e-5
group3 = [Classifiers]     → lr=5e-5
```

---

## 🌐 Production Deployment

### Strategy 1: REST API (Flask/FastAPI)

```python
# app.py
from fastapi import FastAPI
from transformers import AutoTokenizer
import torch

app = FastAPI()

# Load model (once at startup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ABSAIndoBERTGated().to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

@app.post("/predict")
async def predict(text: str):
    inputs = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    result = model.predict(**inputs)
    return result

# Jalankan: uvicorn app:app --host 0.0.0.0 --port 8000
```

### Strategy 2: Streamlit Web App

```python
# app_streamlit.py (sudah ada di repo)
import streamlit as st

st.title("ABSA Shopee Mykonos Analyzer")

text = st.text_area("Masukkan review:")
if st.button("Analisis"):
    result = model.predict(text)
    
    for i, aspect in enumerate(ASPECT_NAMES):
        sentiment = result["sentiments"][i]
        confidence = result["confidences"][i]
        
        st.metric(
            label=aspect,
            value=sentiment,
            delta=f"Confidence: {confidence}"
        )

# Jalankan: streamlit run app_streamlit.py
```

### Strategy 3: Docker Containerization

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model files
COPY absa_shopee_mykonos/ ./absa_shopee_mykonos/
COPY checkpoints/ ./checkpoints/

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Strategy 4: Model Optimization untuk Production

#### a) Quantization (Kurangi ukuran model)
```python
# 4-bit quantization
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("...")

# Convert ke int8
quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Hasil: 1/4 ukuran, inference lebih cepat
```

#### b) Knowledge Distillation (Buat model lebih kecil)
```python
# Teacher model (besar, akurat)
teacher = ABSAIndoBERTGated()

# Student model (kecil, cepat)
class StudentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(...)  # Lebih kecil
        # ... simpler architecture
```

#### c) ONNX Export (Cross-platform)
```python
import onnx
from transformers import convert_pytorch_to_onnx

model_path = "checkpoints/best_model.pt"
onnx_path = "model.onnx"

convert_pytorch_to_onnx(
    model_path,
    onnx_path,
    opset_version=14,
)

# Bisa dijalankan di:
# - ONNX Runtime (C++, Java, etc)
# - TensorRT (NVIDIA GPU optimization)
# - CoreML (Apple devices)
```

---

## 💡 Best Practices

### 1. Data Preprocessing

```python
# ✓ GOOD
text = "Produk berkualitas, harganya mahal"
# - Clean & normalized
# - Lowercase
# - No extra whitespace

# ✗ BAD
text = "PRODUK   BERKUALITAS,,,   harganya  MAHAL  !!!"
# - Extra spaces, punctuation, mixed case
```

### 2. Batching & Padding

```python
# ✓ GOOD - Gunakan DataLoader dengan padding
loader = DataLoader(dataset, batch_size=32, 
                   collate_fn=custom_collate)

# ✗ BAD - Padding manual setiap batch
inputs = pad_sequences(batch, maxlen=128)
```

### 3. Model Checkpointing

```python
# ✓ GOOD - Save best model only
if val_f1 > best_f1:
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'metrics': val_f1,
    }, 'best_model.pt')

# ✗ BAD - Save setiap epoch
torch.save(model.state_dict(), f'model_epoch_{epoch}.pt')
```

### 4. Evaluation Metrics untuk ABSA

```python
# Multi-aspect evaluation
for aspect_idx, aspect_name in enumerate(ASPECTS):
    preds = predictions[:, aspect_idx]
    labels = labels[:, aspect_idx]
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    print(f"{aspect_name}: Acc={accuracy:.4f}, F1={f1:.4f}")

# Macro-average (sama bobot semua aspek)
macro_f1 = np.mean([f1_score(...) for each aspect])
```

### 5. Error Analysis

```python
# Identifikasi error patterns
misclassified = []
for pred, true, text in zip(predictions, labels, texts):
    if pred != true:
        misclassified.append({
            'text': text,
            'pred': pred,
            'true': true,
            'aspect': ASPECTS[aspect_idx]
        })

# Analisis: error lebih banyak di aspek mana?
#          apa pattern dari misclassified examples?
```

---

## 📊 Performance Metrics

### Target Metrics

```
Baseline (Standard IndoBERT):
- Accuracy: 0.75
- F1-score: 0.72

Dengan Gated Attention:
- Accuracy: 0.82 (+7%)
- F1-score: 0.80 (+8%)

Dengan Full Optimization:
- Accuracy: 0.85-0.88
- F1-score: 0.83-0.86
```

---

## 🔗 References

1. **BERT**: Devlin et al., 2018 - "BERT: Pre-training of Deep Bidirectional Transformers"
2. **IndoBERT**: Wilie et al., 2020 - "IndoNLU: Benchmark and Resources for the Indonesian Language"
3. **ABSA**: Pontiki et al., 2016 - "SemEval-2016 Task 5: Aspect Based Sentiment Analysis"
4. **Gated Attention**: Huang et al., 2018 - "Gated Attention Networks"

---

**Happy Learning! 🚀**
