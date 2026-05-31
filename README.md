# GPT-2 From Scratch using PyTorch

## Introduction

This project implements a GPT-2 style language model completely from scratch using PyTorch.

The objective is to understand every component of a modern autoregressive transformer-based language model including tokenization, attention mechanisms, transformer blocks, training, and text generation.

Unlike using pretrained HuggingFace models, every component was implemented manually.

The project covers:

- Byte Pair Encoding (BPE)
- Token Embeddings
- Positional Embeddings
- Layer Normalization
- Multi-Head Causal Self Attention
- Feed Forward Networks
- Decoder-only Transformer
- Cross Entropy Training
- Temperature Sampling
- Top-K Sampling
- Multinomial Sampling

---

# Inference Results

## Text Generation

<p align="center">
  <img src="assets/gpt2_generation.gif" width="900">
</p>

The GIF demonstrates autoregressive text generation where the model predicts one token at a time and appends it back to the sequence until the desired context length is reached.

---

# Google Colab

🔗 PASTE_COLAB_LINK

---

# Handwritten Report

📄 PASTE_HANDWRITTEN_REPORT_LINK

---

# Trained Model

📦 PASTE_MODEL_LINK

---

# Theory

# Dataset and Preprocessing

The training corpus was collected from public-domain books obtained from Project Gutenberg.

Preprocessing steps:

- Remove table of contents
- Remove extra spaces
- Remove unnecessary symbols
- Clean text formatting
- Convert text into token sequences

The cleaned text corpus is then used to train the GPT model. :contentReference[oaicite:1]{index=1}

---

# Byte Pair Encoding (BPE)

GPT-2 uses Byte Pair Encoding for tokenization.

Instead of storing complete words in the vocabulary, BPE learns frequently occurring subword units.

Example:

```text
low
lowest
newer
wider
```

The algorithm repeatedly:

1. Counts adjacent token pairs
2. Finds the most frequent pair
3. Merges the pair
4. Repeats until the vocabulary size is reached

Advantages:

- Handles unknown words
- Smaller vocabulary
- Better generalization
- Efficient tokenization

The GPT-2 tokenizer contains approximately 50,257 tokens. :contentReference[oaicite:2]{index=2}

---

# GPT-2 Architecture

GPT-2 is a Decoder-Only Transformer.

Unlike encoder-decoder architectures, GPT only uses transformer decoder blocks.

A causal attention mask ensures that future tokens are hidden from the current token. :contentReference[oaicite:3]{index=3}

---

## Training Configuration

| Parameter | Value |
|------------|------------|
| Vocabulary Size | 50,257 |
| Embedding Dimension | 512 |
| Attention Heads | 8 |
| Transformer Layers | 6 |
| Context Length | 256 |
| Learning Rate | 3e-4 |
| Batch Size | 128 |
| Epochs | 10 |

:contentReference[oaicite:4]{index=4}

---

# Overall Architecture

```text
Input Tokens
      │
      ▼
Token Embedding
      │
      ▼
Positional Embedding
      │
      ▼
Transformer Block × 6
      │
      ▼
Final LayerNorm
      │
      ▼
Linear Output Layer
      │
      ▼
Vocabulary Probabilities
```

---

# Token Embedding

Input tensor:

```text
[B, Context_Length]
```

After embedding:

```text
[B, 256, 512]
```

The embedding matrix maps token IDs into dense vector representations. :contentReference[oaicite:5]{index=5}

---

# Positional Embedding

Since transformers do not contain recurrence, positional embeddings are added to encode token order.

Position embedding matrix:

```text
[256, 512]
```

After adding token embeddings and positional embeddings:

```text
[B,256,512]
```

:contentReference[oaicite:6]{index=6}

---

# Layer Normalization

LayerNorm normalizes activations across the embedding dimension.

For each token:

1. Compute mean
2. Compute variance
3. Normalize activations
4. Apply learnable scale and bias

This improves training stability and convergence. :contentReference[oaicite:7]{index=7}

---

# Multi-Head Causal Self Attention

The model uses:

- 8 Attention Heads
- Head Dimension = 64

Query, Key and Value matrices:

```text
Q = XWq
K = XWk
V = XWv
```

The attention mechanism computes:

```text
Attention(Q,K,V)
=
Softmax(QKᵀ / √dk)V
```

A causal mask is applied before softmax so future tokens remain inaccessible to the current token. :contentReference[oaicite:8]{index=8}

---

# Feed Forward Network (MLP)

The MLP expands the embedding dimension:

```text
512 → 2048 → 512
```

Activation:

```text
GELU
```

The feed-forward network enables non-linear transformations of token representations. :contentReference[oaicite:9]{index=9}

---

# Output Layer

The final transformer representation passes through:

```text
Linear(512 → 50257)
```

Output:

```text
[B,256,50257]
```

representing probabilities for every vocabulary token. :contentReference[oaicite:10]{index=10}

---

# Loss Function

Cross Entropy Loss is used during training.

```text
Loss = -Σ yi log(ŷi)
```

The model learns to predict the next token given all previous tokens. :contentReference[oaicite:11]{index=11}

---

# Inference and Text Generation

During inference:

1. Input prompt is tokenized.
2. Tokens are passed through GPT.
3. Next-token probabilities are predicted.
4. A token is sampled.
5. The generated token is appended to the sequence.
6. Repeat until the context limit is reached.

This process produces autoregressive text generation. :contentReference[oaicite:12]{index=12}

---

# Temperature Sampling

Temperature controls randomness.

- T < 1 → More deterministic
- T > 1 → More random

Example:

```text
Temperature = 0.5
→ High confidence predictions

Temperature = 1.5
→ More diverse generations
```

:contentReference[oaicite:13]{index=13}

---

# Top-K Sampling

Only the K highest probability tokens are retained.

Example:

```text
K = 50
```

All other tokens are masked out.

Sampling is then performed only among the top-K candidates.

This improves generation quality and reduces garbage outputs. :contentReference[oaicite:14]{index=14}

---

# Conclusion

This project demonstrates a complete implementation of GPT-2 from scratch, covering tokenization, transformer architecture, causal self-attention, training, and autoregressive text generation. The implementation provides a detailed understanding of how modern large language models operate internally.

---

## Author

**Pranav Deshpande**  
* IIT Jodhpur
* Deep Learning
* NLP
* Transformers
* Large Language Models
