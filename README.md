# Deep Dive into Transformer

A from-scratch implementation and systematic comparison of Transformer architectures, exploring Encoder-Decoder vs Decoder-only models with various architectural choices.

## Overview

This project pursues three main objectives:

1. **Deep Understanding**: Implement the Transformer architecture from scratch without relying on pre-built libraries
2. **Architecture Comparison**: Investigate two variations — **Encoder-Decoder (ED)** and **Decoder-only (D-only)**
3. **Design Choices Analysis**: Compare performance across different architectural choices:
   - Layer normalization placement (Pre-LN vs Post-LN)
   - Activation functions (ReLU, GELU, Swish, GLU variants)
   - Positional encoding methods (Sinusoidal vs Learned)

## Transformer

Our implementation includes:

- **Scaled Dot-Product Attention**: Core attention mechanism with optional masking
- **Multi-Head Attention**: Parallel attention heads for different representation subspaces
- **Position-wise Feed-Forward Networks**: Two linear transformations with activation
- **Positional Encoding**: Both sinusoidal (fixed) and learned embeddings
- **Activation function**: ReLU, GELU, GLU, FFN_GLUM FFN_ReGLU, FFN_GEGLU, FFN_SWIGLU

## Datasets

### Text Generation (Decoder-only)

| Dataset | Size | Description |
|---------|------|-------------|
| Names | 32k names | Character-level name generation |
| Poetry | 17k poems | Russian classic poetry (BPE tokenization) |

### Machine Translation (Encoder-Decoder & Decoder-only)

| Dataset | Size | Description |
|---------|------|-------------|
| Eng-Rus | 790k pairs | English-Russian sentence pairs |

## Experiments

### Model Configurations

**Names Dataset:**
- Embedding: 8, FFN hidden: 64, Heads: 8×64
- Sequence length: 10, Dropout: 0.3, Batch: 32
- ~100k parameters

**Poetry Dataset:**
- Embedding: 256, FFN hidden: 384, Heads: 8×32
- Sequence length: 512, Dropout: 0.5, Batch: 128
- ~7M parameters

### Generation Examples

**Names Generation:**
```
semlyn, lailana, cheongua
shesly, ebfemon, denalleea
deyr, onsist, hamaiel, motes

```

**Poetry Generation:**
```
И в звон зубах лед и гроз,
И мухомзы и озноб
Зорь, чуть ослепленный зазвон.
Лежит в печчитаньи шваль.
Возванит, то летит роман,
Весь высохший саблею блеснет,
И в зелени
```

```
Лежит у соседней вязей,
С веток дугой стоит терпит.
Час поговорот.
Идет, не идет, у шлет
Разxтая причины ждет,
Треляет, озор!
Он знает - как выкнут или выше,
Чудотворный вопрос:
Цветы если б, даже не расстаться,
К Орфаши-пестины,
В наездниках у снегов,
Пока они обзели нет,
И сколько их с гжунглипусочек
Искал команлице,—
```

## Authors

- **Tarasov M.**
- **Mustashkin A.**

**Mentor:** Anisimova N.

---

<p align="center">
  <i>AI360 Project School</i>
</p>