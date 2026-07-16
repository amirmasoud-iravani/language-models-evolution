# Part 6 — Strengths and limitations

[← Main README](../README.md) · [← Previous](./05-transformer-decoder-and-training.md) · [Next →](./07-numpy-implementation.md)

---

## 15. Why Transformers are strong

### Comparison

| Model | Context mechanism | Parallel training | Main bottleneck |
|---|---|---:|---|
| n-gram | Fixed $n-1$ history | High | Sparse counts and short context |
| Feed-forward neural LM | Learned embeddings, fixed window | High | Fixed context |
| RNN | Recurrent hidden state | Low | Long recurrent paths |
| LSTM/GRU | Gated recurrent memory | Low | Still sequential |
| RNN + attention | Dynamic decoder context | Partly sequential | Recurrence remains |
| Transformer | Self-attention and cross-attention | High | Quadratic attention cost |

### 15.1 Shorter paths between distant tokens

In an RNN, information from token $1$ may pass through many hidden states before reaching token $n$:

$$
x_1\rightarrow h_1\rightarrow h_2\rightarrow\cdots\rightarrow h_n
$$

In self-attention, token $1$ can interact directly with token $n$ in one layer:

$$
x_1\leftrightarrow x_n
$$

This makes long-range dependencies easier to represent.

### 15.2 Parallel computation

All encoder tokens can compute queries, keys, and values simultaneously.

RNN states inside one layer must be computed sequentially.

This makes Transformer training more suitable for GPUs and TPUs.

### 15.3 Dynamic context for every token

Bahdanau and Luong attention gave the decoder dynamic access to encoder states.

Transformer self-attention gives every token dynamic access to every relevant token in the same sequence.

### 15.4 Multiple relation types

Multi-head attention allows several attention patterns to be learned in parallel.

### 15.5 Better scaling

Transformer architectures scale effectively with:

- more data;
- more parameters;
- more compute;
- pretraining objectives;
- transfer learning.

This scaling behavior helped make models such as BERT, T5, GPT, and many multilingual models possible.

---

---

## 16. Limitations

Transformers solve several recurrent-model bottlenecks, but they introduce new ones.

### 16.1 Quadratic self-attention

For $n$ tokens, the score matrix has shape:

$$
n\times n
$$

Standard self-attention therefore requires roughly:

$$
O(n^2)
$$

pairwise scores.

Examples:

$$
1{,}000^2=1{,}000{,}000
$$

but:

$$
100{,}000^2=10^{10}
$$

Very long sequences become expensive.

### 16.2 Position must be added explicitly

Self-attention has no inherent left-to-right recurrence. Order must be represented with positional encodings, learned position embeddings, relative positions, rotary embeddings, or related methods.

### 16.3 Autoregressive generation remains sequential

Training can process many target positions in parallel because masking prevents leakage.

During ordinary generation, however:

$$
y_1\rightarrow y_2\rightarrow y_3\rightarrow\cdots
$$

Each new token depends on previous generated tokens.

### 16.4 Data and compute requirements

Large Transformers often require:

- large datasets;
- substantial memory;
- careful optimization;
- significant computational resources.

For small datasets or very short sequences, simpler architectures can still be useful.

---
