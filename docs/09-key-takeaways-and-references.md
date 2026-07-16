# Part 9 — Key takeaways and references

[← Main README](../README.md) · [← Previous](./08-practice-exercises.md)

---

## 19. Key takeaways

### Historical progression

$$
\boxed{
\text{n-grams}
\rightarrow
\text{RNN}
\rightarrow
\text{LSTM/GRU}
\rightarrow
\text{seq2seq attention}
\rightarrow
\text{Transformer}
}
$$

### Main bottlenecks

- **n-grams:** fixed context and sparse counts;
- **RNNs:** vanishing gradients and sequential computation;
- **LSTM/GRU:** better memory, but still sequential;
- **basic seq2seq:** one fixed context vector;
- **Bahdanau/Luong attention:** dynamic context, but recurrent backbone remains;
- **Transformer:** direct token-to-token interaction and parallel training.

### The central attention formula

$$
\boxed{
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
}
$$

### The central conceptual difference

> An RNN transports information through a sequence of hidden states.

> A Transformer retrieves information directly through learned token-to-token relationships.

### Encoder summary

$$
\boxed{
\text{Embeddings + positions}
\rightarrow
\text{Self-attention}
\rightarrow
\text{Add \& Norm}
\rightarrow
\text{FFN}
\rightarrow
\text{Add \& Norm}
}
$$

### Decoder summary

$$
\boxed{
\text{Masked self-attention}
\rightarrow
\text{Cross-attention}
\rightarrow
\text{FFN}
\rightarrow
\text{Vocabulary softmax}
}
$$

---

---

## 20. References

- Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf).
- Hochreiter, S., & Schmidhuber, J. (1997). [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf).
- Cho, K., et al. (2014). [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078).
- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215).
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473).
- Luong, M.-T., Pham, H., & Manning, C. D. (2015). [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025).
- Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

---

A natural next step is to implement the same attention calculation in PyTorch and then build a minimal single-head Transformer encoder from scratch.
