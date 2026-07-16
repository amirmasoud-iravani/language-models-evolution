# Part 7 — NumPy implementation

[← Main README](../README.md) · [← Previous](./06-strengths-and-limitations.md) · [Next →](./08-practice-exercises.md)

---

## 17. NumPy implementation

The following code reproduces the numerical self-attention example.

```python
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute a numerically stable softmax."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute scaled dot-product attention.

    Parameters
    ----------
    q:
        Query matrix with shape (n_queries, d_k).
    k:
        Key matrix with shape (n_keys, d_k).
    v:
        Value matrix with shape (n_keys, d_v).
    mask:
        Optional additive mask. Use 0 for allowed positions and
        a very negative value for blocked positions.

    Returns
    -------
    output:
        Contextualized representations.
    weights:
        Attention probability matrix.
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, and v must be two-dimensional arrays.")

    if q.shape[1] != k.shape[1]:
        raise ValueError("q and k must have the same feature dimension.")

    if k.shape[0] != v.shape[0]:
        raise ValueError("k and v must contain the same number of tokens.")

    d_k = k.shape[-1]
    scores = q @ k.T / np.sqrt(d_k)

    if mask is not None:
        if mask.shape != scores.shape:
            raise ValueError("mask must have the same shape as the score matrix.")
        scores = scores + mask

    weights = softmax(scores, axis=-1)
    output = weights @ v

    return output, weights


# Artificial vectors for:
# [Ali, book, read]
x = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)

# For teaching purposes:
# W_Q = W_K = W_V = identity
q = x
k = x
v = x

output, weights = scaled_dot_product_attention(q, k, v)

np.set_printoptions(precision=3, suppress=True)

print("Attention weights:")
print(weights)

print("\nContextualized output:")
print(output)
```

Expected output:

```text
Attention weights:
[[0.401 0.198 0.401]
 [0.198 0.401 0.401]
 [0.248 0.248 0.503]]

Contextualized output:
[[0.802 0.599]
 [0.599 0.802]
 [0.752 0.752]]
```

### Causal-mask example

```python
sequence_length = 3

mask = np.triu(
    np.full((sequence_length, sequence_length), -1e9),
    k=1,
)

masked_output, masked_weights = scaled_dot_product_attention(
    q,
    k,
    v,
    mask=mask,
)

print("Causal mask:")
print(mask)

print("\nMasked attention weights:")
print(masked_weights)
```

The first token can attend only to itself. The second token can attend to the first two tokens. The third token can attend to all three.

---
