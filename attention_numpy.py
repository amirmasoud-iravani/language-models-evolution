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
    """Compute scaled dot-product attention."""
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


x = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])

q = x
k = x
v = x

output, weights = scaled_dot_product_attention(q, k, v)

np.set_printoptions(precision=3, suppress=True)
print("Attention weights:")
print(weights)
print("\nContextualized output:")
print(output)
