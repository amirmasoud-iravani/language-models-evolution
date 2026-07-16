# Part 8 — Practice exercises

[← Main README](../README.md) · [← Previous](./07-numpy-implementation.md) · [Next →](./09-key-takeaways-and-references.md)

---

## 18. Practice exercises

### Exercise 1: softmax by hand

Given attention scores:

$$
[1,3,0]
$$

calculate:

$$
\mathrm{softmax}([1,3,0])
$$

Use:

$$
e^1\approx2.718
$$

$$
e^3\approx20.086
$$

$$
e^0=1
$$

Expected result:

$$
[0.114,0.844,0.042]
$$

### Exercise 2: weighted context vector

Given:

$$
\alpha=[0.2,0.7,0.1]
$$

and:

$$
v_1=[1,0]
$$

$$
v_2=[0,2]
$$

$$
v_3=[1,1]
$$

calculate:

$$
c=\sum_i\alpha_iv_i
$$

Solution:

$$
c
=
0.2[1,0]
+
0.7[0,2]
+
0.1[1,1]
$$

$$
c=[0.3,1.5]
$$

### Exercise 3: attention and Persian word order

Consider:

> علی دیروز کتاب را خواند

For the token **خواند**, discuss which tokens may be useful for representing:

- the subject;
- the object;
- temporal information.

A possible answer:

- **علی** provides subject information;
- **کتاب** and **را** provide object information;
- **دیروز** provides temporal information.

A trained model does not receive these labels directly. It learns useful interaction patterns from data and the training objective.

### Exercise 4: causal masking

For a sequence of length $4$, construct an additive causal mask using:

- $0$ for visible positions;
- $-\infty$ for future positions.

Solution:

$$
\begin{bmatrix}
0&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty\\
0&0&0&-\infty\\
0&0&0&0
\end{bmatrix}
$$

### Exercise 5: compare computational paths

For a dependency between token $1$ and token $20$:

- how many recurrent transitions may separate them in an RNN?
- how many self-attention interactions are needed in one Transformer layer?

Conceptual answer:

- an RNN may require information to pass through approximately $19$ recurrent transitions;
- self-attention can connect the two positions directly in one layer.

### Exercise 6: implement learned projections

Replace the identity projections in the NumPy example with small matrices:

```python
w_q = np.array([
    [1.0, 0.0],
    [0.5, 1.0],
])

w_k = np.array([
    [0.5, 1.0],
    [1.0, 0.0],
])

w_v = np.array([
    [1.0, 1.0],
    [0.0, 1.0],
])

q = x @ w_q
k = x @ w_k
v = x @ w_v
```

Then recompute the attention weights and output.

Observe that changing the projections changes both:

- which tokens match;
- what information is retrieved.

---
