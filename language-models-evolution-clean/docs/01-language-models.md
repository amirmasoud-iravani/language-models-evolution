# Part 1 — From n-grams to gated RNNs

[← Main README](../README.md) · [Next →](./02-seq2seq-and-recurrent-attention.md)

---

## 1. The language-modeling problem

A language model assigns a probability to a sequence of tokens:

$$
P(w_1,w_2,\ldots,w_T)
$$

Using the probability chain rule:

$$
P(w_1,w_2,\ldots,w_T)
=
\prod_{t=1}^{T}
P(w_t \mid w_1,\ldots,w_{t-1})
$$

For the sentence:

> I like Persian poetry

the probability is decomposed as:

$$
P(\text{I})
P(\text{like}\mid\text{I})
P(\text{Persian}\mid\text{I like})
P(\text{poetry}\mid\text{I like Persian})
$$

The central NLP problem is therefore:

> How should a model represent the previous context when predicting the next token?

Different generations of language models answer this question differently.

---

---

## 2. Statistical n-gram models

An n-gram model uses only a fixed number of previous tokens.

For a trigram model:

$$
P(w_t\mid w_1,\ldots,w_{t-1})
\approx
P(w_t\mid w_{t-2},w_{t-1})
$$

The probability can be estimated from corpus counts:

$$
P(w_t\mid w_{t-2},w_{t-1})
=
\frac{C(w_{t-2},w_{t-1},w_t)}
{C(w_{t-2},w_{t-1})}
$$

### Example

Suppose a corpus contains:

$$
C(\text{I like})=40
$$

and:

$$
C(\text{I like tea})=30
$$

Then:

$$
P(\text{tea}\mid\text{I like})
=
\frac{30}{40}
=
0.75
$$

So the trigram model assigns probability $0.75$ to *tea* after *I like*.

### Main limitations

#### 1. Fixed context

A trigram model sees only two previous tokens.

Consider:

> The book that I bought yesterday was expensive.

To predict *was*, the subject *book* is several tokens away. A trigram model cannot directly use that dependency.

#### 2. Data sparsity

If an n-gram never appears in the training corpus, its raw count is zero.

For example:

$$
C(\text{I enjoy saffron tea})=0
$$

does not mean the phrase is impossible. It may simply be absent from the corpus.

Smoothing methods reduce this problem, but they do not eliminate the fixed-context limitation.

#### 3. Weak semantic sharing

These sentences are semantically similar:

- I like tea.
- I enjoy coffee.

A symbolic n-gram model does not naturally understand that *like* and *enjoy* are related or that *tea* and *coffee* are related.

---

---

## 3. Recurrent neural networks

A recurrent neural network processes a sequence one token at a time.

At time step $t$, it computes:

$$
h_t
=
\tanh(W_xx_t+W_hh_{t-1}+b_h)
$$

where:

- $x_t$ is the current token representation;
- $h_{t-1}$ is the previous hidden state;
- $h_t$ is the new hidden state.

The next-token distribution is:

$$
P(w_{t+1}\mid w_{\leq t})
=
\mathrm{softmax}(W_oh_t+b_o)
$$

The hidden state is intended to summarize the sequence seen so far.

```mermaid
flowchart LR
    X1[x₁] --> H1[h₁]
    H1 --> H2[h₂]
    X2[x₂] --> H2
    H2 --> H3[h₃]
    X3[x₃] --> H3
    H3 --> O[Next-token prediction]
```

### A tiny scalar example

Assume a one-dimensional RNN:

$$
h_t=\tanh(x_t+0.5h_{t-1})
$$

Let:

$$
h_0=0
$$

and input values:

$$
x_1=1,\qquad x_2=0.5
$$

Then:

$$
h_1=\tanh(1+0.5(0))=\tanh(1)\approx0.762
$$

Next:

$$
h_2=\tanh(0.5+0.5(0.762))
$$

$$
h_2=\tanh(0.881)\approx0.707
$$

The second state contains information from both $x_2$ and the previous state.

### Why ordinary RNNs struggle with long sequences

During backpropagation, gradients pass through many recurrent transitions:

$$
\frac{\partial h_t}{\partial h_{t-k}}
=
\prod_{j=t-k+1}^{t}
\frac{\partial h_j}{\partial h_{j-1}}
$$

Suppose each local derivative is approximately $0.5$. After ten steps:

$$
0.5^{10}\approx0.00098
$$

The training signal becomes extremely small. This is the **vanishing-gradient problem**.

If the repeated derivative is larger than $1$, the gradient may instead become extremely large. This is the **exploding-gradient problem**.

RNNs also remain sequential:

$$
h_1\rightarrow h_2\rightarrow h_3\rightarrow\cdots\rightarrow h_n
$$

The model cannot compute $h_{10}$ before computing $h_1,\ldots,h_9$.

---

---

## 4. LSTM and GRU

LSTM and GRU architectures introduce gates that control the flow of information.

### 4.1 LSTM

An LSTM has a hidden state $h_t$ and a cell state $c_t$.

#### Forget gate

$$
f_t
=
\sigma(W_f[x_t;h_{t-1}]+b_f)
$$

The forget gate decides how much old memory to preserve.

#### Input gate

$$
i_t
=
\sigma(W_i[x_t;h_{t-1}]+b_i)
$$

#### Candidate memory

$$
\tilde c_t
=
\tanh(W_c[x_t;h_{t-1}]+b_c)
$$

#### Cell-state update

$$
c_t
=
f_t\odot c_{t-1}
+
i_t\odot\tilde c_t
$$

#### Output gate

$$
o_t
=
\sigma(W_o[x_t;h_{t-1}]+b_o)
$$

#### Hidden state

$$
h_t=o_t\odot\tanh(c_t)
$$

Here, $\odot$ means element-wise multiplication.

### Scalar LSTM example

Suppose:

$$
c_{t-1}=0.8
$$

$$
f_t=0.9,\qquad i_t=0.3,\qquad \tilde c_t=0.5
$$

Then:

$$
c_t
=
0.9(0.8)+0.3(0.5)
$$

$$
c_t=0.72+0.15=0.87
$$

If:

$$
o_t=0.7
$$

then:

$$
h_t
=
0.7\tanh(0.87)
\approx
0.7(0.701)
\approx
0.491
$$

The model preserves most of the old memory and adds a smaller amount of new information.

### 4.2 GRU

A GRU is a simpler gated recurrent model.

#### Update gate

$$
z_t
=
\sigma(W_zx_t+U_zh_{t-1}+b_z)
$$

#### Reset gate

$$
r_t
=
\sigma(W_rx_t+U_rh_{t-1}+b_r)
$$

#### Candidate state

$$
\tilde h_t
=
\tanh(W_hx_t+U_h(r_t\odot h_{t-1})+b_h)
$$

#### Final state

One common convention is:

$$
h_t
=
(1-z_t)\odot h_{t-1}
+
z_t\odot\tilde h_t
$$

### Scalar GRU example

Suppose:

$$
h_{t-1}=0.6,\qquad
z_t=0.25,\qquad
\tilde h_t=0.2
$$

Then:

$$
h_t
=
0.75(0.6)+0.25(0.2)
$$

$$
h_t=0.45+0.05=0.50
$$

The result remains mostly based on the old state.

### What LSTMs and GRUs solved

They improved:

- gradient flow;
- long-term memory;
- control over forgetting and updating.

### What they did not solve

They still process tokens sequentially:

$$
h_1\rightarrow h_2\rightarrow\cdots\rightarrow h_n
$$

Long-distance information still passes through many recurrent steps.

---
