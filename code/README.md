# Language Modeling Evolution — Python Practices

A progressive, repository-ready notebook series accompanying the markdown chapters
on the evolution of language modeling from n-grams to Transformers.

The examples are deliberately small and inspectable. They emphasize tensor shapes,
manual calculations, assertions, visualizations, and Persian-language examples
rather than benchmark performance.

## Notebook map

| Notebook | Main practice |
|---|---|
| `01-ngrams-rnn-lstm-gru.ipynb` | Counts, smoothing, perplexity, recurrent states, tiny GRU LM |
| `02-seq2seq-recurrent-attention.ipynb` | Fixed-vector bottleneck, Bahdanau and Luong attention |
| `03-self-attention-multihead.ipynb` | Q/K/V, scaled attention, scaling experiment, multi-head attention |
| `04-transformer-encoder.ipynb` | Embeddings, positions, residuals, LayerNorm, FFN, encoder stack |
| `05-transformer-decoder-training.ipynb` | Shifted targets, causal masks, cross-attention, cross-entropy |
| `06-strengths-limitations-complexity.ipynb` | Path length, quadratic score memory, sequential generation |
| `07-numpy-attention-tests.ipynb` | Robust NumPy implementation, tests, masking, multi-head extension |
| `08-practice-exercises.ipynb` | Ten exercises with executable checks |
| `09-capstone-tiny-transformer-lm.ipynb` | End-to-end tiny causal Transformer language model |

## Recommended order

Run the notebooks in numerical order. Notebooks are self-contained, so readers can
also open a single topic without importing code from earlier notebooks.

## Installation

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux/macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter lab
```

## GPU support

Every PyTorch notebook automatically selects CUDA when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The examples are intentionally small and also run on CPU. For an NVIDIA GPU,
install the PyTorch build matching your CUDA environment using the official
PyTorch installation instructions.


## Teaching design

Each notebook follows the same pattern:

1. connect a formula to code;
2. inspect intermediate arrays or tensors;
3. verify shapes and probability sums;
4. visualize a central operation where useful;
5. end with modification-based practice prompts.


## Scope and limitations

These notebooks are educational. The corpora are tiny, tokenization is mostly
whitespace-based, and trained outputs should not be interpreted as robust Persian
language models. A research implementation needs documented preprocessing,
train/validation/test splits, stronger tokenization, reproducible evaluation, and
ethical/data-governance notes.
