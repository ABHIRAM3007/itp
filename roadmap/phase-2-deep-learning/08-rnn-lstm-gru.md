# 2.8 — RNN, LSTM, GRU

## 1. Intuition first
An RNN reads a sequence one token at a time, updating an internal hidden state as it goes — like reading a book while keeping running thoughts. LSTMs and GRUs add gates to control what to remember and forget, alleviating vanishing gradients.

## 2. Why the topic exists
Sequence data (text, audio, time series) has variable length and order matters. Before Transformers, RNNs (especially LSTMs) dominated NLP for a decade.

## 3. What problem it solves
Model sequential data of arbitrary length; handle temporal dependencies.

## 4. Mathematics

### 4.1 Vanilla RNN
$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$; $y_t = W_{hy} h_t + b_y$.

Backprop through time (BPTT) unrolls over $T$ steps.

Vanishing/exploding gradient because $\partial h_t / \partial h_{t-1}$ involves repeated multiplication by $W_{hh}$ times $\tanh'$.

### 4.2 LSTM (Hochreiter & Schmidhuber, 1997)
Gates: input $i_t$, forget $f_t$, output $o_t$, candidate $\tilde c_t$.

$$
i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
$$
$$
f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
$$
$$
o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
$$
$$
\tilde c_t = \tanh(W_c [h_{t-1}, x_t] + b_c)
$$
$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t
$$
$$
h_t = o_t \odot \tanh(c_t)
$$

Cell state $c_t$ has additive updates → gradient can flow across many steps.

### 4.3 GRU (Cho, 2014)
Simpler: reset $r_t$ and update $z_t$ gates.
$r_t = \sigma(W_r [h_{t-1}, x_t])$
$z_t = \sigma(W_z [h_{t-1}, x_t])$
$\tilde h_t = \tanh(W_h [r_t \odot h_{t-1}, x_t])$
$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t$

### 4.4 Bidirectional
Run one RNN forward and one backward; concatenate hidden states.

### 4.5 Encoder–decoder / seq2seq
Encoder RNN compresses input → context vector; decoder RNN generates output.

## 5. Every formula explained
- Sigmoid gates output values in (0, 1): 0 = forget, 1 = keep.
- Cell state is the "highway" that preserves information across time.
- Additive update (rather than repeated multiplication) is why LSTMs beat vanilla RNNs at long-range tasks.

## 6. Variables
$x_t$ input at time $t$; $h_t$ hidden state; $c_t$ cell state (LSTM); $W_*, b_*$ learnable params; $\odot$ elementwise multiply.

## 7. Algorithm — training an RNN
1. Concatenate token embeddings + previous hidden.
2. Compute gates and new hidden.
3. Predict output at each step.
4. Compute per-step loss; sum; backprop through time.
5. Clip gradients.

## 8. Simple example
Character-level LM on "hello": input `h e l l`, target `e l l o`. Trained RNN memorizes the sequence.

## 9. Real-world example
- Google's original neural machine translation (GNMT, 2016) was LSTM-based.
- Speech recognition (DeepSpeech).
- Time-series forecasting (traffic, energy).

## 10. Diagram
```
   x1 → [RNN cell] → h1 → [RNN cell] → h2 → [RNN cell] → h3
                     ↑                   ↑                   ↑
                     y1                  y2                  y3
```

LSTM cell:
```
c_{t-1} ─────────× (f_t)─── + (i_t · ĉ_t) ─── c_t
                                                 │
                                                tanh
                                                 │
h_{t-1}, x_t → gates i, f, o, ĉ                  × (o_t) → h_t
```

## 11. Implementation from scratch — vanilla RNN forward
```python
import numpy as np
def rnn_forward(x, h0, W_xh, W_hh, W_hy, b_h, b_y):    # x: (T, D)
    T, _ = x.shape
    h = h0; H = []; Y = []
    for t in range(T):
        h = np.tanh(x[t] @ W_xh + h @ W_hh + b_h)
        y = h @ W_hy + b_y
        H.append(h); Y.append(y)
    return np.stack(H), np.stack(Y)
```

## 12. Implementation using libraries
```python
import torch.nn as nn
rnn = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
out, (h, c) = rnn(x)   # x: (B, T, 128)
```

## 13. Time complexity
O(T × H²) per layer (recurrent matmul dominates).

## 14. Space complexity
O(T × H) for stored activations (needed for BPTT).

## 15. Advantages
Handles variable-length sequences; small memory vs sequence length compared to naive attention; strong at short-range.

## 16. Disadvantages
Sequential (hard to parallelize on GPU); vanishing gradients over long ranges; largely superseded by Transformers for NLP.

## 17. Interview questions
1. Why do vanilla RNNs suffer vanishing gradients?
2. Explain LSTM gates.
3. LSTM vs GRU — differences.
4. What is BPTT? Truncated BPTT?
5. Why bidirectional RNNs, and when can we not use them?
6. Explain teacher forcing.
7. Seq2seq with and without attention.
8. Compare RNN vs Transformer for long sequences.
9. Layer-norm variants for RNN.
10. Where are RNNs still preferred over Transformers?

## 18. Common mistakes
- Not clipping gradients (LSTM training explodes).
- Forgetting to pack padded sequences (`pack_padded_sequence`).
- Wrong bidirection at inference for autoregressive tasks.

## 19. Optimization techniques
Truncated BPTT; layer normalization inside cells; peephole LSTMs; QRNN / SRU for speed; CuDNN LSTM kernels.

## 20. Coding exercises
1. Implement char-level RNN from scratch (Karpathy's `min-char-rnn`).
2. Implement LSTM cell manually; train on sequence-copy task.
3. Compare RNN, LSTM, GRU on a memory-length benchmark (adding problem).
4. Implement teacher forcing + scheduled sampling.

## 21. Mini project
Char-level LSTM on Shakespeare that generates coherent-ish text.

## 22. Medium project
LSTM encoder–decoder with attention for English → French on a small subset of WMT.

## 23. Advanced project
Reproduce a WaveNet-style dilated convolution baseline vs stacked LSTM for audio; measure MOS.

## 24. Where it is used in industry
Streaming ASR, time-series forecasting, low-latency sequence models on-device (mobile), some tabular sequence problems.

## 25. How companies use it
- Speech-to-text pipelines still use RNN-T or CTC + LSTM.
- Fintech: LSTM for irregular time-series risk modeling.

## 26. When NOT to use it
- Long-context NLP tasks (use Transformers).
- Highly parallel GPU training (RNNs waste hardware).
