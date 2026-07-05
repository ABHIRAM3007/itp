# 3.3 — Seq2Seq & Encoder-Decoder

## 1. Intuition first
Read the input sequence into a compact representation ("encoder"), then generate the output sequence one token at a time ("decoder"). Historically RNN-based; today Transformer-based (T5, BART, mT5, NLLB).

## 2. Why the topic exists
Translation, summarization, question answering, code translation — any task mapping one sequence to another.

## 3. What problem it solves
Learn arbitrary sequence-to-sequence mappings end-to-end.

## 4. Mathematics

### 4.1 Model
Encoder: $H = \text{Enc}(x_{1..T})$. Decoder: $p(y_t \mid y_{<t}, H) = \text{softmax}(\text{Dec}(y_{<t}, H))$.

Loss: sum of per-token cross-entropies with teacher forcing.

### 4.2 Attention (Bahdanau, Luong)
Decoder queries encoder states: $\alpha_{tj} = \text{softmax}(\text{score}(h_t^{dec}, h_j^{enc}))$; context $c_t = \sum_j \alpha_{tj} h_j^{enc}$.

### 4.3 Decoding
- Greedy.
- Beam search (top-$k$ beams by log-prob).
- Sampling (temperature, top-$k$, top-$p$).

### 4.4 Loss variants
Copy mechanisms, coverage loss (to avoid repetition), length normalization.

## 5. Every formula explained
Attention weights sum to 1; context vector is a soft look-up over the encoder. Beam search trades compute for higher-probability outputs.

## 6. Variables
$T, U$ input/output lengths; $H$ encoder outputs; $\alpha_{tj}$ attention weights.

## 7. Algorithm — training + inference
Train with teacher forcing; at inference, feed previous predicted token; use beam search or sampling.

## 8. Simple example
Translate "cats are cute" → "chats sont mignons" with a small encoder-decoder RNN + attention.

## 9. Real-world example
- Google Translate (2016 NMT).
- Amazon Alexa summarization.
- Code translation (Codex, TransCoder).
- Speech-to-text (Whisper is encoder-decoder).

## 10. Diagram
```mermaid
flowchart LR
    X["Source tokens"] --> E["Encoder"] --> H["Encoder outputs"]
    H --> D["Decoder (causal + cross-attn)"]
    D --> Y["Target tokens"]
```

## 11. Implementation from scratch — HF-style with PyTorch
```python
import torch, torch.nn as nn
class Seq2Seq(nn.Module):
    def __init__(self, vocab, d=256, h=4, L=3):
        super().__init__()
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, h, batch_first=True), L)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, h, batch_first=True), L)
        self.tok_emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab)
    def forward(self, src, tgt, tgt_mask):
        s = self.enc(self.tok_emb(src))
        y = self.dec(self.tok_emb(tgt), s, tgt_mask=tgt_mask)
        return self.head(y)
```

## 12. Implementation using libraries
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tok = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
ids = tok("translate English to French: Hello", return_tensors="pt").input_ids
print(tok.decode(model.generate(ids, num_beams=4)[0], skip_special_tokens=True))
```

## 13. Time complexity
Encoder: O(T² d). Decoder generation: O(U (U + T) d) with KV cache reducing to O(U (T + U/2) d).

## 14. Space complexity
Store encoder outputs across all decoder steps.

## 15. Advantages
Handles variable-length input/output; strong for translation/summarization; can incorporate copy mechanisms.

## 16. Disadvantages
Exposure bias (train vs inference distribution mismatch); slow decoding; beam search complexity.

## 17. Interview questions
1. Compare seq2seq with/without attention.
2. Teacher forcing — pros & cons.
3. Beam search vs sampling — quality/latency trade-offs.
4. Length normalization in beam search.
5. Copy mechanism — why?
6. Coverage loss — motivation.
7. Encoder-decoder vs decoder-only (GPT) for translation.
8. Exposure bias — mitigations (scheduled sampling, MRT).
9. Non-autoregressive generation.
10. How does Whisper differ from a text-only seq2seq?

## 18. Common mistakes
- Wrong causal mask on decoder.
- Not shifting labels for teacher forcing.
- Beam search without length penalty → short outputs.

## 19. Optimization techniques
KV caching, prompt caching, distilled decoders, non-autoregressive decoding (Levenshtein Transformer).

## 20. Coding exercises
1. Implement Bahdanau attention.
2. Fine-tune T5-small on a summarization dataset.
3. Add beam search from scratch.
4. Compare greedy / beam / nucleus sampling BLEU.

## 21. Mini project
Fine-tune T5-small on CNN/DailyMail summarization; report ROUGE-L.

## 22. Medium project
Reproduce a mini machine-translation system on IWSLT'14 De→En; achieve reasonable BLEU (>25).

## 23. Advanced project
Non-autoregressive translation baseline + iterative refinement; benchmark against beam-search AR baseline.

## 24. Where it is used in industry
Translation (Google, DeepL, Meta NLLB), summarization, code generation, ASR (Whisper), TTS.

## 25. How companies use it
- Google Translate / DeepL.
- Amazon summarization pipelines.
- GitHub Copilot originally on seq2seq codebases.

## 26. When NOT to use it
- Pure text generation → decoder-only LMs usually simpler and stronger.
- Tasks with hard latency budgets where non-AR alternatives fit better.
