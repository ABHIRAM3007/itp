# 3.4 — BERT-family (Encoder-only Transformers)

## 1. Intuition first
Instead of predicting the next word, mask random words and predict them from left and right context. That gives the model bidirectional understanding of language, perfect for classification and retrieval tasks.

## 2. Why the topic exists
Devlin et al. (2018) showed that a pretrained masked language model, fine-tuned on downstream tasks with a small classification head, set state-of-the-art across 11 NLP benchmarks.

## 3. What problem it solves
Universal sentence/token encoder for classification, retrieval, NER, QA, and more.

## 4. Mathematics

### 4.1 Masked Language Model (MLM)
Randomly select 15% of tokens; replace 80% with `[MASK]`, 10% with random token, 10% unchanged. Loss = CE on masked positions only.

### 4.2 Next Sentence Prediction (NSP)
Original BERT used NSP; later shown unnecessary (RoBERTa drops it).

### 4.3 Architecture
Standard Transformer encoder (12 or 24 layers), bidirectional self-attention, `[CLS]` token for sentence-level tasks.

### 4.4 Fine-tuning
Add task head on `[CLS]` (classification) or per-token (NER); train full model with small LR ($2\!\times\!10^{-5}$).

### 4.5 Variants
- **RoBERTa**: more data, no NSP, longer training.
- **DistilBERT**: distilled 6-layer.
- **ALBERT**: parameter sharing.
- **DeBERTa**: disentangled attention + relative positions.
- **ELECTRA**: replaced-token detection (much more compute-efficient).
- **Multilingual mBERT / XLM-R**.

## 5. Every formula explained
MLM loss $L = -\sum_{i\in\text{masked}} \log p(y_i \mid \tilde x)$. NSP is binary CE over sentence-pair labels.

## 6. Variables
`[CLS]`, `[SEP]`, `[MASK]` tokens; $L$ layers; $d$ hidden.

## 7. Algorithm — fine-tune for classification
1. Tokenize `[CLS] text [SEP]`.
2. Forward through pretrained BERT.
3. Take pooled output at `[CLS]` (or mean-pool).
4. Linear + softmax → task classes.
5. Fine-tune with small LR + warmup.

## 8. Simple example
Sentiment classification on IMDB: fine-tune BERT-base → ~93% accuracy in one epoch.

## 9. Real-world example
- Google Search uses BERT variants to understand queries.
- Support ticket routing at Zendesk / Intercom.
- Compliance / legal document classification.

## 10. Diagram
```
   [CLS] The movie was great . [SEP]
     │      │    │    │   │      │
   ┌───────────────────────────┐
   │  Bidirectional Transformer│
   └───────────────────────────┘
     │
   pooled [CLS] → linear → softmax → positive/negative
```

## 11. Implementation from scratch (mini)
Same as Transformer encoder from 2.10 with MLM head instead of LM head.

## 12. Implementation using libraries
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
mdl = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
outputs = mdl(**enc, labels=labels)
```

Sentence embeddings:
```python
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("all-MiniLM-L6-v2")
vecs = st.encode(["hi", "hello"])
```

## 13. Time complexity
Same as Transformer encoder: O(L n² d). Fine-tuning cheap: minutes on a single GPU.

## 14. Space complexity
110M – 340M parameters for base/large.

## 15. Advantages
Great zero-shot & fine-tuned performance; small and cheap to serve; strong retrieval encoders (SBERT).

## 16. Disadvantages
Not generative; fixed context (usually ≤ 512 tokens without long-variants).

## 17. Interview questions
1. Explain MLM vs Causal LM.
2. Why 15% masking?
3. Why 80/10/10 split for masked tokens?
4. NSP — why did RoBERTa drop it?
5. Difference between BERT-base and BERT-large.
6. Explain sentence-transformers / SBERT.
7. RoBERTa's changes over BERT.
8. What is ELECTRA's replaced token detection?
9. Fine-tuning best practices (LR, warmup, epochs).
10. Encoder vs decoder LMs — when to use which?

## 18. Common mistakes
- Using original BERT for generation (it's not autoregressive).
- Not adding `[CLS]`/`[SEP]` or wrong `token_type_ids`.
- Learning rate too high → catastrophic forgetting.

## 19. Optimization techniques
Distillation (DistilBERT, MiniLM); adapters / LoRA; dynamic padding; ONNX runtime; INT8 quantization for CPU serving.

## 20. Coding exercises
1. Fine-tune DistilBERT on SST-2.
2. Build a BERT-based NER model on CoNLL-2003.
3. Train a re-ranker on MS MARCO with cross-encoder.
4. Compare mean vs `[CLS]` pooling for embeddings.

## 21. Mini project
Toxicity classifier on Jigsaw Toxic Comments.

## 22. Medium project
End-to-end semantic search with SBERT + FAISS on 100k documents.

## 23. Advanced project
Train a MiniLM student from a fine-tuned BERT teacher; deploy INT8 quantized on CPU with FastAPI.

## 24. Where it is used in industry
Search ranking, retrieval, NER, classification, moderation, RAG re-ranker.

## 25. How companies use it
- Google Search (BERT for queries).
- Cross-encoder rerankers (bge-reranker, Cohere Rerank).
- Content moderation classifiers.

## 26. When NOT to use it
- Long-form generation → decoder LLM.
- Very long documents without a long-context variant.
