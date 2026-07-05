# 3.2 — Word Embeddings (Word2Vec, GloVe, fastText)

## 1. Intuition first
Represent each word as a dense vector such that similar words are close in vector space and semantic relations map to vector arithmetic: `king − man + woman ≈ queen`.

## 2. Why the topic exists
One-hot vectors are huge and orthogonal — carry no meaning. Dense embeddings learned from co-occurrence carry semantics.

## 3. What problem it solves
Feature representation for words; transfer learning before contextual embeddings; still used as fast lookup features.

## 4. Mathematics

### 4.1 Word2Vec — Skip-gram with Negative Sampling (Mikolov 2013)
For center word $w$ and context $c$:

$$
P(c \mid w) = \frac{e^{v_c^T v_w}}{\sum_{c'} e^{v_{c'}^T v_w}}
$$

Full softmax expensive. Approximate with **negative sampling**:

$$
\log \sigma(v_c^T v_w) + \sum_{k=1}^K \mathbb{E}_{c_k \sim P_n}[\log \sigma(-v_{c_k}^T v_w)]
$$

Two variants:
- **Skip-gram**: predict context given center.
- **CBOW**: predict center given context.

### 4.2 GloVe (Pennington 2014)
Factorize log co-occurrence matrix:

$$
J = \sum_{i,j} f(X_{ij}) (v_i^T v_j + b_i + b_j - \log X_{ij})^2
$$

with weighting $f$ that damps very frequent pairs.

### 4.3 fastText (Bojanowski 2016)
Represent each word as a sum of subword n-gram embeddings → handles OOVs.

### 4.4 Vector analogy
$v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$.

## 5. Every formula explained
- Skip-gram maximizes probability that observed context tokens are picked over random noise (contrastive).
- GloVe directly targets ratios of co-occurrence probabilities.
- fastText compositional sub-word gives generalization to unseen words.

## 6. Variables
$v_w$ embedding of word $w$; $X_{ij}$ co-occurrence count; $K$ negative samples per positive.

## 7. Algorithm — Skip-gram + NS
1. Slide a window over corpus; enumerate (center, context) pairs.
2. Sample $K$ negatives per positive (unigram^0.75 distribution).
3. SGD to maximize the log-likelihood above.

## 8. Simple example
Train on 100 MB of Wikipedia; nearest neighbors of "python" include "ruby", "perl", "javascript".

## 9. Real-world example
- Pre-2018 NLP pipelines used GloVe/word2vec initialization.
- Recommendation: item2vec, prod2vec — same skip-gram trick on non-text sequences.
- Ad-tech: user embeddings from co-viewed products.

## 10. Diagram
```
  center word ─► lookup ─► vw
                                 dot product with vc → sigmoid → predict positive
  neg samples ─► lookups ─► vcˊ  dot with vw → sigmoid → predict negative
```

## 11. Implementation from scratch — skip-gram with negative sampling
```python
import numpy as np, random
class SkipGramNS:
    def __init__(self, V, d=100, lr=0.01, K=5, seed=0):
        rng = np.random.default_rng(seed)
        self.W_in = rng.normal(scale=0.01, size=(V, d))
        self.W_out = rng.normal(scale=0.01, size=(V, d))
        self.lr = lr; self.K = K
        self.V = V
    def train_pair(self, w, c, neg_sampler):
        vw = self.W_in[w]
        # positive
        vc = self.W_out[c]; z = vc @ vw
        g = 1 / (1 + np.exp(-z)) - 1
        self.W_out[c] -= self.lr * g * vw
        self.W_in[w] -= self.lr * g * vc
        # negatives
        for _ in range(self.K):
            cn = neg_sampler(); vcn = self.W_out[cn]
            z = vcn @ vw
            g = 1 / (1 + np.exp(-z))
            self.W_out[cn] -= self.lr * g * vw
            self.W_in[w]  -= self.lr * g * vcn
```

## 12. Implementation using libraries
```python
from gensim.models import Word2Vec, FastText
model = Word2Vec(sentences=sents, vector_size=200, window=5, sg=1, negative=10, workers=8)
model.wv.most_similar("python")
```

## 13. Time complexity
O(#tokens × window × (1 + K) × d).

## 14. Space complexity
O(|V| × d).

## 15. Advantages
Small, fast to lookup, capture semantic similarity, useful in recommenders.

## 16. Disadvantages
Static (one vector per word ignoring context); polysemy problems.

## 17. Interview questions
1. Skip-gram vs CBOW.
2. Why negative sampling? What is the noise distribution?
3. GloVe vs Word2Vec — differences.
4. Explain the analogy property mathematically.
5. Why does fastText handle OOV words?
6. Cosine similarity vs Euclidean for embeddings.
7. Compare static vs contextual embeddings (BERT).
8. How would you use word embeddings in a recommender?
9. Curse of high dimensionality in embedding space.
10. Explain hierarchical softmax.

## 18. Common mistakes
- Not lowercasing / normalizing.
- Random init vs pretrained — use pretrained for small data.
- Mismatch between tokenizer at training and use.

## 19. Optimization techniques
Subsampling frequent words; hierarchical softmax; noise contrastive estimation; product quantization for storage.

## 20. Coding exercises
1. Train Word2Vec on a Wikipedia dump.
2. Reproduce the king-man+woman analogy.
3. Train item2vec on MovieLens sessions.
4. Visualize embeddings with UMAP/t-SNE.

## 21. Mini project
Semantic search over an FAQ dataset using averaged Word2Vec vectors.

## 22. Medium project
prod2vec for e-commerce recommendations; evaluate NDCG on held-out sessions.

## 23. Advanced project
Compare Word2Vec, GloVe, fastText, and mean-pooled BERT embeddings on downstream STS-B and analogy benchmarks.

## 24. Where it is used in industry
Legacy NLP pipelines; recommender systems; fast semantic retrieval before neural re-ranker.

## 25. How companies use it
- Airbnb listing embeddings.
- Spotify song2vec.
- Twitter user embeddings for follow suggestions.

## 26. When NOT to use it
- Whenever context-dependent meaning matters — use BERT/LLM embeddings.
- Very small vocab.
