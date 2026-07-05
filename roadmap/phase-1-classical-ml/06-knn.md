# 1.6 — K-Nearest Neighbors (KNN)

## 1. Intuition first

To classify a new point, look at its **k nearest** training points and take a majority vote (or average, for regression). No training — the "model" is the dataset itself. This is called a **lazy** or **instance-based** learner.

## 2. Why the topic exists

Fundamental non-parametric baseline. Great for teaching distance metrics, curse of dimensionality, and lazy learning. Still very useful for retrieval (semantic search, RAG) — a KNN over embedding space is what a vector DB does.

## 3. What problem it solves

Classification, regression, and retrieval where local structure of data matters and you don't need a compact model.

## 4. Mathematics

Distance metrics:

- **Euclidean (L2)**: $d(\mathbf{x},\mathbf{x}') = \sqrt{\sum_j (x_j - x'_j)^2}$
- **Manhattan (L1)**: $d = \sum_j |x_j - x'_j|$
- **Minkowski (Lp)**: $d = (\sum |x_j - x'_j|^p)^{1/p}$
- **Cosine**: $d = 1 - \frac{\mathbf{x}\cdot\mathbf{x}'}{\|\mathbf{x}\|\|\mathbf{x}'\|}$
- **Mahalanobis**: $d = \sqrt{(\mathbf{x}-\mathbf{x}')^T \Sigma^{-1} (\mathbf{x}-\mathbf{x}')}$

Classification prediction (uniform vote):

$$
\hat y = \arg\max_c \sum_{i \in \mathcal{N}_k(\mathbf{x})} \mathbb{1}[y_i = c]
$$

Regression:

$$
\hat y = \frac{1}{k}\sum_{i \in \mathcal{N}_k(\mathbf{x})} y_i
$$

Weighted version — closer neighbors have more say:

$$
\hat y = \frac{\sum_i \tfrac{1}{d_i^2} y_i}{\sum_i \tfrac{1}{d_i^2}}
$$

## 5. Every formula explained

- Distance functions define "nearness". Choice matters — cosine for text, Euclidean for standardized numeric features, Mahalanobis when features have covariance.
- Uniform-vote KNN is the empirical estimate of $P(y=c \mid \mathbf{x})$ over a local neighborhood.

## 6. Variables

| Symbol | Meaning |
|--------|---------|
| $k$ | number of neighbors |
| $\mathcal{N}_k(\mathbf{x})$ | set of $k$ nearest training points to $\mathbf{x}$ |
| $d_i$ | distance to $i$-th neighbor |

## 7. Algorithm

1. Standardize features.
2. Build a spatial index (KD-tree, Ball-tree, HNSW).
3. For each query:
   1. Retrieve $k$ nearest neighbors.
   2. Aggregate labels (mode / mean, weighted or not).

## 8. Simple example

$k=3$ on Iris: query point is (5.1, 3.5). Nearest 3 flowers are all setosa → classify as setosa.

## 9. Real-world example

- **Semantic search / RAG**: retrieve top-k embeddings similar to a query.
- **Recommenders**: item-item collaborative filtering.
- **Image dedup / near-duplicate detection**.

## 10. Diagram

```
   x2
    |     x  x  x
    |   x    ?     o     <- ? = query; nearest 3 are all "x"
    |  x   x         o o
    |               o
    +---------------------- x1
```

```mermaid
flowchart LR
    Q["Query x"] --> D["Compute distances to all train points"]
    D --> S["Sort / index → k nearest"]
    S --> V["Vote / average"]
    V --> P["Prediction"]
```

## 11. Implementation from scratch

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5, metric="l2"):
        self.k, self.metric = k, metric
    def fit(self, X, y):
        self.X = np.asarray(X, float); self.y = np.asarray(y)
        return self
    def _dist(self, x):
        if self.metric == "l2":
            return np.linalg.norm(self.X - x, axis=1)
        if self.metric == "l1":
            return np.abs(self.X - x).sum(axis=1)
        if self.metric == "cosine":
            a = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)
            b = x / np.linalg.norm(x)
            return 1 - a @ b
        raise ValueError(self.metric)
    def predict(self, X):
        out = []
        for x in np.asarray(X, float):
            idx = np.argpartition(self._dist(x), self.k)[:self.k]
            out.append(Counter(self.y[idx]).most_common(1)[0][0])
        return np.array(out)
```

## 12. Implementation using libraries

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
m = KNeighborsClassifier(n_neighbors=5, weights="distance",
                         algorithm="auto", metric="minkowski", p=2)
m.fit(X_train, y_train)
```

For huge datasets, use **FAISS** (Facebook AI Similarity Search) or **HNSWlib**.

## 13. Time complexity

- Naive query: O(nd) per query — brute-force.
- KD-tree / Ball-tree: O(d log n) for low $d$; degrades to O(n) for $d > 20$ (curse of dimensionality).
- HNSW / IVF-PQ: sub-linear approximate, empirically O(log n).

## 14. Space complexity

O(nd) — you keep the entire training set.

## 15. Advantages

- Zero training time.
- Adapts to arbitrary decision boundaries.
- Great for retrieval-like problems.

## 16. Disadvantages

- Slow at inference.
- Memory-hungry.
- Sensitive to feature scale and irrelevant features.
- Curse of dimensionality.

## 17. Interview questions

1. Why must you standardize features for KNN?
2. Effect of $k$ on bias-variance?
3. Curse of dimensionality — what and why?
4. Explain KD-tree, Ball-tree, LSH.
5. When would you use cosine vs Euclidean?
6. Why is KNN considered non-parametric?
7. How does KNN behave under class imbalance?
8. Weighted vs uniform voting — trade-offs.
9. Time complexity of naive vs indexed KNN.
10. How is KNN related to a vector database?

## 18. Common mistakes

- Not scaling features.
- Choosing $k$ without CV.
- Using KNN on high-dim raw pixels/text without embeddings.
- Storing full training data in production — memory blows up.

## 19. Optimization techniques

- Approximate nearest neighbors (FAISS, ScaNN, HNSW).
- Dimensionality reduction (PCA/UMAP) before KNN.
- Product quantization.
- GPU brute force for medium-scale.

## 20. Coding exercises

1. Add cosine similarity and weighted-distance voting.
2. Implement a KD-tree from scratch.
3. Show curse of dimensionality: pairwise distances concentrate as $d$ grows.
4. Compare exact vs FAISS approximate on 1 M vectors.

## 21. Mini project

Handwritten-digit KNN on MNIST; hyperparameter search over $k$ and metric.

## 22. Medium project

Build an image-similarity search using ResNet features and FAISS on 100k Unsplash images.

## 23. Advanced project

Build a full RAG retrieval pipeline: embed 1 M documents with an encoder, index with HNSW, evaluate recall@k, and add a reranker.

## 24. Where it is used in industry

Recommenders, semantic search, RAG systems, anomaly detection, deduplication.

## 25. How companies use it

- Pinterest visual search (approximate KNN on billions of image embeddings).
- Spotify song similarity.
- Every modern RAG stack (Pinecone, Weaviate, Qdrant, Milvus).

## 26. When NOT to use it

- Tabular classification with millions of rows and low latency requirements.
- Very high-dim raw data without a good embedding.
