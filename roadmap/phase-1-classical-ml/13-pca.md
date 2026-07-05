# 1.13 — Principal Component Analysis (PCA)

## 1. Intuition first

Rotate the axes so the first axis captures the most variance in the data, the second captures the next-most (and is orthogonal to the first), and so on. Keep only the first $k$ axes → you compressed the data with minimal information loss.

## 2. Why the topic exists

- Reduce dimensionality (compression, visualization).
- Remove correlation between features.
- Denoise (small components often just noise).
- Speed up downstream models.

## 3. What problem it solves

Unsupervised linear dimensionality reduction & decorrelation.

## 4. Mathematics

Given centered data matrix $X \in \mathbb{R}^{n \times d}$ (mean-subtracted). The covariance is

$$
\Sigma = \tfrac{1}{n-1} X^T X \in \mathbb{R}^{d \times d}
$$

Eigen decomposition: $\Sigma \mathbf{v}_j = \lambda_j \mathbf{v}_j$. Sort $\lambda_1 \geq \lambda_2 \geq \ldots \geq 0$. The top-$k$ eigenvectors form $V_k \in \mathbb{R}^{d \times k}$; project:

$$
Z = X V_k \in \mathbb{R}^{n \times k}
$$

Fraction of variance explained by top-$k$:

$$
\text{ExplainedVar}(k) = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j}
$$

**Equivalence with SVD**: if $X = U \Sigma V^T$ is the SVD, principal components are columns of $V$; $\lambda_j = \sigma_j^2 / (n - 1)$.

## 5. Every formula explained

- $\Sigma$ is $d \times d$ symmetric positive semi-definite.
- Eigenvalues = variances along principal axes.
- Truncating to top-$k$ minimizes reconstruction error $\|X - Z V_k^T\|_F^2$ over all rank-$k$ projections.

## 6. Variables

| Symbol | Meaning |
|--------|---------|
| $X$ | centered data matrix |
| $\Sigma$ | covariance matrix |
| $\mathbf{v}_j, \lambda_j$ | $j$-th eigenvector/value |
| $V_k$ | top-$k$ eigenvectors |
| $Z$ | projected data |

## 7. Algorithm

1. Mean-center $X$ (and often standardize each feature).
2. Compute SVD $X = U \Sigma V^T$ (preferred numerically over covariance eigendecomp).
3. Keep first $k$ columns of $V$ as $V_k$.
4. Project: $Z = X V_k$.
5. Reconstruct (optional): $\tilde X = Z V_k^T + \bar x$.

## 8. Simple example

3D points shaped like a pancake → top 2 PCs align with the pancake plane, third axis is noise.

## 9. Real-world example

- Eigenfaces (face recognition via PCA on face images).
- Genomics: reduce 20 000 SNP dimensions to 20 for population-structure plots.
- Whitening features before feeding into a downstream model.

## 10. Diagram

```
Original (correlated)   After PCA (aligned to variance axes)
     x2                        z2
      |    /                    |
      |   / o                   |
      |  o                      |    (small variance)
      | o                       |
      |o                        |
      +---------- x1            +---------- z1 (largest variance)
```

## 11. Implementation from scratch

```python
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.k = n_components
    def fit(self, X):
        X = np.asarray(X, float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:self.k]                 # (k, d)
        self.singular_values_ = S[:self.k]
        var = (S ** 2) / (len(X) - 1)
        self.explained_variance_ratio_ = var[:self.k] / var.sum()
        return self
    def transform(self, X):
        return (np.asarray(X, float) - self.mean_) @ self.components_.T
    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_
```

## 12. Implementation using libraries

```python
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
p = PCA(n_components=0.95, svd_solver="auto").fit(X)     # keep 95% variance
print(p.explained_variance_ratio_.cumsum())
```

## 13. Time complexity

- Full SVD: O(min(nd², n²d)).
- Randomized SVD (`svd_solver="randomized"`): O(nd log k).

## 14. Space complexity

O(nd) or O(min(n,d)²) for covariance.

## 15. Advantages

- Simple, fast, deterministic.
- Great for visualization and preprocessing.
- Optimal linear reconstruction.

## 16. Disadvantages

- Purely linear — cannot capture non-linear manifolds.
- Sensitive to scaling.
- Components can be hard to interpret.

## 17. Interview questions

1. Derive PCA from maximizing projected variance.
2. Prove PCA yields the optimal rank-$k$ linear reconstruction.
3. PCA vs SVD — relationship.
4. Why standardize features before PCA?
5. How to choose the number of components?
6. Kernel PCA — intuition.
7. Difference between PCA, ICA, and NMF.
8. When PCA is unhelpful?
9. Explain whitening.
10. Difference between PCA and Autoencoders.

## 18. Common mistakes

- Forgetting to mean-center.
- Ignoring scaling — dominant-scale features dominate.
- Overfitting to noise components.

## 19. Optimization techniques

- Randomized SVD for large data.
- Incremental PCA for streaming.
- Sparse PCA for interpretable loadings.

## 20. Coding exercises

1. Prove PCA = SVD in code by comparing components.
2. Compute cumulative explained-variance curve.
3. Reconstruct a face from top-$k$ eigenfaces; plot MSE vs $k$.
4. Compare PCA and t-SNE / UMAP visualization on MNIST.

## 21. Mini project

Eigenfaces on Yale/LFW; use as features for a KNN classifier.

## 22. Medium project

PCA + logistic regression pipeline for high-dim genomics data; compare with regularized-only logistic.

## 23. Advanced project

Implement Kernel PCA from scratch; visualize the "swiss roll" unrolled.

## 24. Where it is used in industry

Preprocessing, visualization, feature compression, drift detection (compare covariance across time).

## 25. How companies use it

- Ad-tech: dimension reduction of huge sparse features.
- Bioinformatics: PC plots ubiquitous in genomics.
- Recommenders: latent-factor visualization.

## 26. When NOT to use it

- Non-linear manifolds — use UMAP / t-SNE / autoencoders.
- When interpretability of original features matters and PCA loads mix everything.
