# 1.11 — Support Vector Machines (SVM)

## 1. Intuition first

Find the hyperplane that separates two classes with the **largest possible margin**. Only the closest points ("support vectors") determine the boundary. Non-linear? Project into a higher-dimensional space via a **kernel**, then find a linear boundary there.

## 2. Why the topic exists

Pre-deep-learning era's most powerful classifier. Elegant theory (max-margin, convex QP). Still excellent on small-to-medium datasets and for one-class / anomaly detection.

## 3. What problem it solves

Classification, regression (SVR), one-class outlier detection, ranking.

## 4. Mathematics

### 4.1 Hard-margin linear SVM

Given $\{(\mathbf{x}_i, y_i)\}$, $y_i \in \{-1, +1\}$, linearly separable:

$$
\min_{\mathbf{w}, b} \tfrac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.}\ y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1\ \forall i
$$

Geometric margin = $2 / \|\mathbf{w}\|$.

### 4.2 Soft-margin

Allow slack $\xi_i \geq 0$:

$$
\min_{\mathbf{w}, b, \xi} \tfrac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i \quad \text{s.t.}\ y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i
$$

$C$ trades margin width vs classification errors.

### 4.3 Hinge loss form

$$
J(\mathbf{w}) = \tfrac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))
$$

### 4.4 Dual & kernel trick

Lagrangian dual leads to:

$$
\max_\alpha \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \langle \mathbf{x}_i, \mathbf{x}_j\rangle
$$

subject to $0 \leq \alpha_i \leq C$ and $\sum \alpha_i y_i = 0$. Prediction:

$$
f(\mathbf{x}) = \text{sign}\Big(\sum_i \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\Big)
$$

Replace inner product by a **kernel** $K$:

- Linear: $K = \mathbf{x}^T\mathbf{x}'$.
- Polynomial: $K = (\gamma \mathbf{x}^T\mathbf{x}' + r)^p$.
- RBF (Gaussian): $K = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$.

## 5. Every formula explained

- Maximum margin = most robust hyperplane, tightest generalization bound.
- Slack $\xi_i$ allows outliers; $C$ = cost of violation.
- Dual formulation depends only on inner products → kernel trick lets us work in high (even infinite) dim spaces without computing coordinates.

## 6. Variables

| Symbol | Meaning |
|--------|---------|
| $\mathbf{w}, b$ | hyperplane |
| $\xi_i$ | slack for $i$-th sample |
| $C$ | regularization / error cost |
| $\alpha_i$ | dual variables |
| $K$ | kernel function |
| $\gamma$ | kernel width (RBF) |

## 7. Algorithm — SMO (Sequential Minimal Optimization)

Iteratively pick two $\alpha_i, \alpha_j$; solve the 2-variable QP analytically; repeat until KKT conditions satisfied within tolerance.

For linear SVM at scale: use `LinearSVC` (liblinear) or SGD with hinge loss.

## 8. Simple example

Two 2D clusters separated by a wide gap. SVM picks the hyperplane in the middle; support vectors are the closest few points from each cluster.

## 9. Real-world example

- Handwritten digit recognition (pre-DNN era).
- Text classification (linear SVM on TF-IDF is a *very* strong baseline).
- Face detection (Viola-Jones + SVM historically).

## 10. Diagram

```
     x2
      |    o o
      |   o     ★ ← support vectors
      | -----margin-----
      |          ★
      |             x  x
      |               x
      +-------------------- x1
```

## 11. Implementation from scratch — SGD on hinge loss

```python
import numpy as np

class LinearSVM:
    def __init__(self, C=1.0, lr=0.01, n_iters=1000):
        self.C = C; self.lr = lr; self.n_iters = n_iters
    def fit(self, X, y):                # y ∈ {-1, +1}
        n, d = X.shape
        self.w = np.zeros(d); self.b = 0.0
        for _ in range(self.n_iters):
            margins = y * (X @ self.w + self.b)
            mask = margins < 1
            grad_w = self.w - self.C * (X[mask] * y[mask][:, None]).sum(axis=0)
            grad_b = -self.C * y[mask].sum()
            self.w -= self.lr * grad_w / n
            self.b -= self.lr * grad_b / n
        return self
    def predict(self, X):
        return np.sign(X @ self.w + self.b)
```

## 12. Implementation using libraries

```python
from sklearn.svm import SVC, SVR, LinearSVC, OneClassSVM
SVC(kernel="rbf", C=1.0, gamma="scale", probability=False).fit(X, y)
LinearSVC(C=1.0, loss="squared_hinge", dual=False).fit(X_sparse, y)  # scales better
```

## 13. Time complexity

- Kernel SVM: O(n² d) to O(n³) — bad for large n.
- Linear SVM (liblinear): O(n d) practically.
- Prediction: O(#support-vectors × d).

## 14. Space complexity

Stores support vectors. For RBF this can be a large fraction of the training set.

## 15. Advantages

- Solid theory, unique optimum (convex QP).
- Works well in high-dim spaces (text).
- Effective with small data.

## 16. Disadvantages

- Poor scaling to millions of samples with non-linear kernels.
- Requires feature scaling.
- Probabilistic outputs need extra calibration (Platt scaling).
- Hyperparameter tuning ($C$, $\gamma$) is finicky.

## 17. Interview questions

1. Derive the primal and dual of the SVM.
2. What is the kernel trick? Give three kernels.
3. Meaning of $C$ and $\gamma$?
4. What is a support vector?
5. Why does SVM need feature scaling?
6. Hinge loss vs logistic loss.
7. SVM vs Logistic Regression when to choose?
8. Explain the geometric margin.
9. Time complexity of training a kernel SVM?
10. What is One-Class SVM? Where is it used?

## 18. Common mistakes

- Forgetting to scale features.
- Using RBF on huge datasets.
- Ignoring class imbalance (`class_weight="balanced"`).
- Using `SVC` when `LinearSVC` would do the job.

## 19. Optimization techniques

- Random Fourier Features to approximate RBF for large data.
- Nyström method.
- Warm-start over the $C$ path.

## 20. Coding exercises

1. Implement soft-margin SVM via cvxpy quadratic solver.
2. Compare linear SVM and logistic regression on 20-newsgroups.
3. Visualize decision boundary for RBF SVM with varying $\gamma$.
4. Implement One-Class SVM for novelty detection on MNIST.

## 21. Mini project

Text classification on 20-newsgroups with `LinearSVC`; achieve > 90% F1.

## 22. Medium project

Face recognition on LFW using HOG + kernel SVM.

## 23. Advanced project

Custom kernel for graph classification; benchmark vs graph neural networks on small molecule datasets (MUTAG).

## 24. Where it is used in industry

Text classification, bioinformatics (kernel methods on sequences), anomaly detection.

## 25. How companies use it

- Bioinformatics research groups (protein classification).
- Novelty / anomaly detection systems.
- Legacy production baselines still surviving in NLP pipelines.

## 26. When NOT to use it

- Millions of samples with a non-linear kernel — too slow.
- Perceptual data — DNNs dominate.
