# 1.5 — Regularization (L1, L2, Elastic Net)

## 1. Intuition first

An unregularized model minimizes fit-error only. It will happily grow huge weights to memorize noise. **Regularization adds a penalty** on weight magnitude so the model prefers simpler, more generalizable solutions.

## 2. Why the topic exists

- Prevent overfitting.
- Stabilize under multicollinearity.
- Perform feature selection (L1).
- Encode a Gaussian (L2) or Laplace (L1) prior on weights.

## 3. What problem it solves

Bias-variance trade-off: adds a bit of bias to greatly reduce variance and improve test error.

## 4. Mathematics

**Ridge (L2)**:

$$
J_{\text{ridge}}(\mathbf{w}) = \tfrac{1}{n}\sum_i \ell(f(\mathbf{x}_i), y_i) + \lambda \|\mathbf{w}\|_2^2
$$

**Lasso (L1)**:

$$
J_{\text{lasso}}(\mathbf{w}) = \tfrac{1}{n}\sum_i \ell(f(\mathbf{x}_i), y_i) + \lambda \|\mathbf{w}\|_1
$$

**Elastic Net**:

$$
J_{\text{en}}(\mathbf{w}) = \tfrac{1}{n}\sum_i \ell(\ldots) + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2
$$

Bayesian interpretation: L2 = Gaussian prior on $\mathbf{w}$; L1 = Laplace prior (spike at 0).

Closed form ridge for OLS: $\mathbf{w}^* = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$.

L1 has no closed form; solved with coordinate descent, proximal gradient (ISTA/FISTA), or LARS.

## 5. Every formula explained

- $\lambda$ controls the trade-off: $\lambda = 0$ → OLS; $\lambda \to \infty$ → $\mathbf{w} \to 0$.
- L2 shrinks weights smoothly toward 0.
- L1 sets many weights to *exactly* 0 → sparse solution → feature selection.

## 6. Variables

| Symbol | Meaning |
|--------|---------|
| $\lambda$ | regularization strength |
| $\|\mathbf{w}\|_2^2$ | sum of squared weights |
| $\|\mathbf{w}\|_1$ | sum of absolute weights |

## 7. Algorithm — coordinate descent for Lasso

For each coordinate $j$, holding others fixed:

$$
w_j \leftarrow \text{soft-threshold}\Big(\tfrac{1}{n}\sum_i x_{ij}(y_i - \hat y_i^{(-j)}), \tfrac{\lambda}{2}\Big)
$$

where $\text{soft}(z, \tau) = \text{sign}(z) \max(|z| - \tau, 0)$.

## 8. Simple example

10 features, only 3 truly relevant. Lasso with a well-chosen $\lambda$ zeroes out the 7 irrelevant ones. Ridge keeps all 10 with small values.

## 9. Real-world example

- Genomics: 20 000 genes, 200 patients → without L1 you severely overfit; Lasso selects handfuls of relevant genes.
- Marketing mix modeling: L2 stabilizes coefficients across highly correlated media channels.

## 10. Diagram — geometric view of the constraint region

```
   L2 (ridge):      L1 (lasso):
      circle           diamond
     _______           /\
    /       \         /  \
   |    +    |       +----+
    \_______/         \  /
                       \/
   Contact point → smooth  Contact point → axis (sparse)
```

## 11. Implementation from scratch — coordinate descent for Lasso

```python
import numpy as np

def soft_threshold(z, tau):
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0)

def lasso_cd(X, y, lam=0.1, n_iter=1000, tol=1e-6):
    n, d = X.shape
    w = np.zeros(d)
    xj_norm = (X ** 2).sum(axis=0)
    for _ in range(n_iter):
        w_old = w.copy()
        for j in range(d):
            r = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ r
            w[j] = soft_threshold(rho, lam * n / 2) / xj_norm[j]
        if np.linalg.norm(w - w_old) < tol:
            break
    return w
```

## 12. Implementation using libraries

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
Ridge(alpha=1.0).fit(X, y)
Lasso(alpha=0.01).fit(X, y)
ElasticNet(alpha=0.01, l1_ratio=0.5).fit(X, y)
LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0)
```

## 13. Time complexity

Ridge closed form: same as OLS. Lasso CD: O(nd × #passes). Proximal-gradient: similar.

## 14. Space complexity

O(d) for weights.

## 15. Advantages

- Better generalization.
- Numerical stability.
- Feature selection (L1).

## 16. Disadvantages

- Adds a hyperparameter to tune.
- L1 is unstable when features are correlated (may pick one arbitrarily).
- Requires feature scaling.

## 17. Interview questions

1. Why do we regularize?
2. L1 vs L2 — mathematical and practical differences.
3. Why does L1 induce sparsity? (geometric argument above)
4. Bayesian interpretation of L2 vs L1.
5. Why must you standardize features before regularization?
6. What is Elastic Net and when to use it?
7. How do you tune $\lambda$ (cross-validation, LARS path)?
8. Does regularization affect the intercept?
9. Difference between weight decay and L2 regularization in Adam?
10. What is early stopping? Is it a regularizer?

## 18. Common mistakes

- Regularizing the bias term.
- Not standardizing features.
- Using the same $\lambda$ across datasets of different scale.
- Assuming Lasso is deterministic in the presence of correlated features.

## 19. Optimization techniques

- Warm-start solutions along a $\lambda$-path.
- FISTA (accelerated proximal gradient).
- Group Lasso for grouped features.
- Nested CV to tune $\lambda$ honestly.

## 20. Coding exercises

1. Implement Lasso via ISTA and FISTA; compare convergence.
2. Plot the coefficient path over $\lambda$ (like `sklearn`'s LassoLars).
3. Show ridge stabilizes coefficients on a multicollinear dataset.
4. Reproduce Zou & Hastie's Elastic Net example.

## 21. Mini project

Genomic feature selection: on a small microarray dataset, use Lasso + CV to identify predictive genes.

## 22. Medium project

Interpret a Lasso model on the Boston/California housing data — report selected features, coefficient path plot, cross-validated $\lambda$.

## 23. Advanced project

Implement a scalable coordinate-descent Lasso in Cython/Numba that matches `sklearn.Lasso` performance on 1 M × 1 K data.

## 24. Where it is used in industry

Everywhere linear/logistic models are used. Regularization is the difference between production-grade and toy models.

## 25. How companies use it

- Ad-tech: L2 in massive-scale LR.
- Bioinformatics: Lasso for biomarker discovery.
- Weight decay in every deep-learning training run.

## 26. When NOT to use it

- Truly interpretable, unregularized coefficients required for regulatory reporting.
- Deep learning with BatchNorm — weight decay is still used but interacts subtly (see AdamW).
