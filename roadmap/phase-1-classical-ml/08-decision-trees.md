# 1.8 — Decision Trees

## 1. Intuition first

Ask a sequence of yes/no questions on features until you can commit to a label. Each internal node = a question ("is age > 30?"), each leaf = a prediction. Humans understand these easily → interpretable ML.

## 2. Why the topic exists

- Non-linear, interaction-capturing, no-scaling-needed model.
- Foundation of Random Forests and Gradient Boosting — the algorithms that dominate tabular ML.

## 3. What problem it solves

Classification and regression on tabular data with mixed feature types, missing values, and non-linear interactions.

## 4. Mathematics

At each node, choose the split $(j, t)$ (feature $j$, threshold $t$) that maximizes **impurity reduction**:

$$
\Delta I = I(\text{parent}) - \Big(\tfrac{n_L}{n} I(\text{left}) + \tfrac{n_R}{n} I(\text{right})\Big)
$$

Common impurity measures:

- **Gini**: $I_G = 1 - \sum_c p_c^2$.
- **Entropy**: $I_H = -\sum_c p_c \log p_c$; information gain = $\Delta I_H$.
- **MSE** (regression): $I_M = \tfrac{1}{n}\sum_i (y_i - \bar y)^2$.

Prediction at a leaf: majority class (classification) or mean $y$ (regression).

## 5. Every formula explained

- Gini and entropy both measure class purity — small ⇒ pure. Nearly identical in practice.
- MSE for regression trees: minimizing variance in each child.
- Impurity reduction picks the most informative split.

## 6. Variables

| Symbol | Meaning |
|--------|---------|
| $p_c$ | fraction of samples in class $c$ at this node |
| $n, n_L, n_R$ | sample counts at node/left/right |
| $j, t$ | feature index and threshold |

## 7. Algorithm — CART (Classification & Regression Trees)

1. If stopping criterion met (max depth, min leaf size, pure node), return leaf.
2. For every feature $j$: sort values; scan candidate thresholds; compute impurity reduction.
3. Pick the best $(j, t)$.
4. Recurse on left and right subsets.

Post-processing: prune (cost-complexity pruning) to reduce overfitting.

## 8. Simple example

Predict "play tennis" given weather (Outlook, Humidity, Wind). First split on Outlook (highest info gain), then within Sunny split on Humidity, and so on.

## 9. Real-world example

- Credit scoring "reason codes" (regulatory need for interpretability).
- Rule extraction for medical diagnosis.
- Baseline for tabular ML competitions.

## 10. Diagram

```
              Outlook?
             /   |    \
         Sunny  Overcast  Rain
           |      |         |
      Humidity?  Play      Wind?
       /    \             /    \
     High  Low         Weak  Strong
       |    |           |      |
      No   Play        Play   No
```

## 11. Implementation from scratch (classification, Gini)

```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature; self.threshold = threshold
        self.left = left; self.right = right; self.value = value

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth; self.min_samples_split = min_samples_split

    @staticmethod
    def _gini(y):
        _, cnt = np.unique(y, return_counts=True)
        p = cnt / cnt.sum()
        return 1 - (p ** 2).sum()

    def _best_split(self, X, y):
        best = (None, None, np.inf)
        n, d = X.shape
        base = self._gini(y)
        for j in range(d):
            vals = np.unique(X[:, j])
            for t in (vals[:-1] + vals[1:]) / 2:
                left = y[X[:, j] <= t]; right = y[X[:, j] > t]
                if len(left) == 0 or len(right) == 0: continue
                g = (len(left)*self._gini(left) + len(right)*self._gini(right)) / n
                if g < best[2]:
                    best = (j, t, g)
        return best

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1:
            return Node(value=Counter(y).most_common(1)[0][0])
        j, t, _ = self._best_split(X, y)
        if j is None:
            return Node(value=Counter(y).most_common(1)[0][0])
        mask = X[:, j] <= t
        return Node(j, t,
                    self._build(X[mask], y[mask], depth + 1),
                    self._build(X[~mask], y[~mask], depth + 1))

    def fit(self, X, y):
        self.root = self._build(np.asarray(X, float), np.asarray(y), 0)
        return self

    def _predict_one(self, x, node):
        if node.value is not None: return node.value
        return self._predict_one(x, node.left if x[node.feature] <= node.threshold else node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in np.asarray(X, float)])
```

## 12. Implementation using libraries

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
clf = DecisionTreeClassifier(criterion="gini", max_depth=5, min_samples_leaf=10)
clf.fit(X_train, y_train)
plot_tree(clf, feature_names=..., class_names=..., filled=True)
```

## 13. Time complexity

Training: O(n d log n) average. Prediction: O(depth) = O(log n) for balanced trees.

## 14. Space complexity

O(#nodes).

## 15. Advantages

- Interpretable.
- Handles mixed types and missing values (with surrogate splits).
- No feature scaling required.
- Captures interactions automatically.

## 16. Disadvantages

- High variance — small data change → very different tree.
- Greedy splits → locally optimal but often not globally.
- Struggles with linear relationships (has to staircase).

## 17. Interview questions

1. Gini vs entropy — do they usually differ?
2. Why do decision trees overfit? How do we prevent it?
3. What is cost-complexity pruning?
4. How do trees handle missing values?
5. Why don't we need to scale features?
6. When does a tree beat a linear model? When does it lose?
7. Explain how a regression tree makes predictions.
8. Difference between CART and ID3/C4.5.
9. Time complexity of training?
10. How do trees form the basis of Random Forest and XGBoost?

## 18. Common mistakes

- Growing unlimited-depth trees.
- Ignoring class imbalance.
- Using default hyperparameters on huge datasets.
- Treating "feature importance" as causal.

## 19. Optimization techniques

- Pre-sort features once; incremental update of impurity.
- Histogram-based splits (LightGBM style) — O(n × d × #bins).
- Regularize via `min_samples_leaf`, `max_depth`, `ccp_alpha`.

## 20. Coding exercises

1. Add entropy criterion.
2. Add regression trees (MSE criterion).
3. Implement cost-complexity pruning.
4. Add support for categorical splits.
5. Visualize decision boundary on 2D data.

## 21. Mini project

Titanic tree of depth 3 with `plot_tree`; interpret splits.

## 22. Medium project

Compare CART, Random Forest, and Gradient Boosting on a Kaggle tabular dataset; produce learning curves and SHAP feature-importance plots.

## 23. Advanced project

Implement a from-scratch histogram-based tree learner and benchmark against LightGBM on 10 M rows.

## 24. Where it is used in industry

Everywhere as building blocks of ensembles; standalone in credit scoring and interpretable rule extraction.

## 25. How companies use it

- Banks (regulator-required explainability).
- Insurance underwriting.
- Feature importance analysis before training deeper models.

## 26. When NOT to use it

- Perceptual data (images/audio/text) — use DNNs.
- Very high-dim sparse data.
- When smooth decision boundaries matter.
