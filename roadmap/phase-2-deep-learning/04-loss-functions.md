# 2.4 — Loss Functions

## 1. Intuition first
The loss defines "what wrong looks like". It converts predictions and targets into a scalar you minimize. Wrong loss → wrong model, regardless of architecture.

## 2. Why the topic exists
Different tasks (regression, classification, ranking, generation, contrastive) need different loss geometries. Also, losses derive from probabilistic models (MLE), so choice encodes assumptions.

## 3. What problem it solves
Provides a differentiable objective compatible with gradient-based optimization.

## 4. Mathematics — the essential zoo

### Regression
- **MSE**: $\tfrac{1}{n}\sum (y-\hat y)^2$. Gaussian MLE.
- **MAE**: $\tfrac{1}{n}\sum |y-\hat y|$. Laplace MLE, robust.
- **Huber**: quadratic near 0, linear far away — combines the two.
- **Quantile (pinball)**: $\max(\tau(y-\hat y), (\tau-1)(y-\hat y))$ for quantile $\tau$.
- **Log-Cosh**: smooth Huber.

### Binary classification
- **BCE**: $-y\log\hat p - (1-y)\log(1-\hat p)$.
- **BCE with logits** = numerically stable (`log-sum-exp`).
- **Hinge**: $\max(0, 1 - y\hat y)$ with $y\in\{-1,+1\}$ (SVM).
- **Focal**: $-(1-\hat p_y)^\gamma \log \hat p_y$ — downweights easy examples (RetinaNet).

### Multi-class
- **Categorical cross-entropy** = negative log-likelihood over softmax.
- **Label-smoothed CE**: replace one-hot with $(1-\epsilon) e_y + \epsilon/K$ — regularization.

### Ranking / metric
- **Triplet loss**: $\max(0, d(a, p) - d(a, n) + m)$.
- **Contrastive loss / InfoNCE**: $-\log\frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}}$ (CLIP).
- **Pairwise logistic**: for LambdaRank etc.

### Sequence / generation
- **Cross-entropy over tokens** (next-token prediction).
- **CTC loss** (Connectionist Temporal Classification) — variable-length alignment (speech).

### Segmentation
- **Dice loss**: $1 - \frac{2\sum p_i g_i}{\sum p_i + \sum g_i}$.
- **IoU / Jaccard loss**.
- **Focal + Dice** hybrid.

### Detection
- **Smooth L1 / GIoU** for bounding boxes.
- **Focal** for class imbalance.

### Generative
- **KL divergence** (VAE, distillation).
- **Adversarial loss** (GAN).
- **DPO / RLHF preference losses** — see 3.6.

## 5. Every formula explained
See table above. Key insight: most classification losses = negative log-likelihood of an underlying probabilistic model.

## 6. Variables
$y$ target, $\hat y / \hat p$ prediction, $\tau$ quantile or temperature, $\gamma$ focal focusing param, $m$ margin.

## 7. Algorithm — pick your loss
1. Identify output type & assumed noise model.
2. Consider class imbalance / hard examples.
3. Consider robustness needs.
4. Ensure numerical stability (`with_logits` variants).
5. Sanity-check gradient magnitude.

## 8. Simple example
Predicting age from photo: MAE is more robust to old-people outliers than MSE.

## 9. Real-world example
- Focal loss enabled RetinaNet to match two-stage detectors.
- InfoNCE loss trained CLIP on 400M image-text pairs.
- Label smoothing standard in Inception, ViT, LLM pretraining.

## 10. Diagram
```
MSE:  parabola      MAE: V-shape       Huber: parabola in, linear out
Focal: BCE × (1 - p̂)^γ   damps easy examples
```

## 11. Implementation from scratch
```python
import numpy as np

def mse(y, p): return ((y - p) ** 2).mean()
def mae(y, p): return np.abs(y - p).mean()
def huber(y, p, d=1.0):
    r = y - p; a = np.abs(r)
    return np.where(a <= d, 0.5 * r**2, d * (a - 0.5 * d)).mean()

def bce_logits(y, z):                         # numerically stable
    return np.maximum(z, 0).mean() - (y * z).mean() + np.log1p(np.exp(-np.abs(z))).mean()

def focal(y, p, gamma=2.0, eps=1e-8):
    pt = np.where(y == 1, p, 1 - p)
    return (-((1 - pt) ** gamma) * np.log(pt + eps)).mean()

def infonce(z_a, z_b, temp=0.07):             # (n, d) normalized
    s = z_a @ z_b.T / temp
    labels = np.arange(len(z_a))
    logsumexp = np.log(np.exp(s).sum(1))
    return (-s[np.arange(len(z_a)), labels] + logsumexp).mean()
```

## 12. Implementation using libraries
```python
import torch.nn as nn, torch.nn.functional as F
nn.MSELoss(); nn.L1Loss(); nn.SmoothL1Loss()
nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
nn.CrossEntropyLoss(label_smoothing=0.1)
nn.CTCLoss()
```

## 13. Time complexity
O(n) per sample or O(nB) for a mini-batch.

## 14. Space complexity
O(1) reduction after per-example compute.

## 15. Advantages
Aligns training with objective. Some (focal, label smoothing) fight overfit.

## 16. Disadvantages
Non-convex losses can be hard to optimize; some (adversarial) have complex dynamics.

## 17. Interview questions
1. Derive BCE from MLE.
2. Why not use MSE for classification?
3. Explain focal loss.
4. What is label smoothing? What does it accomplish?
5. What is CTC loss?
6. Difference between contrastive and triplet loss.
7. Why is `BCEWithLogitsLoss` more stable than sigmoid + BCE?
8. When would you use Huber over MSE?
9. What is KL divergence and when do you use it?
10. Explain quantile loss and its use in probabilistic forecasting.

## 18. Common mistakes
- Sigmoid + BCE (double sigmoid).
- Wrong reduction leading to different effective LR.
- Ignoring class imbalance (missing `pos_weight`).
- Not clamping softmax outputs before `log`.

## 19. Optimization techniques
Loss scaling for mixed precision; auxiliary / multi-task losses; loss weighting via uncertainty (Kendall & Gal).

## 20. Coding exercises
1. Implement label smoothing manually.
2. Implement focal loss and reproduce a small RetinaNet-style demo.
3. Compute quantile forecasts on time series.
4. Show that BCE gradient wrt logit is $\sigma(z) - y$.

## 21. Mini project
Compare MSE vs Huber for predicting delivery times with outliers.

## 22. Medium project
Multi-task learning with uncertainty-weighted losses for joint depth + segmentation.

## 23. Advanced project
Implement supervised contrastive learning (Khosla 2020) with InfoNCE-style loss for CIFAR-100.

## 24. Where it is used in industry
Every training run uses one or more losses. Choosing/inventing the right loss is a research skill.

## 25. How companies use it
- Detection heads use focal + smooth L1.
- LLMs use next-token CE + KL to reference model (DPO).
- Ranking systems use pairwise/listwise losses.

## 26. When NOT to use it
- MSE for classification.
- Vanilla BCE on 100:1 imbalanced data.
- Triplet loss without hard-negative mining.
