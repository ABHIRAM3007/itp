# 2.6 — Deep-Learning Regularization (Dropout, BatchNorm, LayerNorm, Weight Decay, Data Aug)

## 1. Intuition first
Deep nets can memorize. Regularization pushes them to learn *general* patterns via noise, normalization, weight penalties, and data augmentation.

## 2. Why the topic exists
Overparameterized models overfit small data. Regularization is what makes deep learning work in practice.

## 3. What problem it solves
Reduces variance / test error; stabilizes training; enables larger models to generalize.

## 4. Mathematics

### 4.1 Weight decay (L2)
Add $\lambda \|W\|_2^2$ or subtract $\eta\lambda W$ each step (AdamW-style).

### 4.2 Dropout
During training, randomly zero each unit with probability $p$ (scale surviving units by $1/(1-p)$). At inference, use full network.

### 4.3 Batch Normalization
For a mini-batch: $\mu_B = \tfrac{1}{B}\sum x_i,\ \sigma_B^2 = \tfrac{1}{B}\sum (x_i-\mu_B)^2$. Then
$\hat x_i = (x_i - \mu_B)/\sqrt{\sigma_B^2 + \epsilon}$; output $y_i = \gamma \hat x_i + \beta$ with learnable $\gamma, \beta$.

At inference, use running statistics.

### 4.4 Layer Normalization
Normalize across features of a single sample: $\mu = \tfrac{1}{d}\sum_j x_j$, $\sigma^2$ similar. **Standard in Transformers**, avoids dependence on batch size.

### 4.5 Other norms
- Instance Norm — style transfer.
- Group Norm — CV with tiny batches.
- RMSNorm — LLaMA-style, drops mean.

### 4.6 Early stopping
Stop when val loss stops improving; equivalent (approximately) to L2 regularization.

### 4.7 Label smoothing (see 2.4)
### 4.8 Data augmentation
Random crop, flip, color jitter, MixUp, CutMix, RandAugment, TrivialAugment. For NLP: back-translation, word dropout.
### 4.9 Stochastic depth / DropPath
Randomly skip residual blocks during training (used in EfficientNet, ConvNeXt, ViT).

## 5. Every formula explained
- Dropout ≈ ensemble of $2^N$ subnetworks, cheaply.
- BN reduces internal covariate shift + smooths loss landscape.
- LN keeps activation scale per-token, essential for autoregressive transformers.

## 6. Variables
$p$ dropout prob; $\gamma, \beta$ learnable scale/shift; $\mu, \sigma$ batch or feature stats; $\lambda$ weight-decay coefficient.

## 7. Algorithm — where to put what
- CNNs: BN after conv, before activation.
- Transformers: pre-norm LN before each sublayer; residual around.
- MLPs on tabular: Dropout after ReLU; weight decay via AdamW.
- Always: strong data augmentation.

## 8. Simple example
Adding Dropout(0.5) to a fully-connected classifier reduces MNIST test error by ~0.5 points.

## 9. Real-world example
BN was the key trick that made ResNet trainable. Layer Norm is the reason Transformers train stably at billion-parameter scale.

## 10. Diagram
```
   x  ─►  Linear  ─►  BN / LN  ─►  Activation  ─►  Dropout  ─►  next layer
```

## 11. Implementation from scratch
```python
import numpy as np

def dropout_forward(x, p, training=True, rng=None):
    if not training or p == 0: return x, None
    rng = rng or np.random
    mask = (rng.random(x.shape) > p) / (1 - p)
    return x * mask, mask

class BatchNorm1d:
    def __init__(self, d, momentum=0.1, eps=1e-5):
        self.gamma = np.ones(d); self.beta = np.zeros(d)
        self.rm = np.zeros(d); self.rv = np.ones(d)
        self.momentum, self.eps = momentum, eps
    def forward(self, x, training=True):
        if training:
            mu = x.mean(0); var = x.var(0)
            self.rm = (1-self.momentum)*self.rm + self.momentum*mu
            self.rv = (1-self.momentum)*self.rv + self.momentum*var
        else:
            mu, var = self.rm, self.rv
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta

def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)
```

## 12. Implementation using libraries
```python
import torch.nn as nn
nn.Dropout(p=0.1); nn.BatchNorm2d(64); nn.LayerNorm(768); nn.RMSNorm(4096)
```

## 13. Time complexity
Elementwise → O(n).

## 14. Space complexity
BN keeps running stats + saved intermediates for backward.

## 15. Advantages
Better generalization; faster convergence; allows larger LR; enables deep architectures.

## 16. Disadvantages
- BN behaves oddly with small batches, sequences, distributed training.
- Dropout hurts if applied to convolutional feature maps or residual stems.
- Too much augmentation → underfit.

## 17. Interview questions
1. Difference between BN, LN, GN, IN.
2. Why do transformers use LayerNorm instead of BatchNorm?
3. Why scale survivors by $1/(1-p)$ in dropout?
4. Explain internal covariate shift.
5. Weight decay vs L2 in Adam.
6. What is MixUp / CutMix?
7. Explain stochastic depth.
8. What is RMSNorm?
9. Pre-norm vs post-norm transformers.
10. When would you disable BN?

## 18. Common mistakes
- Forgetting `model.eval()` at inference → BN uses batch stats → garbage.
- Applying Dropout right before softmax (loses info).
- Mixing L2 with AdamW's decoupled decay.

## 19. Optimization techniques
Combine multiple regularizers; tune independently. Auto-augment via RandAugment/TrivialAugment. Sharpness-aware minimization.

## 20. Coding exercises
1. Add BatchNorm to your from-scratch MLP; retrain and compare.
2. Implement MixUp and CutMix.
3. Ablate label smoothing on CIFAR-10.
4. Reproduce Group Norm vs BN for tiny-batch training.

## 21. Mini project
Train ResNet-20 on CIFAR-10 with/without BN and standard augmentation.

## 22. Medium project
Study augmentation strength vs test accuracy on CIFAR-100 using RandAugment.

## 23. Advanced project
Implement Consistency Regularization / Mean Teacher for semi-supervised CIFAR-10.

## 24. Where it is used in industry
Every serious deep-learning training pipeline.

## 25. How companies use it
LLM pretraining uses weight decay + dropout + gradient clipping + LN. Vision models rely on data augmentation as the biggest regularizer.

## 26. When NOT to use it
- Dropout in convolutional feature maps rarely helps.
- BN with batch size 1.
- Aug that changes semantic label (over-augmentation).
