# 2.3 — Activation Functions

## 1. Intuition first
Non-linearities give a network the power to represent curves. Without them, stacked linear layers collapse into one linear layer.

## 2. Why the topic exists
To break linearity, keep gradients healthy, and match output ranges to loss functions.

## 3. What problem it solves
Enables universal approximation; helps mitigate vanishing/exploding gradients; provides bounded (or unbounded) outputs.

## 4. Mathematics — the essential zoo

| Name | Formula | Derivative | Range | Notes |
|------|---------|------------|-------|-------|
| Sigmoid | $\sigma(z)=\frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | (0,1) | historical, saturates |
| Tanh | $\tanh(z)$ | $1-\tanh^2(z)$ | (-1,1) | zero-centered, still saturates |
| ReLU | $\max(0,z)$ | $\mathbb{1}[z>0]$ | $[0,\infty)$ | default in CNNs; "dying ReLU" |
| Leaky ReLU | $\max(\alpha z, z)$, $\alpha\!=\!0.01$ | $\alpha$ or 1 | $\mathbb{R}$ | fixes dying ReLU |
| PReLU | learnable $\alpha$ | | | |
| ELU | $z$ if $z>0$, else $\alpha(e^z-1)$ | | | smooth negative |
| GELU | $z\Phi(z)$ | – | $\mathbb{R}$ | default in Transformers |
| SiLU / Swish | $z\sigma(z)$ | | | popular in LLMs |
| Softplus | $\log(1+e^z)$ | $\sigma(z)$ | $(0,\infty)$ | smooth ReLU |
| Softmax | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | – | simplex | multi-class output |
| Mish | $z\tanh(\text{softplus}(z))$ | | | |

## 5. Every formula explained
- Sigmoid/tanh saturate for large |z| → gradient $\to 0$ → vanishing gradients in deep nets.
- ReLU has constant gradient 1 for $z>0$ → cheap, avoids vanishing but "dies" when $z<0$ forever.
- GELU is a smooth, differentiable stochastic-gate approximation; empirically best for Transformers.

## 6. Variables
$z$ pre-activation; $\alpha$ negative slope (Leaky/PReLU); $\Phi$ standard normal CDF (GELU).

## 7. Algorithm — choosing one
- Hidden layers of CNN → ReLU (or Leaky/GELU).
- Hidden layers of Transformer → GELU or SiLU.
- Output for binary classification → sigmoid.
- Output for multi-class → softmax.
- Output for regression → identity (or bounded like sigmoid × scale).

## 8. Simple example
2-layer net with tanh; replacing with ReLU trains faster and reaches lower loss on MNIST.

## 9. Real-world example
- ResNet uses ReLU.
- BERT uses GELU.
- LLaMA uses SwiGLU (Swish × Gate).

## 10. Diagram
```
   ReLU:            Sigmoid:          GELU:
       /                _____             _/
      /               /                  /
     /              /                   /
    /            _/                    /
___/           /                _____/
```

## 11. Implementation from scratch
```python
import numpy as np
def relu(x): return np.maximum(0, x)
def leaky_relu(x, a=0.01): return np.where(x > 0, x, a * x)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
def silu(x): return x / (1 + np.exp(-x))
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)
```

## 12. Implementation using libraries
```python
import torch.nn as nn
nn.ReLU(); nn.LeakyReLU(0.01); nn.GELU(); nn.SiLU(); nn.Softmax(dim=-1)
```

## 13. Time complexity
O(n) elementwise except softmax (O(n) + a normalize).

## 14. Space complexity
Same shape as input.

## 15. Advantages
Non-linearity, cheap, differentiable (a.e.).

## 16. Disadvantages
Saturation (sigmoid/tanh), dead units (ReLU), no bounded range (ReLU can explode).

## 17. Interview questions
1. Why not use sigmoid in hidden layers of deep nets?
2. What is the dying-ReLU problem?
3. Difference between ReLU and GELU.
4. Softmax vs sigmoid — when are they equivalent?
5. What is a swish/SiLU?
6. Explain gradient behavior of tanh.
7. Why must output activation match the loss?
8. What are gated activations (GLU, SwiGLU)?
9. Why don't we use step function as activation?
10. What is Mish and where is it used?

## 18. Common mistakes
- Softmaxing then applying `CrossEntropyLoss` (already applies log-softmax).
- Using sigmoid as output for regression.
- Using ReLU with very high LR → NaN/dead.

## 19. Optimization techniques
Kaiming init pairs with ReLU; Xavier with tanh. Use bias-free layers before BatchNorm.

## 20. Coding exercises
1. Plot every activation and its derivative.
2. Measure fraction of dead ReLUs during training.
3. Replace GELU with ReLU in a mini Transformer; compare.
4. Verify softmax + CE analytic gradient equals PyTorch's autograd.

## 21. Mini project
Ablation study: MNIST MLP with ReLU vs LeakyReLU vs GELU vs Swish. Report accuracy and convergence speed.

## 22. Medium project
Reproduce "Searching for Activation Functions" (Ramachandran et al., 2017) on CIFAR-10.

## 23. Advanced project
Implement Gated Linear Unit variants (GEGLU, SwiGLU) in a small Transformer; benchmark on TinyShakespeare.

## 24. Where it is used in industry
Every neural network everywhere.

## 25. How companies use it
LLaMA/Mistral/DeepSeek use SwiGLU; GPT/BERT use GELU; MobileNet uses ReLU6.

## 26. When NOT to use it
- Skip inserting activation before a softmax loss layer.
- Never at output of a regression head (unless target is bounded).
