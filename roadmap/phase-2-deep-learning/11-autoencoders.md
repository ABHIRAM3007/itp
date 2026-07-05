# 2.11 — Autoencoders & VAEs

## 1. Intuition first
An **autoencoder** is a neural network that learns to reconstruct its input through a low-dimensional bottleneck — forcing it to learn a compressed representation. A **VAE** replaces the deterministic bottleneck with a *distribution*, enabling generation.

## 2. Why the topic exists
Unsupervised representation learning, denoising, anomaly detection, generative modeling, image compression.

## 3. What problem it solves
Learn compact features without labels; generate new samples; detect outliers.

## 4. Mathematics

### 4.1 Autoencoder
Encoder $f: \mathbb{R}^d \to \mathbb{R}^k$; decoder $g: \mathbb{R}^k \to \mathbb{R}^d$; loss $\|x - g(f(x))\|^2$.

### 4.2 Denoising AE
Corrupt input $\tilde x$; train to reconstruct clean $x$.

### 4.3 Sparse AE
Add sparsity penalty on hidden activations (KL to Bernoulli prior).

### 4.4 Variational Autoencoder (VAE)
Encoder outputs mean $\mu(x)$ and log-variance $\log \sigma^2(x)$; latent $z \sim \mathcal{N}(\mu, \sigma^2)$.

**Evidence Lower Bound (ELBO):**

$$
\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\mathrm{KL}}(q(z|x) \parallel p(z))
$$

Minimize $-\text{ELBO}$ = reconstruction loss + KL to standard normal prior.

**Reparameterization trick:** $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$ — makes sampling differentiable.

### 4.5 VQ-VAE
Discrete latents via nearest-neighbor lookup in a learned codebook; foundation for tokenizing images/audio for LLMs.

## 5. Every formula explained
- Reconstruction encourages $z$ to preserve info.
- KL term regularizes latent space to be close to standard normal → interpolatable.
- Reparameterization lets gradient flow through the sampling.

## 6. Variables
$x$ input; $z$ latent; $\mu, \sigma$ posterior params; $p(z)$ prior; $q(z|x)$ approximate posterior.

## 7. Algorithm — VAE step
1. Encode $x$ → $\mu, \log\sigma^2$.
2. Sample $\epsilon$, compute $z = \mu + e^{0.5 \log\sigma^2} \epsilon$.
3. Decode $\hat x = g(z)$.
4. Loss = reconstruction + KL.
5. Backprop.

## 8. Simple example
VAE on MNIST: 2D latent → visualize digits interpolating smoothly across the plane.

## 9. Real-world example
- VQ-VAE tokenizers underpin many audio and image generative models (Jukebox, Whisper's tokenizer analogs, Sora-style pipelines).
- Autoencoders for anomaly detection in industrial IoT.
- Stable Diffusion's VAE encodes images into latent space where the diffusion model operates.

## 10. Diagram
```
   x ─► Encoder ─► μ, σ ─► z = μ + σε ─► Decoder ─► x̂
                                             │
                                         KL(q(z|x) || N(0,I))
```

## 11. Implementation from scratch — VAE in PyTorch
```python
import torch, torch.nn as nn, torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, d_in=784, d_h=400, d_z=20):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.mu = nn.Linear(d_h, d_z); self.lv = nn.Linear(d_h, d_z)
        self.fc2 = nn.Linear(d_z, d_h); self.out = nn.Linear(d_h, d_in)
    def encode(self, x):
        h = F.relu(self.fc1(x)); return self.mu(h), self.lv(h)
    def reparam(self, mu, lv):
        std = torch.exp(0.5 * lv); eps = torch.randn_like(std); return mu + eps * std
    def decode(self, z):
        return torch.sigmoid(self.out(F.relu(self.fc2(z))))
    def forward(self, x):
        mu, lv = self.encode(x); z = self.reparam(mu, lv)
        return self.decode(z), mu, lv

def loss_fn(xh, x, mu, lv):
    bce = F.binary_cross_entropy(xh, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
    return bce + kld
```

## 12. Implementation using libraries
Use `diffusers` or `torchvision` for pretrained VAEs (e.g., Stable Diffusion's).

## 13. Time complexity
Same as forward/backward of underlying networks.

## 14. Space complexity
Latent dim usually small (< 1024).

## 15. Advantages
Unsupervised; smooth latent space; generative; compresses.

## 16. Disadvantages
Blurry samples (Gaussian likelihood). Beaten by GANs / diffusion in image quality.

## 17. Interview questions
1. Explain the reparameterization trick.
2. Derive the ELBO.
3. AE vs VAE.
4. Why does the KL term push $q(z|x)$ toward $N(0, I)$?
5. Explain VQ-VAE.
6. Posterior collapse — what and how to fix (KL annealing, free bits).
7. Difference between $\beta$-VAE and vanilla VAE.
8. Use cases for autoencoders in anomaly detection.
9. Why do VAE images look blurry?
10. How is VAE used inside Stable Diffusion?

## 18. Common mistakes
- Forgetting to sigmoid the decoder for binary reconstruction.
- Bad KL weight → posterior collapse.
- Training on unnormalized data.

## 19. Optimization techniques
KL annealing, free bits, $\beta$-VAE, hierarchical VAEs (NVAE), vector quantization.

## 20. Coding exercises
1. Train VAE on MNIST; interpolate between two digits in latent space.
2. Add $\beta$-VAE and compare disentanglement.
3. Implement VQ-VAE from scratch.
4. Denoising AE for image inpainting.

## 21. Mini project
Latent-space interpolation on MNIST or Fashion-MNIST.

## 22. Medium project
VAE for anomaly detection on MVTec-AD; report AUROC per category.

## 23. Advanced project
VQ-VAE on CIFAR-10 followed by a small autoregressive transformer over discrete latents; sample images.

## 24. Where it is used in industry
Latent-diffusion pipelines; anomaly detection in manufacturing; audio codecs (SoundStream, EnCodec).

## 25. How companies use it
- Stability AI's Stable Diffusion (VAE).
- Meta's EnCodec (VQ audio codec).
- Google's SoundStream.

## 26. When NOT to use it
- High-fidelity image generation alone → diffusion / GAN.
- Small tabular data — regular embeddings suffice.
