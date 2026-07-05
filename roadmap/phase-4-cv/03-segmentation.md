# 4.3 — Segmentation (Semantic, Instance, Panoptic, SAM)

## 1. Intuition first
Predict a label for **every pixel** — semantic segmentation. Distinguish separate objects — instance segmentation. Combine both — panoptic.

## 2. Why the topic exists
Precise spatial understanding needed for medical imaging, self-driving, AR, satellite imagery.

## 3. What problem it solves
Pixel-accurate object masks; scene parsing; anatomy segmentation; land-use mapping.

## 4. Mathematics

### 4.1 Losses
- Cross-entropy per pixel.
- Dice loss: $1 - \frac{2\sum p_i g_i}{\sum p_i + \sum g_i}$.
- IoU / Jaccard loss.
- Boundary loss, focal Tversky, combo losses.

### 4.2 Architectures
- **U-Net** (encoder-decoder + skip connections) — standard for medical.
- **DeepLab-v3+** (dilated conv + ASPP).
- **Mask R-CNN** — Faster R-CNN + mask head → instance segmentation.
- **Mask2Former, MaskFormer** — transformer set-prediction segmentation.
- **SAM (Segment Anything Model)** — Meta's foundation model with prompt-based masks (point/box/text).

### 4.3 Metrics
Pixel accuracy; mean IoU (mIoU); Dice; boundary F1; panoptic quality (PQ).

## 5. Every formula explained
Dice loss = 2 · intersection / (predicted + gold) — good for imbalanced masks.

## 6. Variables
Number of classes; input size; foreground fraction.

## 7. Algorithm — U-Net training
1. Encoder downsamples via conv+pool.
2. Decoder upsamples via transposed conv / bilinear + skip connection concatenation.
3. Final 1×1 conv projects to $C$ channels + softmax.
4. Loss: CE + Dice (weighted).
5. Data augmentation: flips, elastic, color, coarse dropout.

## 8. Simple example
Segment cars from road scenes at pixel level; visualize overlay.

## 9. Real-world example
- Radiology (tumor segmentation).
- Self-driving lane / drivable-area segmentation.
- Satellite imagery (building footprints).
- Video-conferencing background blur (mediapipe / MODNet).

## 10. Diagram
```
Input ─► conv ─► pool ─┐        ┌─► upconv ─► conv ─► output
                       └────►(skip)───┘
        encoder                    decoder
```

## 11. Implementation from scratch — mini U-Net (PyTorch)
```python
import torch, torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.n = nn.Sequential(nn.Conv2d(i,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                               nn.Conv2d(o,o,3,padding=1), nn.BatchNorm2d(o), nn.ReLU())
    def forward(self, x): return self.n(x)

class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=1, f=64):
        super().__init__()
        self.d1 = DoubleConv(c_in, f);   self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(f, f*2);    self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(f*2, f*4);  self.p3 = nn.MaxPool2d(2)
        self.b  = DoubleConv(f*4, f*8)
        self.u3 = nn.ConvTranspose2d(f*8, f*4, 2, 2); self.c3 = DoubleConv(f*8, f*4)
        self.u2 = nn.ConvTranspose2d(f*4, f*2, 2, 2); self.c2 = DoubleConv(f*4, f*2)
        self.u1 = nn.ConvTranspose2d(f*2, f, 2, 2);   self.c1 = DoubleConv(f*2, f)
        self.out = nn.Conv2d(f, c_out, 1)
    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(self.p1(d1)); d3 = self.d3(self.p2(d2))
        b  = self.b(self.p3(d3))
        u3 = self.c3(torch.cat([self.u3(b), d3], 1))
        u2 = self.c2(torch.cat([self.u2(u3), d2], 1))
        u1 = self.c1(torch.cat([self.u1(u2), d1], 1))
        return self.out(u1)
```

## 12. Implementation using libraries
```python
# segmentation-models-pytorch
import segmentation_models_pytorch as smp
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=21)

# Detectron2 for Mask R-CNN or Mask2Former
# ultralytics YOLOv8-seg
# SAM
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth"); pred = SamPredictor(sam)
pred.set_image(img); masks, _, _ = pred.predict(point_coords, point_labels)
```

## 13. Time complexity
Comparable to classification backbones; instance seg heavier due to per-mask decoding.

## 14. Space complexity
Higher activations at full resolution.

## 15. Advantages
Pixel-precise outputs; strong models & pretrained weights; SAM enables promptable segmentation without task-specific labels.

## 16. Disadvantages
Annotation is expensive; boundary errors; small object performance.

## 17. Interview questions
1. Semantic vs instance vs panoptic segmentation.
2. Why Dice loss?
3. U-Net skip connections — why?
4. Explain ASPP in DeepLab.
5. Mask R-CNN architecture.
6. What is SAM and how does it work?
7. Metrics: pixel acc vs mIoU vs Dice.
8. Data augmentation for segmentation.
9. Class imbalance in segmentation.
10. Boundary-aware losses.

## 18. Common mistakes
- Interpolating masks with bilinear when nearest is needed.
- Loss instability with heavy Dice weighting.
- Wrong metric (accuracy on imbalanced masks).

## 19. Optimization techniques
Deep supervision at multiple scales; heavy augmentation (Copy-Paste); Test-time augmentation; distilled SAM variants (MobileSAM, EfficientSAM).

## 20. Coding exercises
1. Train U-Net on the Oxford-IIIT Pets dataset.
2. Evaluate mIoU per class on Cityscapes subset.
3. Use SAM to auto-label a small dataset.
4. Implement Dice loss and boundary loss.

## 21. Mini project
Portrait matting for background blur, using MODNet or a U-Net.

## 22. Medium project
Segment lung nodules on a public CT dataset with a 3D U-Net; report Dice.

## 23. Advanced project
Fine-tune SAM with LoRA adapters on a specialized domain (medical, satellite) and beat a from-scratch U-Net with 10% of the labels.

## 24. Where it is used in industry
Medical imaging (Aidoc, PathAI), self-driving perception, AR (Snap filters), satellite (Planet Labs).

## 25. How companies use it
- Radiology triage.
- Retail (planogram compliance).
- Sports (player tracking).
- Meta's SAM as a foundation model for downstream tasks.

## 26. When NOT to use it
- Coarse localization needs → detection is faster.
- Very small labeled datasets without transfer learning.
