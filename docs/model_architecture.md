# Model architecture (summary)

Full derivations, tensor shapes and design rationale are in `technical_report.md`, Chapter 5.

Both models map an input `x ∈ R^{B×C×224×224}` (C = 4 for RGB-D, 3 for RGB) to four dense maps
of the same spatial size: grasp quality `Q`, `cos 2θ`, `sin 2θ` and normalised width `W`.
The heads (`models/heads.py`) are shared code: four parallel 2×2 convolutions with no output
activation, preceded by optional dropout.

```mermaid
flowchart LR
  X["input 4×224×224<br/>(depth, R, G, B)"] --> C1["conv 9×9 s1 → 32×224×224<br/>BN, ReLU"]
  C1 --> C2["conv 4×4 s2 → 64×112×112<br/>BN, ReLU"]
  C2 --> C3["conv 4×4 s2 → 128×56×56<br/>BN, ReLU"]
  C3 --> R["5 × residual block<br/>128×56×56"]
  R --> D1["deconv 4×4 s2 → 64×113×113<br/>BN, ReLU"]
  D1 --> D2["deconv 4×4 s2 → 32×225×225<br/>BN, ReLU"]
  D2 --> D3["deconv 9×9 s1 → 32×225×225"]
  D3 --> H["dropout p=0.1<br/>4 × conv 2×2"]
  H --> Q["Q 1×224×224"]
  H --> CO["cos2θ"]
  H --> SI["sin2θ"]
  H --> W["W"]
```

| | GG-CNN (baseline) | GR-ConvNet (main) |
|---|---|---|
| Source | Morrison et al., RSS 2018 | Kumra et al., IROS 2020 |
| Encoder | 3 strided convs 9×9/3, 5×5/2, 3×3/2 (32, 16, 8 ch) | 9×9/1, 4×4/2, 4×4/2 (32, 64, 128 ch) + BN |
| Bottleneck | none (8 channels at 19×19) | 5 residual blocks at 128×56×56 |
| Decoder | 3 transposed convs (mirror) + bilinear resize to 225×225 | 3 transposed convs + BN |
| Parameters (RGB-D input) | 70,196 | 1,900,900 |
| Loss (identical) | sum of Smooth-L1 over the 4 maps | same |

Parameter counts printed by `scripts/train.py` (`model parameters:` log line). GR-ConvNet's count
equals the 1,900,900 reported in the paper.
