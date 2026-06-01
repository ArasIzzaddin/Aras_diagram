# Aras' Diagram

A Python implementation of **Aras' diagram** — a graphical tool for performance evaluation of complex models, introduced in:

> Izzaddin, A., Langousis, A., Totaro, V., Yaseen, M., & Iacobellis, V. (2024). *A new diagram for performance evaluation of complex models.* Stochastic Environmental Research and Risk Assessment, 38, 2261–2281. https://doi.org/10.1007/s00477-024-02678-3

## What it does

Aras' diagram evaluates model performance by integrating three components in a single 2D plot:

| Component | Formula | Meaning |
|---|---|---|
| **β** | μ_sim / μ_obs | Bias ratio (x-axis: β − 1) |
| **α** | σ_sim / σ_obs | Variability ratio (y-axis: α − 1) |
| **r** | Pearson correlation | Encoded as line segment length |

Each model is shown as two points connected by a line:
- **Circle** — E_αβ = √[(β−1)² + (α−1)²] — bias + variability error. Filled if r > 0, empty if r < 0.
- **Triangle** — E = √[(1−r)² + (β−1)² + (α−1)²] — total error including correlation
- **Line** — length represents the correlation error component

The origin (0, 0) is the perfect score. Concentric circles mark 10%, 25%, and 50% total error.

## Installation

```bash
pip install numpy matplotlib
```

## Quick start

```python
import numpy as np
from aras_core import compute_aras_components
from aras_plot import plot_aras_diagram
import matplotlib.pyplot as plt

obs = np.array([...])   # observations
sim = np.array([...])   # model output

components = compute_aras_components(sim, obs)
print(components)
# {'r': 0.85, 'beta': 1.05, 'alpha': 0.92, 'E_ab': 0.054, 'L_aras': 0.207, 'kge': 0.793}

# Plot multiple models
models = {
    'Model A': {'precipitation': compute_aras_components(sim_A, obs)},
    'Model B': {'precipitation': compute_aras_components(sim_B, obs)},
}

fig, ax = plt.subplots(figsize=(8, 8))
plot_aras_diagram(models, variable='precipitation', ax=ax)
plt.show()
```

## Files

| File | Description |
|---|---|
| `aras_core.py` | Computes β, α, r, E_αβ, L_Aras (= 1 − KGE) |
| `aras_plot.py` | Produces the Aras diagram in the style of the original paper |

## Citation

```bibtex
@article{izzaddin2024,
  title   = {A new diagram for performance evaluation of complex models},
  author  = {Izzaddin, Aras and Langousis, Andreas and Totaro, Vincenzo
             and Yaseen, Marwah and Iacobellis, Vito},
  journal = {Stochastic Environmental Research and Risk Assessment},
  volume  = {38},
  pages   = {2261--2281},
  year    = {2024},
  doi     = {10.1007/s00477-024-02678-3}
}
```

## License

MIT License — free to use, cite the paper.
