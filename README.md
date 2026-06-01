# Aras' Diagram

A Python implementation of **Aras' diagram** — a graphical tool for performance evaluation of complex models, introduced in:

> Izzaddin, A., Langousis, A., Totaro, V., Yaseen, M., & Iacobellis, V. (2024). *A new diagram for performance evaluation of complex models.* Stochastic Environmental Research and Risk Assessment, 38, 2261–2281. https://doi.org/10.1007/s00477-024-02678-3

## What it does

Aras' diagram evaluates model performance by integrating three components in a single 2D plot:

- **β** = μ_sim / μ_obs — bias ratio (x-axis: β − 1)
- **α** = σ_sim / σ_obs — variability ratio (y-axis: α − 1)
- **r** — Pearson correlation (encoded as line segment length)

The diagram decomposes total model error into:
- **E_αβ** = √[(β−1)² + (α−1)²] — bias + variability error (circle marker)
- **E** = √[(1−r)² + (β−1)² + (α−1)²] — total error including correlation (triangle marker)

The connecting line between circle and triangle represents the **correlation error**.

## Installation

```bash
pip install numpy matplotlib xarray
```

## Quick start

```python
from aras_core import compute_aras_components
from aras_plot import plot_aras_diagram
import matplotlib.pyplot as plt

# Compute components for one model
components = compute_aras_components(sim=model_output, obs=observations)
# Returns: r, beta, alpha, E_ab, L_aras, kge

# Plot multiple models
genomes = {
    'Model-A': {'tas': compute_aras_components(sim_A, obs)},
    'Model-B': {'tas': compute_aras_components(sim_B, obs)},
}

fig, ax = plt.subplots(figsize=(8, 8))
plot_aras_diagram(genomes, variable='tas', ax=ax)
plt.show()
```

## Multi-variable fingerprint

```python
from aras_core import compute_genome, genome_matrix

# Build fingerprint for one model across multiple variables
genome = compute_genome(model_data, obs_data, variables=['tas', 'pr', 'psl'])

# Pairwise distance matrix between models
names, D = genome_matrix(genomes)
```

## Files

| File | Description |
|---|---|
| `aras_core.py` | Core computations: β, α, r, E_αβ, L_Aras, fingerprint distance |
| `aras_plot.py` | Aras diagram (paper style), heatmaps, distance matrix |

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
