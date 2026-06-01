"""
Aras diagram — core computations.
Izzaddin, A. et al. (2024). A new diagram for performance evaluation
of complex models. Stochastic Environmental Research and Risk Assessment,
38, 2261-2281. https://doi.org/10.1007/s00477-024-02678-3

Variability ratio: alpha = sigma_sim / sigma_obs  (Gupta et al. 2009)
Bias ratio:        beta  = mu_sim   / mu_obs
Total error:       total_error = sqrt[(1-r)^2 + (beta-1)^2 + (alpha-1)^2]
"""

import numpy as np


def compute_aras_components(sim: np.ndarray, obs: np.ndarray) -> dict:
    """
    Compute Aras diagram components for one model-observation pair.

    Parameters
    ----------
    sim : array-like — model output (any shape, flattened internally)
    obs : array-like — observations, same shape as sim

    Returns
    -------
    dict with keys:
        r      - Pearson correlation coefficient
        beta   - bias ratio  (mu_sim / mu_obs)
        alpha  - variability ratio  (sigma_sim / sigma_obs)
        E_ab   - bias-variability error  sqrt[(beta-1)^2 + (alpha-1)^2]
        total_error - total error  sqrt[(1-r)^2 + (beta-1)^2 + (alpha-1)^2]
        kge    - Kling-Gupta efficiency  (1 - total_error)
    """
    sim = np.asarray(sim).flatten()
    obs = np.asarray(obs).flatten()

    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]

    if len(sim) < 3:
        return dict(r=np.nan, beta=np.nan, alpha=np.nan,
                    E_ab=np.nan, total_error=np.nan, kge=np.nan)

    mu_sim    = sim.mean()
    mu_obs    = obs.mean()
    sigma_sim = sim.std()
    sigma_obs = obs.std()

    r     = np.corrcoef(sim, obs)[0, 1]
    beta  = mu_sim    / mu_obs    if mu_obs    != 0 else np.nan
    alpha = sigma_sim / sigma_obs if sigma_obs != 0 else np.nan

    E_ab   = np.sqrt((beta  - 1)**2 + (alpha - 1)**2)
    total_error = np.sqrt((1 - r)**2    + (beta  - 1)**2 + (alpha - 1)**2)
    kge    = 1 - total_error

    return dict(r=r, beta=beta, alpha=alpha, E_ab=E_ab, total_error=total_error, kge=kge)
