"""
Core Aras diagram computations.
Izzaddin (2024) — extended to multi-variable fingerprint analysis.

Variability ratio: α = σ_sim / σ_obs  (standard deviation ratio, Gupta et al. 2009)
Bias ratio:        β = μ_sim / μ_obs
Total error:       L_Aras = √[(1−r)² + (β−1)² + (α−1)²]  = 1 − KGE
"""

import numpy as np


def compute_aras_components(sim: np.ndarray, obs: np.ndarray) -> dict:
    """
    Compute Aras diagram components for one model-observation pair.

    Returns
    -------
    dict with keys: r, beta, alpha, E_ab, L_aras, kge
    """
    sim = np.asarray(sim).flatten()
    obs = np.asarray(obs).flatten()

    mask = np.isfinite(sim) & np.isfinite(obs)
    sim, obs = sim[mask], obs[mask]

    if len(sim) < 3:
        return dict(r=np.nan, beta=np.nan, alpha=np.nan,
                    E_ab=np.nan, L_aras=np.nan, kge=np.nan)

    mu_sim    = sim.mean()
    mu_obs    = obs.mean()
    sigma_sim = sim.std()
    sigma_obs = obs.std()

    r     = np.corrcoef(sim, obs)[0, 1]
    beta  = mu_sim  / mu_obs    if mu_obs    != 0 else np.nan
    alpha = sigma_sim / sigma_obs if sigma_obs != 0 else np.nan

    E_ab  = np.sqrt((beta - 1)**2 + (alpha - 1)**2)
    L_aras = np.sqrt((1 - r)**2   + (beta  - 1)**2 + (alpha - 1)**2)
    kge   = 1 - L_aras

    return dict(r=r, beta=beta, alpha=alpha, E_ab=E_ab, L_aras=L_aras, kge=kge)


def compute_genome(model_data: dict, obs_data: dict, variables: list) -> dict:
    """
    Compute the Aras fingerprint across all variables for one model.

    Returns
    -------
    dict  {variable: {r, beta, alpha, E_ab, L_aras, kge}}
    """
    genome = {}
    for var in variables:
        if var not in model_data or var not in obs_data:
            continue
        genome[var] = compute_aras_components(model_data[var], obs_data[var])
    return genome


def genome_distance(genome_a: dict, genome_b: dict) -> float:
    """
    Mean Euclidean distance between two model fingerprints in (r, β, α) space.
    Low distance = shared error structure = models are NOT independent.
    """
    common_vars = set(genome_a.keys()) & set(genome_b.keys())
    if not common_vars:
        return np.nan

    dists = []
    for var in common_vars:
        a = genome_a[var]
        b = genome_b[var]
        if any(np.isnan(a[c]) or np.isnan(b[c]) for c in ['r', 'beta', 'alpha']):
            continue
        d = np.sqrt(
            (a['r']     - b['r']    )**2 +
            (a['beta']  - b['beta'] )**2 +
            (a['alpha'] - b['alpha'])**2
        )
        dists.append(d)

    return float(np.mean(dists)) if dists else np.nan


def genome_matrix(genomes: dict) -> tuple:
    """
    Pairwise fingerprint distance matrix for all models.

    Returns
    -------
    (model_names, NxN distance matrix)
    """
    names = list(genomes.keys())
    n = len(names)
    D = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = genome_distance(genomes[names[i]], genomes[names[j]])
            D[i, j] = d
            D[j, i] = d

    return names, D
