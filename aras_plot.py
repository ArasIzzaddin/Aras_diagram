"""
Aras diagram visualization — single model and genome (multi-model, multi-variable).
"""

import numpy as np
import matplotlib.pyplot as plt


# ── colour palette for model families ──────────────────────────────────────
FAMILY_COLORS = plt.cm.tab20.colors


def plot_aras_diagram(
    genomes: dict,
    variable: str,
    ax=None,
    title: str = None,
    annotate: bool = True,
    error_levels: list = None,
    model_colors: dict = None,
):
    """
    Plot Aras diagram in the exact style of Izzaddin et al. (2024).

    Axes: (beta-1) on x, (alpha-1) on y, centered at origin.
    Each model shown as two points:
      - gold circle  : bias-variability error E_ab = sqrt((beta-1)^2 + (alpha-1)^2)
      - dark triangle: total error E = L_Aras (includes correlation)
    Arrow between them shows correlation error contribution.

    Parameters
    ----------
    genomes      : {model_name: genome_dict}  from aras_core.compute_genome()
    variable     : variable to plot (e.g. 'tas', 'pr')
    ax           : matplotlib Axes (created if None)
    annotate     : label each model
    error_levels : circle radii as fractions (default [0.10, 0.25, 0.50])
    model_colors : {model_name: color}  optional override
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if error_levels is None:
        error_levels = [0.10, 0.25, 0.50]

    # ── Auto-scale range to fit all model points ─────────────────────────────
    all_coords = []
    for genome in genomes.values():
        if variable not in genome:
            continue
        g = genome[variable]
        if any(np.isnan(g.get(k, np.nan)) for k in ['beta', 'alpha', 'r']):
            continue
        bx, ay = g['beta'] - 1, g['alpha'] - 1
        E_ab, L = g['E_ab'], g['L_aras']
        scale = L / E_ab if E_ab > 1e-6 else 1
        all_coords += [abs(bx), abs(ay), abs(bx*scale), abs(ay*scale)]

    LRANGE = max(0.55, min(1.5, max(all_coords) * 1.20)) if all_coords else 0.60

    # ── Style constants (matching the paper) ────────────────────────────────
    GOLD = '#C8860A'

    ax.set_facecolor('white')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-LRANGE, LRANGE)
    ax.set_ylim(-LRANGE, LRANGE)

    # ── Concentric error circles ─────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 500)
    for pct in error_levels:
        if pct > LRANGE:
            continue
        ax.plot(pct * np.cos(theta), pct * np.sin(theta),
                color=GOLD, linewidth=1.4, zorder=2)
        ax.text(0, pct + LRANGE * 0.025, f'{int(pct*100)}% ERROR',
                ha='center', va='bottom', fontsize=8,
                color=GOLD, fontweight='bold')

    # ── Crosshair axes through origin ────────────────────────────────────────
    ax.axhline(0, color='k', linewidth=0.8, zorder=1)
    ax.axvline(0, color='k', linewidth=0.8, zorder=1)

    # ── Quadrant labels ──────────────────────────────────────────────────────
    pad = LRANGE * 0.72
    ax.text(-pad, +pad * 0.6, 'MEAN ↓\nVARIABILITY ↑',
            ha='center', va='center', fontsize=8, color=GOLD,
            fontweight='bold', linespacing=1.6)
    ax.text(+pad, +pad * 0.6, 'MEAN ↑\nVARIABILITY ↑',
            ha='center', va='center', fontsize=8, color=GOLD,
            fontweight='bold', linespacing=1.6)
    ax.text(-pad, -pad * 0.6, 'MEAN ↓\nVARIABILITY ↓',
            ha='center', va='center', fontsize=8, color=GOLD,
            fontweight='bold', linespacing=1.6)
    ax.text(+pad, -pad * 0.6, 'MEAN ↑\nVARIABILITY ↓',
            ha='center', va='center', fontsize=8, color=GOLD,
            fontweight='bold', linespacing=1.6)

    # ── Axis labels and ticks on all four sides (matching paper) ────────────
    ticks = [-0.50, -0.25, 0.00, 0.25, 0.50]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(top=True, right=True, labeltop=True, labelright=True,
                   labelsize=9)

    ax.set_xlabel('BIAS RATIO (β) – 1', fontsize=10, labelpad=8)
    ax.set_ylabel('VARIABILITY RATIO (α) – 1', fontsize=10, labelpad=8)
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_label_position('left')

    # Top and right labels
    ax.set_title('BIAS RATIO (β) – 1', fontsize=10, pad=30)  # top x label

    # ── Model points ─────────────────────────────────────────────────────────
    colors = list(FAMILY_COLORS)
    legend_handles = []

    for idx, (model_name, genome) in enumerate(genomes.items()):
        if variable not in genome:
            continue
        g = genome[variable]
        if any(np.isnan(g.get(k, np.nan)) for k in ['beta', 'alpha', 'r']):
            continue

        bx = g['beta']  - 1   # x in diagram
        ay = g['alpha'] - 1   # y in diagram
        E_ab  = g['E_ab']     # 2D bias-variability distance
        L     = g['L_aras']   # total 3D distance

        color = (model_colors or {}).get(model_name, colors[idx % len(colors)])

        r_val = g['r']

        # Triangle — E_total position along same ray from origin
        if E_ab > 1e-6:
            scale = L / E_ab
            tx, ty = bx * scale, ay * scale
        else:
            tx, ty = bx, ay

        # Line segment first (drawn under markers)
        ax.plot([bx, tx], [ay, ty], color=color, lw=1.8,
                solid_capstyle='round', zorder=5)

        # Circle — E_αβ: filled if r>0, empty if r<0, same model color
        circle_fc = color if r_val >= 0 else 'white'
        ax.scatter(bx, ay, s=90, facecolors=circle_fc,
                   edgecolors=color, linewidths=1.4,
                   zorder=6, marker='o')

        # Triangle — E_total, same model color
        h = ax.scatter(tx, ty, s=110, color=color, edgecolors='k',
                       linewidths=0.7, zorder=6, marker='^')
        legend_handles.append(h)

        if annotate:
            ax.annotate(model_name, (tx, ty),
                        fontsize=6, ha='left', va='bottom',
                        xytext=(4, 4), textcoords='offset points',
                        color=color, fontweight='bold')

    # ── Legend ───────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D4A017',
               markeredgecolor='k', markersize=9,
               label=r'$E_{\alpha\beta}$ (bias-variability error)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='dimgrey',
               markeredgecolor='k', markersize=9,
               label=r'$E$ (total error, incl. correlation)'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=8,
              framealpha=0.9)

    ax.set_title(title or f'Aras Diagram — {variable}', fontsize=12, pad=10)
    return ax


def plot_genome_heatmap(genomes: dict, variables: list, metric: str = 'L_aras'):
    """
    Heatmap of a chosen metric across models × variables.
    Reveals the error fingerprint structure of the ensemble.

    Parameters
    ----------
    genomes   : {model_name: genome_dict}
    variables : list of variable names
    metric    : one of 'L_aras', 'r', 'beta', 'alpha'
    """
    model_names = list(genomes.keys())
    n_models = len(model_names)
    n_vars   = len(variables)

    data = np.full((n_models, n_vars), np.nan)
    for i, mname in enumerate(model_names):
        for j, var in enumerate(variables):
            if var in genomes[mname]:
                data[i, j] = genomes[mname][var].get(metric, np.nan)

    fig, ax = plt.subplots(figsize=(max(8, n_vars * 1.2), max(6, n_models * 0.5)))

    cmap = 'RdYlGn' if metric in ('r', 'kge') else 'RdYlGn_r'
    vmin = 0 if metric == 'L_aras' else (0 if metric == 'r' else 0.5)
    vmax = 1 if metric in ('L_aras', 'r', 'kge') else 1.5

    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(n_vars))
    ax.set_xticklabels(variables, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(model_names, fontsize=8)
    ax.set_title(f'CMIP6 Genome — {metric}', fontsize=13)

    # annotate cells
    for i in range(n_models):
        for j in range(n_vars):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center',
                        fontsize=6, color='k')

    plt.tight_layout()
    return fig, ax


def plot_genome_distance_matrix(names: list, D: np.ndarray):
    """
    Visualise pairwise genome distance matrix.
    Models with low distance are NOT independent — the key result of the paper.
    """
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6),
                                    max(8, len(names) * 0.6)))

    im = ax.imshow(D, cmap='YlOrRd', vmin=0)
    plt.colorbar(im, ax=ax, label='Genome distance')

    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title('CMIP6 Model Genome Distance Matrix\n'
                 '(Low distance = shared error structure = NOT independent)', fontsize=11)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{D[i,j]:.2f}', ha='center', va='center', fontsize=5)

    plt.tight_layout()
    return fig, ax
