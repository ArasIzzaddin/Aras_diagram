"""
Aras diagram visualization.
Izzaddin, A. et al. (2024). A new diagram for performance evaluation
of complex models. Stochastic Environmental Research and Risk Assessment,
38, 2261-2281. https://doi.org/10.1007/s00477-024-02678-3
"""

import numpy as np
import matplotlib.pyplot as plt


FAMILY_COLORS = plt.cm.tab20.colors


def plot_aras_diagram(
    models: dict,
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
      - circle  : E_ab = sqrt((beta-1)^2 + (alpha-1)^2)  [bias-variability error]
                  filled if r > 0, empty if r < 0
      - triangle: E = L_Aras  [total error including correlation]
    Line connecting them represents the correlation error.

    Parameters
    ----------
    models       : {model_name: {variable: components_dict}}
                   components_dict from aras_core.compute_aras_components()
    variable     : key to plot from each model dict (e.g. 'tas', 'pr')
    ax           : matplotlib Axes (created if None)
    annotate     : label each model point
    error_levels : circle radii as fractions (default [0.10, 0.25, 0.50])
    model_colors : {model_name: color}  optional color override

    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if error_levels is None:
        error_levels = [0.10, 0.25, 0.50]

    # ── Auto-scale range to fit all model points ─────────────────────────────
    all_coords = []
    for g_dict in models.values():
        if variable not in g_dict:
            continue
        g = g_dict[variable]
        if any(np.isnan(g.get(k, np.nan)) for k in ['beta', 'alpha', 'r']):
            continue
        bx, ay = g['beta'] - 1, g['alpha'] - 1
        E_ab, L = g['E_ab'], g['L_aras']
        scale = L / E_ab if E_ab > 1e-6 else 1
        all_coords += [abs(bx), abs(ay), abs(bx * scale), abs(ay * scale)]

    LRANGE = max(0.55, min(1.5, max(all_coords) * 1.20)) if all_coords else 0.60
    GOLD   = '#C8860A'

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
        ax.text(0, pct + LRANGE * 0.025, f'{int(pct * 100)}% ERROR',
                ha='center', va='bottom', fontsize=8,
                color=GOLD, fontweight='bold')

    # ── Crosshair ────────────────────────────────────────────────────────────
    ax.axhline(0, color='k', linewidth=0.8, zorder=1)
    ax.axvline(0, color='k', linewidth=0.8, zorder=1)

    # ── Quadrant labels ──────────────────────────────────────────────────────
    pad = LRANGE * 0.72
    for x, y, txt in [
        (-pad, +pad * 0.6, 'MEAN ↓\nVARIABILITY ↑'),
        (+pad, +pad * 0.6, 'MEAN ↑\nVARIABILITY ↑'),
        (-pad, -pad * 0.6, 'MEAN ↓\nVARIABILITY ↓'),
        (+pad, -pad * 0.6, 'MEAN ↑\nVARIABILITY ↓'),
    ]:
        ax.text(x, y, txt, ha='center', va='center', fontsize=8,
                color=GOLD, fontweight='bold', linespacing=1.6)

    # ── Ticks on all four sides ───────────────────────────────────────────────
    ticks = [-0.50, -0.25, 0.00, 0.25, 0.50]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(top=True, right=True, labeltop=True, labelright=True,
                   labelsize=9)
    ax.set_xlabel('BIAS RATIO (β) – 1', fontsize=10, labelpad=8)
    ax.set_ylabel('VARIABILITY RATIO (α) – 1', fontsize=10, labelpad=8)

    # ── Model points ─────────────────────────────────────────────────────────
    colors = list(FAMILY_COLORS)

    for idx, (model_name, g_dict) in enumerate(models.items()):
        if variable not in g_dict:
            continue
        g = g_dict[variable]
        if any(np.isnan(g.get(k, np.nan)) for k in ['beta', 'alpha', 'r']):
            continue

        bx    = g['beta']  - 1
        ay    = g['alpha'] - 1
        E_ab  = g['E_ab']
        L     = g['L_aras']
        r_val = g['r']

        color = (model_colors or {}).get(model_name, colors[idx % len(colors)])

        # Triangle — total error position (same ray, further from origin)
        scale = L / E_ab if E_ab > 1e-6 else 1
        tx, ty = bx * scale, ay * scale

        # Line (drawn first, under markers)
        ax.plot([bx, tx], [ay, ty], color=color, lw=1.8,
                solid_capstyle='round', zorder=5)

        # Circle — filled if r > 0, empty if r < 0
        ax.scatter(bx, ay, s=90,
                   facecolors=color if r_val >= 0 else 'white',
                   edgecolors=color, linewidths=1.4, zorder=6, marker='o')

        # Triangle — total error
        ax.scatter(tx, ty, s=110, color=color, edgecolors='k',
                   linewidths=0.7, zorder=6, marker='^')

        if annotate:
            ax.annotate(model_name, (tx, ty), fontsize=6,
                        ha='left', va='bottom', xytext=(4, 4),
                        textcoords='offset points',
                        color=color, fontweight='bold')

    # ── Legend ───────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=GOLD,
               markeredgecolor='k', markersize=9,
               label=r'$E_{\alpha\beta}$ (bias-variability error)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='dimgrey',
               markeredgecolor='k', markersize=9,
               label=r'$E$ (total error, incl. correlation)'),
    ], loc='lower right', fontsize=8, framealpha=0.9)

    ax.set_title(title or f"Aras' Diagram — {variable}", fontsize=12, pad=10)
    return ax
