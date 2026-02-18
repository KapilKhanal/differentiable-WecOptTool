"""Reusable plotting utilities for Fiacco post-optimality sensitivity.

Provides convenience functions for the most common sensitivity
visualisations. All functions return the Matplotlib ``(fig, axes)`` tuple.

Example
-------
::

    from wecopttool_differentiable import sensitivity, plot_sensitivity_bars

    grad_h = sensitivity(wec, res, waves)
    plot_sensitivity_bars(grad_h)
"""

from __future__ import annotations

__all__ = [
    "plot_sensitivity_bars",
    "plot_frequency_sensitivity",
    "plot_fd_comparison",
]

from typing import Optional, Sequence

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .parametric import BEMParams


_DEFAULT_DISPLAY_NAMES = [
    "Added\nmass",
    "Radiation\ndamping",
    "Hydrostatic\nstiffness",
    "Friction",
    "Froude-\nKrylov",
    "Diffraction",
    "Inertia\nmatrix",
]


def plot_sensitivity_bars(
    grad_h: BEMParams,
    *,
    title: Optional[str] = "Sensitivity of optimal objective to BEM parameters",
    display_names: Optional[Sequence[str]] = None,
    metric: str = "max_abs",
    log_scale: bool = True,
    annotate: bool = True,
    figsize: tuple = (10, 5),
    colors: Optional[Sequence] = None,
    ax: Optional[plt.Axes] = None,
):
    """Bar chart of sensitivity magnitude per BEM parameter field."""
    names = BEMParams._fields
    if display_names is None:
        display_names = _DEFAULT_DISPLAY_NAMES

    if metric == "max_abs":
        vals = [float(jnp.max(jnp.abs(getattr(grad_h, n)))) for n in names]
    elif metric == "norm":
        vals = [float(jnp.linalg.norm(getattr(grad_h, n))) for n in names]
    else:
        raise ValueError(f"metric must be 'max_abs' or 'norm', got {metric!r}")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x = range(len(names))
    if colors is not None:
        bar_colors = colors[:len(names)]
    else:
        cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        bar_colors = [cycle[i % len(cycle)] for i in range(len(names))]

    bars = ax.bar(x, vals, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(list(display_names), fontsize=9)

    ylabel = (r"$\max\,|d\varphi^*/dh_i|$" if metric == "max_abs"
              else r"$\|d\varphi^*/dh_i\|_F$")
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    if annotate:
        for bar, val in zip(bars, vals):
            if val > 0:
                y = bar.get_height() * (1.5 if log_scale else 1.02)
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y,
                    f"{val:.2e}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig, ax


def plot_frequency_sensitivity(
    grad_h: BEMParams,
    omega: np.ndarray,
    *,
    title: Optional[str] = (
        "Per-frequency sensitivity of optimal objective to BEM parameters"),
    wave_freq: Optional[float] = None,
    use_freq_hz: bool = True,
    figsize: tuple = (12, 8),
    colors: Optional[Sequence] = None,
):
    """2×2 subplot of per-frequency sensitivity for the four
    frequency-resolved BEM parameters."""
    omega = np.asarray(omega)
    nfreq = len(omega)

    if use_freq_hz:
        freq = omega / (2 * np.pi)
        xlabel = "Frequency [Hz]"
    else:
        freq = omega
        xlabel = r"$\omega$ [rad/s]"

    cc = colors
    if cc is None:
        cc = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    x = np.arange(nfreq)
    w = 0.35

    ax = axes[0, 0]
    am = grad_h.added_mass
    if am.ndim == 3:
        vals = np.array([float(jnp.linalg.norm(am[i])) for i in range(nfreq)])
    else:
        vals = np.array(jnp.real(am).ravel()[:nfreq])
    ax.bar(x, vals, color=cc[0], edgecolor="black", linewidth=0.5)
    ax.set_title(r"$d\varphi^*/dA(\omega)$  (Added mass)")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0, 1]
    rd = grad_h.radiation_damping
    if rd.ndim == 3:
        vals = np.array([float(jnp.linalg.norm(rd[i])) for i in range(nfreq)])
    else:
        vals = np.array(jnp.real(rd).ravel()[:nfreq])
    ax.bar(x, vals, color=cc[1], edgecolor="black", linewidth=0.5)
    ax.set_title(r"$d\varphi^*/dB(\omega)$  (Radiation damping)")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 0]
    fk = grad_h.Froude_Krylov_force
    if fk.ndim == 3:
        re = np.array([float(jnp.max(jnp.abs(jnp.real(fk[i])))) for i in range(nfreq)])
        im = np.array([float(jnp.max(jnp.abs(jnp.imag(fk[i])))) for i in range(nfreq)])
    else:
        re = np.array(jnp.real(fk).ravel()[:nfreq])
        im = np.array(jnp.imag(fk).ravel()[:nfreq])
    ax.bar(x - w / 2, re, w, label="Real", color=cc[2], edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, im, w, label="Imag", color=cc[3], edgecolor="black", linewidth=0.5)
    ax.set_title(r"$d\varphi^*/dF_K(\omega)$  (Froude-Krylov)")
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1, 1]
    df = grad_h.diffraction_force
    if df.ndim == 3:
        re = np.array([float(jnp.max(jnp.abs(jnp.real(df[i])))) for i in range(nfreq)])
        im = np.array([float(jnp.max(jnp.abs(jnp.imag(df[i])))) for i in range(nfreq)])
    else:
        re = np.array(jnp.real(df).ravel()[:nfreq])
        im = np.array(jnp.imag(df).ravel()[:nfreq])
    ax.bar(x - w / 2, re, w, label="Real", color=cc[4 % len(cc)],
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, im, w, label="Imag", color=cc[5 % len(cc)],
           edgecolor="black", linewidth=0.5)
    ax.set_title(r"$d\varphi^*/dF_D(\omega)$  (Diffraction)")
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    if wave_freq is not None:
        for ax in axes.flat:
            ax.axvline(
                wave_freq if not use_freq_hz else wave_freq,
                color="red", linestyle="--", alpha=0.7,
                label=f"Wave freq ({wave_freq:.3g} Hz)")
            ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_fd_comparison(
    analytical: Sequence[float],
    fd: Sequence[float],
    names: Sequence[str],
    *,
    title: Optional[str] = (
        "Analytical vs. Finite-Difference Gradient"),
    show_relative_error: bool = True,
    error_threshold: float = 0.05,
    figsize: tuple = (12, 5),
    colors: Optional[tuple] = None,
):
    """Side-by-side bar chart comparing analytical and finite-difference
    gradients, with an optional relative-error panel."""
    analytical = np.asarray(analytical, dtype=float)
    fd = np.asarray(fd, dtype=float)

    if colors is None:
        cc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        c_anal, c_fd = cc[0], cc[1]
    else:
        c_anal, c_fd = colors[0], colors[1]

    display = [n.replace("_", "\n") for n in names]
    x = np.arange(len(names))
    w = 0.35

    if show_relative_error:
        fig, axes = plt.subplots(
            2, 1, figsize=figsize,
            gridspec_kw={"height_ratios": [3, 1]})
        ax_top = axes[0]
        ax_bot = axes[1]
    else:
        fig, ax_top = plt.subplots(figsize=figsize)
        axes = ax_top

    ax_top.bar(x - w / 2, analytical, w,
               label="Analytical (Fiacco VJP)",
               color=c_anal, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax_top.bar(x + w / 2, fd, w,
               label="Finite difference (NLP re-solve)",
               color=c_fd, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(display, fontsize=9)
    ax_top.set_ylabel(r"$d\varphi^*/dh_i$", fontsize=12)
    if title:
        ax_top.set_title(title, fontsize=14)
    ax_top.legend(fontsize=10)
    ax_top.axhline(0, color="gray", linewidth=0.5)
    ax_top.grid(axis="y", alpha=0.3)

    if show_relative_error:
        denom = np.where(np.abs(analytical) > 1e-15, np.abs(analytical), 1.0)
        rel_err = np.abs(fd - analytical) / denom
        bar_colors = [
            "C2" if e < error_threshold
            else ("C8" if e < 6 * error_threshold else "C3")
            for e in rel_err
        ]
        ax_bot.bar(x, rel_err, color=bar_colors,
                   edgecolor="black", linewidth=0.5, alpha=0.85)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(display, fontsize=9)
        ax_bot.set_ylabel("Relative error", fontsize=11)
        ax_bot.set_yscale("log")
        ax_bot.axhline(error_threshold, color="gray", linestyle="--",
                       linewidth=0.8, label=f"{error_threshold:.0%} threshold")
        ax_bot.legend(fontsize=9)
        ax_bot.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig, axes
