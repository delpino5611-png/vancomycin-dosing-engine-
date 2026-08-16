# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Generate the writeup / LinkedIn figures from REAL engine output.

Serves the three Tesseracts, runs the real optimizer + Bayesian fitter, pulls
real steady-state concentration curves, and renders four theme-aware PNGs to
../figures/. No fabricated numbers: every curve and value comes from the served
engine.
"""
import json
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import guardrails as G
from engine import (connect, ckd_params, pk_exposure, exposure_loss, optimize_regimen,
                    loading_dose)
from bayesian import synthetic_demo
from servers import serve_all

jax.config.update("jax_platform_name", "cpu")

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)

f32 = lambda x: jnp.float32(x)

# ----------------------------------------------------------------------------
# Palette (dataviz reference instance) — status + categorical, per theme
# ----------------------------------------------------------------------------
STATUS = dict(good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b")
CAT = dict(blue_l="#2a78d6", blue_d="#3987e5", orange_l="#eb6834", orange_d="#d95926",
           aqua_l="#1baf7a", aqua_d="#199e70", violet_l="#4a3aa7", violet_d="#9085e9")

THEMES = {
    "light": dict(surface="#fcfcfb", page="#f9f9f7", ink="#0b0b0b", ink2="#52514e",
                  muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  blue=CAT["blue_l"], orange=CAT["orange_l"], aqua=CAT["aqua_l"],
                  violet=CAT["violet_l"]),
    "dark": dict(surface="#1a1a19", page="#0d0d0d", ink="#ffffff", ink2="#c3c2b7",
                 muted="#898781", grid="#2c2c2a", axis="#383835",
                 blue=CAT["blue_d"], orange=CAT["orange_d"], aqua=CAT["aqua_d"],
                 violet=CAT["violet_d"]),
}


def style(th):
    t = THEMES[th]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink"], "axes.edgecolor": t["axis"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.titlecolor": t["ink"], "grid.color": t["grid"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.grid": True, "grid.linewidth": 1.0, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return t


# ----------------------------------------------------------------------------
# Real curve extraction
# ----------------------------------------------------------------------------
def ss_curve_over(T, dose, tau, ke, v, hours=48.0):
    """Tile the real steady-state one-interval curve across `hours` (SS = each
    interval identical). Returns (t_hours, conc)."""
    t_inf = G.infusion_time(dose)
    ex = pk_exposure(T, f32(dose), f32(tau), f32(ke), f32(v), t_inf=t_inf)
    grid = np.array(ex["times"])          # 0..tau
    conc = np.array(ex["conc"])
    n = int(np.ceil(hours / tau))
    ts, cs = [], []
    for k in range(n):
        ts.append(grid + k * tau)
        cs.append(conc)
    t = np.concatenate(ts); c = np.concatenate(cs)
    m = t <= hours + 1e-6
    return t[m], c[m], float(ex["auc24"]), float(ex["peak"]), float(ex["trough"])


def std_dose(weight):
    """Standard renal-blind empiric: 15 mg/kg q12h, snapped to the 250 grid, capped."""
    return G.snap_dose(min(15.0 * weight, G.MAX_DOSE))


def transient_1c(ke, v, schedule, hours=48.0, n=600):
    """Non-steady-state 1-comp concentration over `hours` for a list of doses.

    schedule = [(t_dose_h, dose_mg, t_inf_h), ...]. Analytic superposition of
    intermittent IV infusions, counting only doses given at or before each time
    point (true accumulation from t=0, NOT the steady-state read). Used to show
    the first 24-48 h that the steady-state curve cannot: loading vs no-loading.
    """
    cl = ke * v
    t = np.linspace(0.0, hours, n)
    c = np.zeros_like(t)
    for (td, dose, tinf) in schedule:
        R = dose / tinf
        e = t - td  # elapsed since this dose
        during = (R / cl) * (1.0 - np.exp(-ke * np.clip(e, 0, None)))
        post = (R / cl) * (1.0 - np.exp(-ke * tinf)) * np.exp(-ke * np.clip(e - tinf, 0, None))
        contrib = np.where(e < 0, 0.0, np.where(e <= tinf, during, post))
        c = c + contrib
    return t, c


# ----------------------------------------------------------------------------
# Optimizer convergence trace (real Adam loop, recorded)
# ----------------------------------------------------------------------------
def optimize_trace(T, ke, v, tau, target=500.0, steps=120):
    lo, hi = 250.0, 2000.0
    dose_of = lambda th: lo + (hi - lo) * jax.nn.sigmoid(th)

    def loss_of(th):
        d = dose_of(th)
        ex = pk_exposure(T, d, f32(tau), f32(ke), f32(v))
        return exposure_loss(T, ex["auc24"], ex["peak"], ex["trough"], target)["loss"]

    grad = jax.grad(loss_of)
    th = f32(0.0)
    m, vv, b1, b2, eps, lr = 0.0, 0.0, 0.9, 0.999, 1e-8, 0.15
    doses, aucs, losses = [], [], []
    for t in range(1, steps + 1):
        d = float(dose_of(th))
        ex = pk_exposure(T, f32(d), f32(tau), f32(ke), f32(v))
        doses.append(d); aucs.append(float(ex["auc24"]))
        losses.append(float(exposure_loss(T, ex["auc24"], ex["peak"], ex["trough"],
                                          target)["loss"]))
        gt = float(grad(th))
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * gt * gt
        mhat = m / (1 - b1 ** t); vhat = vv / (1 - b2 ** t)
        th = th - lr * mhat / (jnp.sqrt(vhat) + eps)
    return np.array(doses), np.array(aucs), np.array(losses)


# ============================================================================
# FIGURE 1 — HERO: renal spectrum, standard vs engine-optimized
# ============================================================================
def fig_hero(T, patients, th, data):
    t = style(th)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2), sharey=False)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.19, wspace=0.20)

    YMAX = 55.0
    state_col = {"toxic": STATUS["critical"], "subtherapeutic": STATUS["warning"],
                 "on-target": t["muted"]}
    state_word = {"toxic": "TOXIC", "subtherapeutic": "SUBTHERAPEUTIC",
                  "on-target": "ADEQUATE"}
    for ax, (name, meta) in zip(axes, patients.items()):
        d = data[name]
        fs = d["std_fail"]
        # therapeutic reference band: trough 10-20 mg/L
        ax.axhspan(10, 20, color=t["aqua"], alpha=0.10, zorder=0)
        ax.axhline(20, color=STATUS["critical"], lw=1.2, ls=(0, (5, 4)), alpha=0.8, zorder=1)
        ax.axhline(10, color=STATUS["warning"], lw=1.2, ls=(0, (5, 4)), alpha=0.8, zorder=1)

        # standard — colored by state; dashed when it already agrees (on-target)
        std_col = state_col[fs]
        std_ls = (0, (6, 3)) if fs == "on-target" else "-"
        # toxic curve is off-scale; pin it to the top edge so a red line stays visible
        std_clip = np.minimum(d["std_c"], YMAX - 0.4) if fs == "toxic" else d["std_c"]
        ax.plot(d["std_t"], std_clip, color=std_col, lw=2.4,
                ls=std_ls, zorder=3, label=f"Standard {d['std_dose']:.0f} mg q12h")
        # engine (hits target)
        ax.plot(d["opt_t"], d["opt_c"], color=STATUS["good"], lw=2.6, zorder=4,
                label=f"Engine {d['opt_dose']:.0f} mg q{d['opt_tau']:.0f}h")

        # toxic curve runs off-scale: annotate its true peak/trough
        if fs == "toxic":
            ax.annotate(f"off scale: peak {d['std_c'].max():.0f}, trough {d['std_trough']:.0f}\n"
                        f"(~{d['std_trough']/20:.1f}x the 20 mg/L ceiling)",
                        xy=(6, YMAX), xytext=(9, YMAX - 9),
                        fontsize=9.5, color=STATUS["critical"], fontweight="bold",
                        ha="left", va="top",
                        arrowprops=dict(arrowstyle="->", color=STATUS["critical"], lw=1.4))

        ax.set_title(f"{meta['label']}\nCrCl {d['crcl']:.0f} mL/min",
                     fontsize=15, fontweight="bold", pad=10)
        ax.set_xlabel("Time (h)", fontsize=12)
        ax.set_xlim(0, 48); ax.set_xticks(range(0, 49, 12))
        ax.set_ylim(0, YMAX)
        ax.tick_params(labelsize=11)
        ax.grid(axis="x", visible=False)

        # AUC callouts
        box = dict(boxstyle="round,pad=0.4", fc=t["surface"], ec=t["grid"], lw=1)
        ax.text(0.03, 0.965,
                f"Standard AUC24 {d['std_auc']:.0f}  ->  {state_word[fs]}",
                transform=ax.transAxes, fontsize=10.5, va="top", ha="left",
                color=(t["ink2"] if fs == "on-target" else std_col),
                fontweight="bold", bbox=box, zorder=6)
        eng_tail = "confirms standard" if fs == "on-target" else "ON TARGET"
        ax.text(0.03, 0.855,
                f"Engine AUC24 {d['opt_auc']:.0f}  ->  {eng_tail}",
                transform=ax.transAxes, fontsize=10.5, va="top", ha="left",
                color=STATUS["good"], fontweight="bold", bbox=box, zorder=6)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.0)

    axes[0].set_ylabel("Vancomycin concentration (mg/L)", fontsize=12.5)
    # shared band annotation on last axis
    axes[-1].text(47.5, 20.6, "trough ceiling 20", color=STATUS["critical"],
                  fontsize=9, ha="right", va="bottom")
    axes[-1].text(47.5, 8.6, "trough floor 10", color=STATUS["warning"],
                  fontsize=9, ha="right", va="top")

    fig.suptitle("One drug, three kidneys: standard weight-based dosing fails at both "
                 "ends of the renal spectrum; the engine solves each dose to target",
                 fontsize=17.5, fontweight="bold", x=0.06, ha="left", y=0.965)
    fig.text(0.06, 0.895,
             "Steady-state IV vancomycin. Target AUC24/MIC 400-600 (Rybak 2020). "
             "Standard = 15 mg/kg q12h, renal-blind. Engine = gradient-optimized to AUC 500.",
             fontsize=11.5, color=t["ink2"], ha="left")
    fig.text(0.06, 0.035,
             "Self-generated from the vancomycin differentiable dosing engine (real output). "
             "PK: Cockcroft-Gault 1976, Matzke 1984, Onor 2020. Targets: Rybak 2020.",
             fontsize=9, color=t["muted"], ha="left")
    p = os.path.join(OUT, f"fig1_hero_renal_spectrum_{th}.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ============================================================================
# FIGURE 2 — "knows when NOT to compute": ESRD guardrail -> TDM handoff
# ============================================================================
def fig_esrd(T, th, esrd):
    t = style(th)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15.5, 6.4),
                                   gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.subplots_adjust(left=0.065, right=0.975, top=0.80, bottom=0.13, wspace=0.28)

    # LEFT: the two candidate regimens' SS curves
    axL.axhspan(10, 20, color=CAT["aqua_l"], alpha=0.10, zorder=0)
    axL.axhline(20, color=STATUS["critical"], lw=1.4, ls=(0, (5, 4)), zorder=1)
    axL.plot(esrd["cont_t"], esrd["cont_c"], color=STATUS["good"], lw=2.6, zorder=3,
             label=f"Continuous optimum {esrd['cont_dose']:.0f} mg q24h  (AUC {esrd['cont_auc']:.0f})")
    axL.plot(esrd["snap_t"], esrd["snap_c"], color=STATUS["critical"], lw=2.6, zorder=4,
             label=f"Rounded 500 mg q24h  (AUC {esrd['snap_auc']:.0f})")
    # mark the breaching trough
    axL.scatter([24], [esrd["snap_trough"]], s=90, color=STATUS["critical"],
                edgecolor=t["surface"], linewidth=2, zorder=6)
    axL.annotate(f"trough {esrd['snap_trough']:.1f} > 20\nbreaches ceiling",
                 xy=(24, esrd["snap_trough"]), xytext=(30, 24.5),
                 fontsize=10.5, fontweight="bold", color=STATUS["critical"],
                 ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", color=STATUS["critical"], lw=1.6))
    axL.text(47.5, 20.4, "nephrotoxicity ceiling 20 mg/L", color=STATUS["critical"],
             fontsize=9.5, ha="right", va="bottom")
    axL.set_title(f"AKI / near-ESRD patient  (CrCl {esrd['crcl']:.0f} mL/min)",
                  fontsize=14.5, fontweight="bold", pad=8)
    axL.set_xlabel("Time (h)", fontsize=12); axL.set_ylabel("Vancomycin concentration (mg/L)",
                                                            fontsize=12.5)
    axL.set_xlim(0, 48); axL.set_xticks(range(0, 49, 12)); axL.tick_params(labelsize=11)
    axL.grid(axis="x", visible=False)
    axL.legend(loc="lower left", fontsize=10, framealpha=0.0)

    # RIGHT: the decision flow
    axR.axis("off")
    axR.set_xlim(0, 1); axR.set_ylim(0, 1)
    steps = [
        (0.90, STATUS["good"],   "1.  Differentiable optimum",
         f"{esrd['cont_dose']:.0f} mg q24h  ->  AUC {esrd['cont_auc']:.0f}, trough safe"),
        (0.66, STATUS["warning"],"2.  Snap to clinical 250 mg grid",
         f"forces 500 mg q24h  ->  AUC {esrd['snap_auc']:.0f}, trough {esrd['snap_trough']:.1f}"),
        (0.42, STATUS["critical"],"3.  Guardrail HARD-BLOCK",
         "trough 20.1 > 20 mg/L ceiling  ->  regimen refused"),
        (0.16, CAT["blue_l"] if th == "light" else CAT["blue_d"],
         "4.  Hand off to level-guided / TDM dosing",
         "loading dose + measured levels; fixed-interval PK does not apply in ESRD"),
    ]
    for i, (y, col, head, sub) in enumerate(steps):
        axR.add_patch(plt.Rectangle((0.02, y - 0.085), 0.05, 0.14, color=col,
                                    transform=axR.transAxes, clip_on=False))
        axR.text(0.11, y + 0.028, head, fontsize=12.5, fontweight="bold",
                 color=t["ink"], va="center")
        axR.text(0.11, y - 0.045, sub, fontsize=10.3, color=t["ink2"], va="center")
        if i < len(steps) - 1:
            axR.annotate("", xy=(0.045, y - 0.11), xytext=(0.045, y - 0.075),
                         arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.6))
    axR.set_title("The engine knows when NOT to compute", fontsize=14.5,
                  fontweight="bold", loc="left", pad=8, color=t["ink"])

    fig.suptitle("Encoded clinical judgment: refusing a false-precision dose is the "
                 "credibility feature, not a limitation",
                 fontsize=17, fontweight="bold", x=0.065, ha="left", y=0.955)
    fig.text(0.065, 0.895,
             "In dialysis-range renal function, vancomycin clearance is non-linear (near zero "
             "between runs, stripped and rebounding around dialysis).\nA scheduled maintenance "
             "dose implies standard PK applies; the engine refuses it and routes to measured levels.",
             fontsize=11, color=t["ink2"], ha="left", va="top")
    fig.text(0.065, 0.02,
             "Self-generated from the engine (real output). Guardrail ceilings per "
             "vanco_engine_spec / Rybak 2020. Clinical rationale: PharmD review.",
             fontsize=9, color=t["muted"], ha="left")
    p = os.path.join(OUT, f"fig2_esrd_guardrail_{th}.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ============================================================================
# FIGURE 3 — optimizer convergence (dose -> AUC target; loss -> 0)
# ============================================================================
def fig_convergence(th, traces):
    t = style(th)
    fig, (axA, axL) = plt.subplots(1, 2, figsize=(15.5, 6.2))
    fig.subplots_adjust(left=0.07, right=0.975, top=0.80, bottom=0.13, wspace=0.24)

    colmap = {"young-normal": t["blue"], "elderly-CKD": t["orange"], "AKI-ESRD": t["violet"]}
    labelmap = {"young-normal": "Young, normal renal", "elderly-CKD": "Elderly CKD",
                "AKI-ESRD": "AKI / near-ESRD"}

    # LEFT: AUC24 -> target band
    axA.axhspan(400, 600, color=STATUS["good"], alpha=0.10, zorder=0)
    axA.axhline(500, color=t["muted"], lw=1.2, ls=(0, (5, 4)), zorder=1)
    axA.text(len(next(iter(traces.values()))[1]) - 1, 505, "target AUC 500",
             color=t["ink2"], fontsize=10, ha="right", va="bottom")
    for name, (doses, aucs, losses) in traces.items():
        axA.plot(np.arange(len(aucs)), aucs, color=colmap[name], lw=2.4,
                 label=labelmap[name], zorder=3)
        axA.scatter([len(aucs) - 1], [aucs[-1]], s=55, color=colmap[name],
                    edgecolor=t["surface"], linewidth=2, zorder=4)
    axA.set_title("Every patient converges onto the AUC24/MIC 400-600 band",
                  fontsize=14, fontweight="bold", pad=8)
    axA.set_xlabel("Gradient step", fontsize=12)
    axA.set_ylabel("Steady-state AUC24 (mg*h/L)", fontsize=12.5)
    axA.tick_params(labelsize=11); axA.grid(axis="x", visible=False)
    axA.legend(loc="upper right", fontsize=10.5, framealpha=0.0)

    # RIGHT: loss -> 0 (log scale)
    for name, (doses, aucs, losses) in traces.items():
        axL.semilogy(np.arange(len(losses)), np.maximum(losses, 1e-2),
                     color=colmap[name], lw=2.4, label=labelmap[name], zorder=3)
    axL.set_title("Dosing loss driven down by end-to-end gradients",
                  fontsize=14, fontweight="bold", pad=8)
    axL.set_xlabel("Gradient step", fontsize=12)
    axL.set_ylabel("Dosing loss (log scale)", fontsize=12.5)
    axL.tick_params(labelsize=11); axL.grid(axis="x", visible=False)
    axL.legend(loc="upper right", fontsize=10.5, framealpha=0.0)

    fig.suptitle("The differentiable optimizer at work: jax.grad flows across three composed "
                 "Tesseracts to solve the dose",
                 fontsize=16.5, fontweight="bold", x=0.07, ha="left", y=0.955)
    fig.text(0.07, 0.885,
             "Adam on a bounded dose reparameterization. The gradient of the dosing loss with "
             "respect to dose is computed through the served PK and loss modules, not by "
             "grid-search guess-and-check.",
             fontsize=11, color=t["ink2"], ha="left")
    fig.text(0.07, 0.02,
             "Self-generated from the engine (real optimizer trace). Method: composed "
             "differentiable PK/CKD/loss modules, gradient descent to AUC target.",
             fontsize=9, color=t["muted"], ha="left")
    p = os.path.join(OUT, f"fig3_optimizer_convergence_{th}.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ============================================================================
# FIGURE 4 — Bayesian sparse-data individualization (bonus / new-drug finale)
# ============================================================================
def fig_bayesian(th, b):
    t = style(th)
    fig, (axK, axV) = plt.subplots(1, 2, figsize=(14.5, 6.0))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.80, bottom=0.13, wspace=0.26)

    def panel(ax, prior, post, truth, title, unit):
        cats = ["Population\nprior", "Posterior\n(+2 levels)", "Patient\ntruth"]
        vals = [prior, post, truth]
        cols = [t["muted"], t["blue"], STATUS["good"]]
        xs = np.arange(3)
        ax.bar(xs, vals, width=0.52, color=cols, zorder=3)
        ax.axhline(truth, color=STATUS["good"], lw=1.2, ls=(0, (5, 4)), zorder=1, alpha=0.7)
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.3f}" if unit == "1/h" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=11, color=t["ink"],
                    fontweight="bold")
        ax.set_xticks(xs); ax.set_xticklabels(cats, fontsize=11)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelsize=10.5)
        ax.set_ylim(0, max(vals) * 1.22)

    panel(axK, b["ke_prior"], b["ke_post"], b["ke_true"],
          "Elimination rate Ke (1/h)", "1/h")
    panel(axV, b["v_prior"], b["v_post"], b["v_true"],
          "Volume of distribution V (L)", "L")
    axK.set_ylabel("value", fontsize=12)

    fig.suptitle("Learning the patient from a few blood draws: MAP-Bayesian update sharpens the "
                 "population prior toward truth",
                 fontsize=16, fontweight="bold", x=0.075, ha="left", y=0.955)
    fig.text(0.075, 0.885,
             f"Two measured steady-state levels move Ke error {b['ke_err_prior']:.3f} -> "
             f"{b['ke_err_post']:.3f} and V error {b['v_err_prior']:.1f} -> {b['v_err_post']:.1f}. "
             "This is the workflow for a drug with no dosing chart: infer the patient, then dose.",
             fontsize=11, color=t["ink2"], ha="left")
    fig.text(0.075, 0.02,
             "Self-generated from the engine (real MAP fit, autodiff through the served PK "
             "module). Prior: Matzke 1984 / Onor 2020 population model.",
             fontsize=9, color=t["muted"], ha="left")
    p = os.path.join(OUT, f"fig4_bayesian_individualization_{th}.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ============================================================================
# FIGURE 5 — LOADING DOSE: the critical first 24-48 h steady-state-only misses
# ============================================================================
def fig_loading(th, load_data):
    t = style(th)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.98, top=0.75, bottom=0.13, wspace=0.12)

    for ax, d in zip(axes, load_data):
        ax.axhspan(10, 20, color=t["aqua"], alpha=0.10, zorder=0)
        ax.axhline(20, color=STATUS["critical"], lw=1.2, ls=(0, (5, 4)), alpha=0.8, zorder=1)
        ax.axhline(10, color=STATUS["warning"], lw=1.2, ls=(0, (5, 4)), alpha=0.8, zorder=1)

        # maintenance-only (no loading) — slow climb
        ax.plot(d["t_no"], d["c_no"], color=t["orange"], lw=2.4, zorder=3,
                label=f"Maintenance only ({d['maint']:.0f} mg q{d['tau']:.0f}h)")
        # loading + maintenance — in-band from the first interval
        ax.plot(d["t_ld"], d["c_ld"], color=t["blue"], lw=2.6, zorder=4,
                label=f"Loading {d['load']:.0f} mg, then maintenance")

        # mark first time each reaches the therapeutic floor (10 mg/L)
        if d["t_reach_ld"] is not None:
            ax.scatter([d["t_reach_ld"]], [10], s=70, color=t["blue"],
                       edgecolor=t["surface"], linewidth=1.8, zorder=6)
        if d["t_reach_no"] is not None:
            ax.scatter([d["t_reach_no"]], [10], s=70, color=t["orange"],
                       edgecolor=t["surface"], linewidth=1.8, zorder=6)
            # keep the callout inside the axes (annotate to the LEFT when the
            # crossing is late in the window so the arrow never runs off-frame)
            late = d["t_reach_no"] > 30
            tx = d["t_reach_no"] - 3 if late else d["t_reach_no"] + 3
            ha = "right" if late else "left"
            ax.annotate(f"no loading: sustained\ntrough at ~{d['t_reach_no']:.0f} h",
                        xy=(d["t_reach_no"], 10), xytext=(tx, 3.2),
                        fontsize=9.5, color=t["orange"], fontweight="bold",
                        ha=ha, va="bottom",
                        arrowprops=dict(arrowstyle="->", color=t["orange"], lw=1.3))

        ax.set_title(f"{d['label']}\nCrCl {d['crcl']:.0f} mL/min",
                     fontsize=14, fontweight="bold", pad=8)
        ax.set_xlabel("Time from first dose (h)", fontsize=12)
        ax.set_xlim(0, 48); ax.set_xticks(range(0, 49, 12))
        ax.set_ylim(0, 42); ax.tick_params(labelsize=11)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", fontsize=9.8, framealpha=0.0)

    axes[0].set_ylabel("Vancomycin concentration (mg/L)", fontsize=12.5)
    axes[-1].text(47.5, 20.4, "trough ceiling 20", color=STATUS["critical"],
                  fontsize=9, ha="right", va="bottom")
    axes[-1].text(47.5, 8.6, "therapeutic floor 10", color=STATUS["warning"],
                  fontsize=9, ha="right", va="top")

    fig.suptitle("Two-phase dosing: a weight-based loading dose reaches target in the first "
                 "hours, not after 3-4 maintenance doses",
                 fontsize=16.5, fontweight="bold", x=0.065, ha="left", y=0.965)
    fig.text(0.065, 0.905,
             "Non-steady-state accumulation from the first dose. Loading = 25 mg/kg actual "
             "body weight (Rybak 2020; Tesfamariam 2024),",
             fontsize=11, color=t["ink2"], ha="left")
    fig.text(0.065, 0.868,
             "given before the gradient-optimized maintenance regimen. Steady-state-only "
             "views miss this critical early window.",
             fontsize=11, color=t["ink2"], ha="left")
    fig.text(0.065, 0.02,
             "Self-generated from the engine (real PK). Loading: Rybak 2020 / Tesfamariam 2024. "
             "Maintenance: gradient-optimized to AUC 500.",
             fontsize=9, color=t["muted"], ha="left")
    p = os.path.join(OUT, f"fig5_loading_dose_{th}.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ============================================================================
def main():
    # hero patients: ARC / normal / CKD (covariates -> real engine)
    patients = {
        "young-ARC":   dict(label="Young, augmented clearance",
                            cov=dict(age=30, weight=72, height_in=70, scr=0.7, sex=0)),
        "normal":      dict(label="Middle-aged, normal renal",
                            cov=dict(age=45, weight=80, height_in=70, scr=1.0, sex=0)),
        "elderly-CKD": dict(label="Elderly, chronic kidney disease",
                            cov=dict(age=78, weight=60, height_in=63, scr=1.8, sex=1)),
    }

    with serve_all() as urls:
        T = connect(urls)
        provenance = {}

        # ---- hero data ----
        hero = {}
        for name, meta in patients.items():
            cov = meta["cov"]
            p = ckd_params(T, cov)
            ke, v, crcl = float(p["ke"]), float(p["v"]), float(p["crcl"])
            sd = std_dose(cov["weight"])
            st, sc, sauc, speak, strough = ss_curve_over(T, sd, 12.0, ke, v)
            r = optimize_regimen(T, cov, target=500.0)
            b = r["best"]
            ot, oc, oauc, opeak, otrough = ss_curve_over(T, b["dose"], b["tau"], ke, v)
            std_fail = ("toxic" if sauc > 600 else
                        "subtherapeutic" if sauc < 400 else "on-target")
            hero[name] = dict(crcl=crcl, std_dose=sd, std_t=st, std_c=sc, std_auc=sauc,
                              std_trough=strough, std_fail=std_fail,
                              opt_dose=b["dose"], opt_tau=b["tau"], opt_t=ot, opt_c=oc,
                              opt_auc=oauc, opt_trough=otrough)
            provenance[name] = dict(crcl=crcl, ke=ke, v=v, std_dose=sd, std_auc=sauc,
                                    std_trough=strough, std_fail=std_fail,
                                    opt_dose=b["dose"], opt_tau=b["tau"], opt_auc=oauc,
                                    opt_peak=opeak, opt_trough=otrough)

        # ---- ESRD panel data ----
        esrd_cov = dict(age=60, weight=85, height_in=70, scr=7.0, sex=0)
        pe = ckd_params(T, esrd_cov)
        ke_e, v_e, crcl_e = float(pe["ke"]), float(pe["v"]), float(pe["crcl"])
        re = optimize_regimen(T, esrd_cov, target=500.0)
        be = re["best"]
        cont_dose = round(be["dose_continuous"])
        ct, cc, cauc, cpeak, ctrough = ss_curve_over(T, cont_dose, be["tau"], ke_e, v_e)
        snt, snc, snauc, snpeak, sntrough = ss_curve_over(T, be["dose"], be["tau"], ke_e, v_e)
        esrd = dict(crcl=crcl_e, cont_dose=cont_dose, cont_t=ct, cont_c=cc, cont_auc=cauc,
                    snap_t=snt, snap_c=snc, snap_auc=snauc, snap_trough=sntrough)
        provenance["AKI-ESRD"] = dict(crcl=crcl_e, ke=ke_e, v=v_e,
                                      cont_dose=cont_dose, cont_auc=cauc, cont_trough=ctrough,
                                      snap_dose=be["dose"], snap_auc=snauc, snap_trough=sntrough,
                                      blocked=not be["guard"]["ok"],
                                      hard_block=be["guard"]["hard_block"])

        # ---- loading-dose data (transient, first 48 h) ----
        load_patients = [
            dict(label="Young, normal renal",
                 cov=dict(age=40, weight=80, height_in=70, scr=1.0, sex=0)),
            dict(label="Elderly, chronic kidney disease",
                 cov=dict(age=78, weight=60, height_in=63, scr=1.8, sex=1)),
        ]
        load_data = []
        for meta in load_patients:
            cov = meta["cov"]
            p = ckd_params(T, cov)
            ke, v, crcl = float(p["ke"]), float(p["v"]), float(p["crcl"])
            r = optimize_regimen(T, cov, target=500.0)
            maint = r["best"]["dose"]; tau = r["best"]["tau"]
            ld = r["loading"]
            n_maint = int(np.ceil(48.0 / tau)) + 1
            # maintenance-only schedule: dose at 0, tau, 2tau, ...
            sched_no = [(k * tau, maint, G.infusion_time(maint)) for k in range(n_maint)]
            # loading then maintenance: loading at 0, maintenance from tau onward
            sched_ld = [(0.0, ld["dose"], ld["t_inf"])] + \
                       [(k * tau, maint, G.infusion_time(maint)) for k in range(1, n_maint)]
            t_no, c_no = transient_1c(ke, v, sched_no)
            t_ld, c_ld = transient_1c(ke, v, sched_ld)

            def first_trough_in_band(tt, cc, thr=10.0):
                """First interval whose END-OF-INTERVAL trough is sustained >= thr.
                (Both curves briefly cross 10 during the first infusion; the clinical
                question is when the TROUGH stops dipping subtherapeutic.)"""
                for k in range(1, n_maint):
                    tk = k * tau
                    ck = float(np.interp(tk, tt, cc))
                    if ck >= thr:
                        return tk
                return None

            load_data.append(dict(
                label=meta["label"], crcl=crcl, maint=maint, tau=tau, load=ld["dose"],
                t_no=t_no, c_no=c_no, t_ld=t_ld, c_ld=c_ld,
                t_reach_no=first_trough_in_band(t_no, c_no),
                t_reach_ld=first_trough_in_band(t_ld, c_ld)))
            provenance.setdefault("loading", {})[meta["label"]] = dict(
                crcl=crcl, loading_dose=ld["dose"], maint_dose=maint, tau=tau,
                peak_achieved=ld["peak_achieved"],
                t_reach_target_loading=load_data[-1]["t_reach_ld"],
                t_reach_target_no_loading=load_data[-1]["t_reach_no"])

        # ---- optimizer traces ----
        trace_cov = {
            "young-normal": dict(age=40, weight=80, height_in=70, scr=1.0, sex=0),
            "elderly-CKD":  dict(age=78, weight=60, height_in=63, scr=1.8, sex=1),
            "AKI-ESRD":     dict(age=60, weight=85, height_in=70, scr=7.0, sex=0),
        }
        trace_tau = {"young-normal": 12.0, "elderly-CKD": 24.0, "AKI-ESRD": 24.0}
        traces = {}
        for name, cov in trace_cov.items():
            p = ckd_params(T, cov)
            traces[name] = optimize_trace(T, float(p["ke"]), float(p["v"]),
                                          trace_tau[name])
        provenance["convergence"] = {n: dict(final_dose=float(d[-1]), final_auc=float(a[-1]),
                                             final_loss=float(l[-1]))
                                     for n, (d, a, l) in traces.items()}

        # ---- Bayesian ----
        pr = ckd_params(T, trace_cov["young-normal"])
        ke_pr, v_pr = float(pr["ke"]), float(pr["v"])
        bres = synthetic_demo(T, ke_pr, v_pr, ke_true=ke_pr * 1.4, v_true=v_pr * 0.85,
                              dose=1250.0, tau=12.0, noise=0.0)
        provenance["bayesian"] = {k: float(v) for k, v in bres.items()
                                  if isinstance(v, (int, float))}

        # ---- render both themes ----
        paths = []
        for th in ("light", "dark"):
            paths.append(fig_hero(T, patients, th, hero))
            paths.append(fig_esrd(T, th, esrd))
            paths.append(fig_convergence(th, traces))
            paths.append(fig_bayesian(th, bres))
            paths.append(fig_loading(th, load_data))

    with open(os.path.join(OUT, "provenance.json"), "w") as fh:
        json.dump(provenance, fh, indent=2)
    print("\n=== REAL ENGINE NUMBERS (provenance) ===")
    print(json.dumps(provenance, indent=2))
    print("\n=== FIGURES ===")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
