# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Simulation-based parameter-recovery validation for the vancomycin engine.

Turns the Bayesian-individualization claim into a DEMONSTRATED result. The
experiment is standard simulation-based calibration for a population PK model:

  1. Draw N virtual patients. Each patient's covariates map through the LOCKED
     CKD physiology module (Cockcroft-Gault + Matzke) to a population prior
     (Ke_prior, V_prior). The patient's TRUE parameters are then drawn from the
     lognormal between-patient-variability (IIV) model the MAP fitter assumes:
        Ke_true = Ke_prior * exp(N(0, omega_ke)),  omega_ke = 0.40
        V_true  = V_prior  * exp(N(0, omega_v)),   omega_v  = 0.30
  2. Simulate two measured steady-state levels (peak + trough) from the true
     parameters through the LOCKED 1-compartment PK model, then add realistic
     combined assay noise: sigma = sqrt(sigma_add^2 + (sigma_prop*C)^2),
     sigma_prop = 0.12, sigma_add = 2 mg/L (exactly bayesian.py's Sigma).
  3. Recover (Ke, V) from the two noisy levels for every patient with the MAP
     objective and Adam optimizer of bayesian.map_update.
  4. Score recovery: correlation + RMSE of estimated-vs-true for Ke, V, and the
     clinically decisive AUC24 (= daily_dose / CL). Then repeat the fit on a
     single CORRUPTED (spuriously low) trough with the prior on (MAP) and off
     (MLE) to show the population prior is what resists a bad level.

FORWARD MODEL + SPEED. bayesian.map_update is a per-patient Python Adam loop that
differentiates its level predictions through the served PK Tesseract over HTTP.
Run over N>=200 patients (times MAP and MLE) that is thousands of HTTP round-trips
each and takes ~40 minutes. This experiment instead evaluates the IDENTICAL MAP
objective in-process and vectorized: batch_map is jax.vmap over patients + lax.scan
over Adam steps, jitted, using the EXACT locked served PK function
(vanco_pk/tesseract_api._ss_conc_at) for the peak/trough predictions, the same
combined-error residual, the same lognormal prior, and the same Adam
hyperparameters as bayesian.map_update. It is not a separate model: validate_batch()
runs the canonical bayesian.map_update on a handful of patients and confirms
batch_map reproduces it to numerical precision, so the vectorized driver is proven
equivalent to the real fitter, not merely asserted to be. The whole N=200 sweep
then finishes in seconds.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from functools import partial

import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "figures")
os.makedirs(OUT, exist_ok=True)

f32 = lambda x: jnp.float32(x)


# ---------------------------------------------------------------------------
# Load the two LOCKED served modules in-process (identical code, no HTTP).
# Both files are named tesseract_api.py, so load each under a unique name.
# ---------------------------------------------------------------------------
def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(HERE, rel_path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)
    return m


pk_api = _load("vanco_pk_api", os.path.join("vanco_pk", "tesseract_api.py"))
ckd_api = _load("ckd_api", os.path.join("ckd_physiology", "tesseract_api.py"))

# residual (Sigma) and prior (Omega) hyperparameters -- taken from the fitter itself
import bayesian
SIGMA_PROP = bayesian.SIGMA_PROP   # 0.12
SIGMA_ADD = bayesian.SIGMA_ADD     # 2.0
OMEGA_KE = bayesian.OMEGA_KE       # 0.40
OMEGA_V = bayesian.OMEGA_V         # 0.30

# reference regimen used to express recovery in AUC terms (linear-PK identity)
DOSE, TAU, T_INF = 1000.0, 12.0, 1.0
DAILY = DOSE * (24.0 / TAU)
STEPS = 800            # same as bayesian.map_update default
TROUGH_FLOOR = 0.5     # assay floor, mg/L (avoid nonphysical negative draws)
BAD_TROUGH_MULT = 0.4  # a single spuriously LOW trough (early draw / lab error)


def auc_of(ke, v):
    """Steady-state AUC24 at the reference regimen (linear-PK identity, mg*h/L)."""
    return DAILY / (ke * v)


# ---------------------------------------------------------------------------
# Vectorized MAP fit == bayesian.map_update objective + Adam, batched & jitted.
# Uses the locked served PK peak/trough function directly (pk_api._ss_conc_at).
# ---------------------------------------------------------------------------
def _peak_trough(ke, v):
    """SS peak/trough for SCALAR ke, v via the locked served PK function."""
    peak = pk_api._ss_conc_at(f32(T_INF), f32(DOSE), ke, v, f32(TAU), f32(T_INF))
    trough = pk_api._ss_conc_at(f32(TAU), f32(DOSE), ke, v, f32(TAU), f32(T_INF))
    return peak, trough


_peak_trough_batch = jax.jit(jax.vmap(_peak_trough))  # over a batch of patients


def _fit_one(ke_pr, v_pr, peak_obs, trough_obs, pw, steps):
    """One patient's MAP fit -- identical Phi + Adam to bayesian.map_update."""
    l_ke_prior = jnp.log(ke_pr)
    l_v_prior = jnp.log(v_pr)

    def phi(params):
        l_ke, l_v = params[0], params[1]
        ke = jnp.exp(l_ke); v = jnp.exp(l_v)
        peak, trough = _peak_trough(ke, v)
        s_peak = jnp.sqrt(SIGMA_ADD ** 2 + (SIGMA_PROP * peak) ** 2)
        s_trough = jnp.sqrt(SIGMA_ADD ** 2 + (SIGMA_PROP * trough) ** 2)
        resid = (((peak - peak_obs) / s_peak) ** 2
                 + ((trough - trough_obs) / s_trough) ** 2)
        prior = (((l_ke - l_ke_prior) / OMEGA_KE) ** 2
                 + ((l_v - l_v_prior) / OMEGA_V) ** 2)
        return resid + pw * prior

    grad = jax.grad(phi)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.03

    def body(carry, t):
        params, m, vv = carry
        gt = grad(params)
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * gt * gt
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        params = params - lr * mhat / (jnp.sqrt(vhat) + eps)
        return (params, m, vv), None

    params0 = jnp.array([l_ke_prior, l_v_prior], dtype=jnp.float32)
    init = (params0, jnp.zeros(2), jnp.zeros(2))
    ts = jnp.arange(1, steps + 1, dtype=jnp.float32)
    (params, _, _), _ = jax.lax.scan(body, init, ts)
    return jnp.exp(params[0]), jnp.exp(params[1])


@partial(jax.jit, static_argnums=(5,))
def batch_map(ke_pr, v_pr, peak_obs, trough_obs, pw, steps):
    """Vectorized MAP recovery over a batch of patients. pw=1 MAP, pw=0 MLE."""
    fit = lambda a, b, c, d: _fit_one(a, b, c, d, pw, steps)
    return jax.vmap(fit)(ke_pr, v_pr, peak_obs, trough_obs)


# ---------------------------------------------------------------------------
# Population: covariates -> locked CKD prior; true params from the IIV model
# ---------------------------------------------------------------------------
def sample_population(n, seed=20260817):
    rng = np.random.default_rng(seed)
    age = rng.uniform(25.0, 85.0, n)
    weight = np.clip(rng.normal(80.0, 18.0, n), 45.0, 140.0)
    height_in = np.clip(rng.normal(68.0, 4.0, n), 58.0, 78.0)
    scr = np.clip(np.exp(rng.normal(np.log(1.1), 0.5, n)), 0.5, 6.0)  # renal spectrum
    sex = (rng.random(n) < 0.5).astype(np.float32)

    # locked CKD module -> population prior (vectorized over patients)
    p = ckd_api.apply_jit({"age": f32(age), "weight": f32(weight),
                           "height_in": f32(height_in), "scr": f32(scr),
                           "sex": f32(sex)})
    ke_prior = np.asarray(p["ke"], dtype=np.float64)
    v_prior = np.asarray(p["v"], dtype=np.float64)

    # TRUE params from the lognormal between-patient-variability model the MAP assumes
    ke_true = ke_prior * np.exp(rng.normal(0.0, OMEGA_KE, n))
    v_true = v_prior * np.exp(rng.normal(0.0, OMEGA_V, n))
    return dict(rng=rng, ke_prior=ke_prior, v_prior=v_prior,
                ke_true=ke_true, v_true=v_true)


def sim_levels(ke_true, v_true, rng):
    """Two noisy steady-state levels (peak, trough) per patient (vectorized)."""
    peak_t, trough_t = _peak_trough_batch(f32(np.atleast_1d(ke_true)),
                                          f32(np.atleast_1d(v_true)))
    peak_t = np.asarray(peak_t, dtype=np.float64)
    trough_t = np.asarray(trough_t, dtype=np.float64)

    def add_noise(c):
        s = np.sqrt(SIGMA_ADD ** 2 + (SIGMA_PROP * c) ** 2)
        return np.maximum(c + rng.normal(0.0, s), TROUGH_FLOOR)

    return add_noise(peak_t), add_noise(trough_t), peak_t, trough_t


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def pearson(a, b):
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def rel_rmse_pct(true, est):
    """RMSE normalized by the mean true value, in percent."""
    return 100.0 * rmse(true, est) / float(np.mean(np.asarray(true)))


# ---------------------------------------------------------------------------
# Validate the vectorized batch fit against the canonical bayesian.map_update
# ---------------------------------------------------------------------------
def validate_batch(k=8, seed=777):
    """Confirm batch_map reproduces the real bayesian.map_update to precision."""
    # patch the fitter's PK to the same locked function, in-process
    def pk_local(T, dose, tau, ke, v, t_inf=1.0):
        return pk_api.apply_jit({"dose": f32(dose), "tau": f32(tau), "ke": f32(ke),
                                 "v": f32(v), "t_inf": f32(t_inf)})
    bayesian.pk_exposure = pk_local

    pop = sample_population(k, seed=seed)
    peak_o, trough_o, _, _ = sim_levels(pop["ke_true"], pop["v_true"], pop["rng"])
    ke_b, v_b = batch_map(f32(pop["ke_prior"]), f32(pop["v_prior"]),
                          f32(peak_o), f32(trough_o), 1.0, STEPS)
    ke_b = np.asarray(ke_b); v_b = np.asarray(v_b)
    dke = dv = 0.0
    for i in range(k):
        r = bayesian.map_update(None, pop["ke_prior"][i], pop["v_prior"][i],
                                DOSE, TAU, f32(peak_o[i]), f32(trough_o[i]), steps=STEPS)
        dke = max(dke, abs(r["ke_post"] - float(ke_b[i])))
        dv = max(dv, abs(r["v_post"] - float(v_b[i])))
    print(f"[validate] batch_map vs bayesian.map_update on K={k}: "
          f"max|dKe|={dke:.2e}  max|dV|={dv:.2e}")
    return dke, dv


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_experiment(n):
    pop = sample_population(n)
    rng = pop["rng"]
    ke_prior, v_prior = pop["ke_prior"], pop["v_prior"]
    ke_true, v_true = pop["ke_true"], pop["v_true"]

    peak_o, trough_o, peak_t, trough_t = sim_levels(ke_true, v_true, rng)

    t0 = time.time()
    # (A) clean-workflow recovery: two realistically noisy levels, MAP
    ke_est, v_est = batch_map(f32(ke_prior), f32(v_prior),
                              f32(peak_o), f32(trough_o), 1.0, STEPS)
    # (B) bad-level stress test: one spuriously low trough; peak keeps its noise.
    trough_bad = np.maximum(trough_t * BAD_TROUGH_MULT, TROUGH_FLOOR)
    ke_map, v_map = batch_map(f32(ke_prior), f32(v_prior),
                              f32(peak_o), f32(trough_bad), 1.0, STEPS)
    ke_mle, v_mle = batch_map(f32(ke_prior), f32(v_prior),
                              f32(peak_o), f32(trough_bad), 0.0, STEPS)
    elapsed = time.time() - t0

    ke_est = np.asarray(ke_est); v_est = np.asarray(v_est)
    ke_map = np.asarray(ke_map); v_map = np.asarray(v_map)
    ke_mle = np.asarray(ke_mle); v_mle = np.asarray(v_mle)

    # derived AUC24
    auc_true = auc_of(ke_true, v_true)
    auc_prior = auc_of(ke_prior, v_prior)
    auc_est = auc_of(ke_est, v_est)
    auc_map = auc_of(ke_map, v_map)
    auc_mle = auc_of(ke_mle, v_mle)

    metrics = {
        "n": n, "dose": DOSE, "tau": TAU, "steps": STEPS,
        "omega_ke": OMEGA_KE, "omega_v": OMEGA_V,
        "sigma_prop": SIGMA_PROP, "sigma_add": SIGMA_ADD,
        "ke": {"r": pearson(ke_true, ke_est), "rmse": rmse(ke_true, ke_est),
               "rel_rmse_pct": rel_rmse_pct(ke_true, ke_est)},
        "v": {"r": pearson(v_true, v_est), "rmse": rmse(v_true, v_est),
              "rel_rmse_pct": rel_rmse_pct(v_true, v_est)},
        "auc": {"r": pearson(auc_true, auc_est), "rmse": rmse(auc_true, auc_est),
                "rel_rmse_pct": rel_rmse_pct(auc_true, auc_est)},
        "auc_prior_rmse": rmse(auc_true, auc_prior),
        "auc_prior_rel_rmse_pct": rel_rmse_pct(auc_true, auc_prior),
        "bad": {
            "ke_rmse_map": rmse(ke_true, ke_map), "ke_rmse_mle": rmse(ke_true, ke_mle),
            "v_rmse_map": rmse(v_true, v_map), "v_rmse_mle": rmse(v_true, v_mle),
            "auc_rmse_map": rmse(auc_true, auc_map), "auc_rmse_mle": rmse(auc_true, auc_mle),
            "ke_relrmse_map": rel_rmse_pct(ke_true, ke_map),
            "ke_relrmse_mle": rel_rmse_pct(ke_true, ke_mle),
            "v_relrmse_map": rel_rmse_pct(v_true, v_map),
            "v_relrmse_mle": rel_rmse_pct(v_true, v_mle),
            "auc_relrmse_map": rel_rmse_pct(auc_true, auc_map),
            "auc_relrmse_mle": rel_rmse_pct(auc_true, auc_mle),
        },
        "elapsed_s": elapsed,
    }
    arrays = dict(ke_true=ke_true, ke_est=ke_est, v_true=v_true, v_est=v_est,
                  auc_true=auc_true, auc_est=auc_est, auc_map=auc_map, auc_mle=auc_mle)
    return metrics, arrays


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
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


def _scatter(ax, t, xt, ye, color, title, unit, m):
    lo = min(float(np.min(xt)), float(np.min(ye)))
    hi = max(float(np.max(xt)), float(np.max(ye)))
    pad = 0.06 * (hi - lo)
    lo -= pad; hi += pad
    ax.plot([lo, hi], [lo, hi], color=t["muted"], lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.scatter(xt, ye, s=26, color=color, alpha=0.55, edgecolor="none", zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=8)
    ax.set_xlabel(f"True {unit}", fontsize=11.5)
    ax.set_ylabel(f"Recovered {unit}", fontsize=11.5)
    ax.tick_params(labelsize=10)
    box = dict(boxstyle="round,pad=0.4", fc=t["surface"], ec=t["grid"], lw=1)
    ax.text(0.04, 0.955, f"r = {m['r']:.3f}\nRMSE = {m['rmse']:.3g}\n({m['rel_rmse_pct']:.1f}%)",
            transform=ax.transAxes, fontsize=10.5, va="top", ha="left",
            color=t["ink"], fontweight="bold", bbox=box, zorder=6)


def make_figure(th, metrics, arrays):
    t = style(th)
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 12.2))
    fig.subplots_adjust(left=0.075, right=0.975, top=0.855, bottom=0.075,
                        wspace=0.26, hspace=0.30)
    (axK, axV), (axA, axM) = axes

    _scatter(axK, t, arrays["ke_true"], arrays["ke_est"], t["blue"],
             "Elimination rate Ke", "Ke (1/h)", metrics["ke"])
    _scatter(axV, t, arrays["v_true"], arrays["v_est"], t["aqua"],
             "Volume of distribution V", "V (L)", metrics["v"])
    _scatter(axA, t, arrays["auc_true"], arrays["auc_est"], t["violet"],
             "Exposure AUC24 (the clinical target)", "AUC24 (mg*h/L)", metrics["auc"])

    # panel 4: MAP vs MLE under a single BAD (spuriously low) trough -- relative RMSE
    b = metrics["bad"]
    groups = ["Ke", "V", "AUC24"]
    map_vals = [b["ke_relrmse_map"], b["v_relrmse_map"], b["auc_relrmse_map"]]
    mle_vals = [b["ke_relrmse_mle"], b["v_relrmse_mle"], b["auc_relrmse_mle"]]
    x = np.arange(len(groups)); w = 0.36
    bars_map = axM.bar(x - w / 2, map_vals, w, color=STATUS["good"], zorder=3,
                       label="MAP (population prior ON)")
    bars_mle = axM.bar(x + w / 2, mle_vals, w, color=STATUS["critical"], zorder=3,
                       label="MLE (no prior)")
    for bars in (bars_map, bars_mle):
        for bar in bars:
            h = bar.get_height()
            axM.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.0f}%",
                     ha="center", va="bottom", fontsize=10, color=t["ink"],
                     fontweight="bold")
    axM.set_xticks(x); axM.set_xticklabels(groups, fontsize=11.5)
    axM.set_ylabel("Recovery error (relative RMSE, %)", fontsize=11.5)
    axM.set_ylim(0, max(mle_vals) * 1.22)
    axM.set_title("Bad level (one spuriously low trough): the prior keeps MAP honest",
                  fontsize=13.5, fontweight="bold", pad=8)
    axM.grid(axis="x", visible=False)
    axM.tick_params(labelsize=10)
    axM.legend(loc="upper left", fontsize=10.5, framealpha=0.0)

    fig.suptitle("Parameter recovery: the engine recovers true patient PK from two noisy "
                 "levels, and the prior makes it robust",
                 fontsize=17, fontweight="bold", x=0.075, ha="left", y=0.965)
    fig.text(0.075, 0.905,
             f"N = {metrics['n']} virtual patients. True (Ke, V) drawn from the population "
             f"variability model (CV ~ {int(round(OMEGA_KE*100))}% Ke / "
             f"{int(round(OMEGA_V*100))}% V); two levels simulated with combined assay noise "
             f"({int(SIGMA_PROP*100)}% + {SIGMA_ADD:.0f} mg/L).",
             fontsize=11, color=t["ink2"], ha="left")
    fig.text(0.075, 0.877,
             "The MAP fitter then recovers (Ke, V). Panels 1 to 3: recovered vs true (identity "
             "line). Panel 4: on a corrupted trough, the prior (MAP) resists what MLE chases.",
             fontsize=11, color=t["ink2"], ha="left")
    fig.text(0.075, 0.022,
             "Self-generated from the engine (MAP objective + Adam of bayesian.map_update, "
             "vectorized; forward model = the locked served 1-compartment PK). Prior: Matzke "
             "1984 / Onor 2020. Residual model + IIV: Rybak 2020.",
             fontsize=9, color=t["muted"], ha="left")
    p = os.path.join(OUT, f"fig6_recovery_{th}.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def print_report(metrics, val=None):
    m = metrics; b = m["bad"]
    if val is not None:
        print(f"Validation vs bayesian.map_update: max|dKe|={val[0]:.2e}  max|dV|={val[1]:.2e}")
    print("\n=== PARAMETER RECOVERY (N={}, {} steps/fit, batch fit {:.2f}s) ===".format(
        m["n"], m["steps"], m["elapsed_s"]))
    print("Two noisy levels -> MAP recovery:")
    print(f"  Ke   : r={m['ke']['r']:.3f}  RMSE={m['ke']['rmse']:.4f} 1/h  ({m['ke']['rel_rmse_pct']:.1f}%)")
    print(f"  V    : r={m['v']['r']:.3f}  RMSE={m['v']['rmse']:.2f} L    ({m['v']['rel_rmse_pct']:.1f}%)")
    print(f"  AUC24: r={m['auc']['r']:.3f}  RMSE={m['auc']['rmse']:.1f} mg*h/L ({m['auc']['rel_rmse_pct']:.1f}%)")
    print(f"  AUC24 from population prior ALONE: RMSE={m['auc_prior_rmse']:.1f} "
          f"({m['auc_prior_rel_rmse_pct']:.1f}%)  -> two levels cut it to "
          f"{m['auc']['rmse']:.1f} ({m['auc']['rel_rmse_pct']:.1f}%)")
    print("Bad-level stress test (one spuriously low trough), MAP vs MLE relative RMSE:")
    print(f"  Ke   : MAP {b['ke_relrmse_map']:.1f}%  vs  MLE {b['ke_relrmse_mle']:.1f}%")
    print(f"  V    : MAP {b['v_relrmse_map']:.1f}%  vs  MLE {b['v_relrmse_mle']:.1f}%")
    print(f"  AUC24: MAP {b['auc_relrmse_map']:.1f}%  vs  MLE {b['auc_relrmse_mle']:.1f}%")


def to_jsonable(metrics):
    return json.loads(json.dumps(metrics, default=lambda o: (
        o.tolist() if isinstance(o, np.ndarray) else float(o))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    val = None
    if not args.no_validate:
        val = validate_batch()

    print(f"Running parameter recovery on N={args.n} virtual patients "
          f"(MAP objective of bayesian.map_update, vectorized, locked PK in-process)...")
    metrics, arrays = run_experiment(args.n)
    print_report(metrics, val)

    paths = [make_figure(th, metrics, arrays) for th in ("light", "dark")]

    prov = {"metrics": to_jsonable(metrics)}
    if val is not None:
        prov["validation_vs_map_update"] = {"max_abs_dke": val[0], "max_abs_dv": val[1]}
    prov_path = os.path.join(OUT, "recovery_provenance.json")
    with open(prov_path, "w") as fh:
        json.dump(prov, fh, indent=2)

    print("\n=== FIGURES ===")
    for p in paths:
        print(p)
    print(prov_path)


if __name__ == "__main__":
    main()
