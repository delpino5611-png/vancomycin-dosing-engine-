# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Composition + gradient optimizer for the vancomycin dosing engine.

Composes the three served Tesseracts with tesseract-jax so jax.grad flows
end-to-end across HTTP boundaries:

    covariates --[T1 ckd_physiology]--> ke,v --[T2 vanco_pk]--> auc24,peak,trough
              --[T3 exposure_loss]--> loss

- verify_gradients(): end-to-end grad of loss w.r.t. a covariate (SCr) through all
  three Tesseracts, checked against central finite differences; plus grad w.r.t. dose.
- optimize_regimen(): for each CrCl-feasible interval, bounded gradient descent on
  dose (sigmoid reparam keeps dose in [250,2000] so it cannot explode -> no NaN),
  then the guardrail layer picks the best safe regimen.
"""
import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

import guardrails as G

f32 = lambda x: jnp.float32(x)

LOADING_CAP = 3000.0  # mg — hard cap on the loading dose (Rybak 2020: caution > 3 g)


def connect(urls):
    T = {
        "ckd": Tesseract.from_url(urls["ckd"]),
        "pk": Tesseract.from_url(urls["pk"]),
        "loss": Tesseract.from_url(urls["loss"]),
    }
    if "pk2c" in urls:  # optional 2-compartment stretch path
        T["pk2c"] = Tesseract.from_url(urls["pk2c"])
    return T


def ckd_params(T, cov):
    """T1: covariates -> ke, v, cl, crcl (returns full dict)."""
    return apply_tesseract(
        T["ckd"],
        {
            "age": f32(cov["age"]),
            "weight": f32(cov["weight"]),
            "height_in": f32(cov["height_in"]),
            "scr": f32(cov["scr"]),
            "sex": f32(cov["sex"]),
        },
        vmap_method="sequential",
    )


def pk_exposure(T, dose, tau, ke, v, t_inf=1.0):
    """T2: (dose,tau,ke,v) -> auc24, peak, trough, conc curve."""
    return apply_tesseract(
        T["pk"],
        {"dose": dose, "tau": tau, "ke": ke, "v": v, "t_inf": f32(t_inf)},
        vmap_method="sequential",
    )


def pk_exposure_2c(T, dose, tau, cl, v1, q, v2, t_inf=1.0):
    """T2b (stretch): 2-compartment (dose,tau,cl,v1,q,v2) -> auc24, peak, trough, curve."""
    return apply_tesseract(
        T["pk2c"],
        {"dose": dose, "tau": tau, "cl": cl, "v1": v1, "q": q, "v2": v2,
         "t_inf": f32(t_inf)},
        vmap_method="sequential",
    )


def priors_2c(p_ckd, weight):
    """Derive 2-compartment priors from the CKD module output + weight.

    - CL   : the renally-driven clearance from T1 (Ke*V) — unchanged, drug is ~90%
             renally cleared, so CrCl still dominates.
    - V1   : central volume ~7.5 L in normal renal function, scaled UP in impairment
             (Radica Z.: ~7.1-7.5 L normal -> ~22-30 L impaired), capped [7.5, 25].
    - Q    : inter-compartmental clearance ~7 L/h (population vanco 2-comp).
    - V2   : peripheral volume so that Vss = V1 + V2 ~= 0.7 L/kg (matches the 1-comp
             total volume; AUC is volume-independent so the AUC identity is preserved).
    """
    crcl = float(p_ckd["crcl"]); cl = float(p_ckd["cl"])
    v1 = min(max(7.5 * (90.0 / max(crcl, 20.0)), 7.5), 25.0)
    q = 7.0
    vss = 0.7 * float(weight)
    v2 = max(vss - v1, 15.0)
    return {"cl": cl, "v1": v1, "q": q, "v2": v2, "crcl": crcl}


def exposure_loss(T, auc24, peak, trough, target=500.0, mic=1.0):
    """T3: exposure metrics -> scalar loss. Loss targets AUC24/MIC (default MIC=1)."""
    return apply_tesseract(
        T["loss"],
        {"auc24": auc24, "peak": peak, "trough": trough,
         "target_auc": f32(target), "mic": f32(mic)},
        vmap_method="sequential",
    )


def loss_from_covariate_scr(T, cov, dose, tau, target=500.0, mic=1.0):
    """End-to-end scalar: SCr -> (T1) -> (T2) -> (T3) -> loss. Differentiable in scr."""
    def f(scr):
        p = apply_tesseract(
            T["ckd"],
            {
                "age": f32(cov["age"]),
                "weight": f32(cov["weight"]),
                "height_in": f32(cov["height_in"]),
                "scr": scr,
                "sex": f32(cov["sex"]),
            },
            vmap_method="sequential",
        )
        ex = pk_exposure(T, f32(dose), f32(tau), p["ke"], p["v"])
        out = exposure_loss(T, ex["auc24"], ex["peak"], ex["trough"], target, mic)
        return out["loss"]
    return f


def verify_gradients(T, cov, dose=1000.0, tau=12.0):
    """End-to-end gradient checks across all three composed Tesseracts."""
    results = {}

    # (1) grad of loss w.r.t. SCr through T1 -> T2 -> T3
    f = loss_from_covariate_scr(T, cov, dose, tau)
    scr0 = f32(cov["scr"])
    g = float(jax.grad(f)(scr0))
    eps = 0.01
    fd = (float(f(scr0 + eps)) - float(f(scr0 - eps))) / (2 * eps)
    results["dloss_dscr"] = g
    results["dloss_dscr_fd"] = fd
    results["dloss_dscr_relerr"] = abs(g - fd) / (abs(fd) + 1e-9)
    results["scr_nan"] = bool(jnp.isnan(jnp.float32(g)))

    # (2) grad of loss w.r.t. dose through T2 -> T3 (fix ke,v from T1)
    p = ckd_params(T, cov)
    ke, v = p["ke"], p["v"]

    def fdose(d):
        ex = pk_exposure(T, d, f32(tau), ke, v)
        return exposure_loss(T, ex["auc24"], ex["peak"], ex["trough"])["loss"]

    d0 = f32(dose)
    gd = float(jax.grad(fdose)(d0))
    fdd = (float(fdose(d0 + 1.0)) - float(fdose(d0 - 1.0))) / 2.0
    results["dloss_ddose"] = gd
    results["dloss_ddose_fd"] = fdd
    results["dloss_ddose_relerr"] = abs(gd - fdd) / (abs(fdd) + 1e-9)
    results["dose_nan"] = bool(jnp.isnan(jnp.float32(gd)))
    return results


def _optimize_dose(T, ke, v, tau, target=500.0, mic=1.0, steps=400):
    """Bounded gradient descent on dose (Adam) for one interval. Returns optimal dose (mg)."""
    lo, hi = 250.0, 2000.0

    def dose_of(theta):
        return lo + (hi - lo) * jax.nn.sigmoid(theta)

    def loss_of(theta):
        d = dose_of(theta)
        ex = pk_exposure(T, d, f32(tau), ke, v)
        return exposure_loss(T, ex["auc24"], ex["peak"], ex["trough"], target, mic)["loss"]

    grad = jax.grad(loss_of)
    theta = f32(0.0)  # start mid-range (dose ~1125 mg)
    m, vv, b1, b2, eps, lr = 0.0, 0.0, 0.9, 0.999, 1e-8, 0.15
    for t in range(1, steps + 1):
        gt = float(grad(theta))
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * gt * gt
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        theta = theta - lr * mhat / (jnp.sqrt(vhat) + eps)
    return float(dose_of(theta))


def loading_dose(v, weight, target_peak=30.0, mg_per_kg=25.0):
    """Empiric weight-based loading dose (Rybak 2020; Tesfamariam 2024).

    Two equivalent framings, reconciled and capped:
      - weight-based:  D_load = mg_per_kg * ABW  (Rybak 2020: 25-30 mg/kg actual wt)
      - PK-based:      D_load ~= Vd * target_peak  (dose to reach an initial peak)
    Returns the weight-based dose (guideline-standard) plus the PK-implied peak it
    achieves in the patient's own Vd, and the cap flag. Hard cap 3000 mg (Rybak:
    caution above 3 g, esp. in ARC). Given BEFORE the maintenance regimen so a
    narrow-therapeutic-index drug reaches target in the first 24-48 h instead of
    after 3-4 maintenance doses (the steady-state-only gap).
    """
    d_wt = mg_per_kg * float(weight)
    capped = d_wt > LOADING_CAP
    d_load = min(d_wt, LOADING_CAP)
    d_load_snap = G.snap_dose_loading(d_load)
    peak_achieved = d_load_snap / float(v)  # C_peak ~= D/Vd right after infusion
    d_pk = float(v) * float(target_peak)    # PK-based dose to hit target_peak
    return {
        "mg_per_kg": mg_per_kg, "weight": float(weight),
        "dose_weight_based": d_wt, "dose_pk_based": d_pk,
        "dose": d_load_snap, "capped": capped, "cap": LOADING_CAP,
        "peak_achieved": peak_achieved, "target_peak": float(target_peak),
        "t_inf": G.infusion_time(d_load_snap),
    }


def optimize_regimen(T, cov, target=500.0, mic=1.0):
    """Full patient optimization: T1 params -> per-interval dose search -> guardrails -> best.

    Two-phase output: an empiric LOADING dose (weight-based, given first) plus the
    gradient-optimized MAINTENANCE regimen snapped to the discrete schedule.
    """
    p = ckd_params(T, cov)
    ke = float(p["ke"]); v = float(p["v"]); cl = float(p["cl"]); crcl = float(p["crcl"])
    load = loading_dose(v, cov["weight"])
    mic_flag = G.mic_alternative_agent_flag(mic)
    candidates = []
    for tau in G.feasible_intervals(crcl):
        d_cont = _optimize_dose(T, p["ke"], p["v"], tau, target, mic)
        # AUC at the CONTINUOUS differentiable optimum (what jax.grad actually achieves)
        ex_c = pk_exposure(T, f32(d_cont), f32(tau), p["ke"], p["v"],
                           t_inf=G.infusion_time(d_cont))
        auc_cont = float(ex_c["auc24"])
        d_snap = G.snap_dose(d_cont)
        t_inf = G.infusion_time(d_snap)
        ex = pk_exposure(T, f32(d_snap), f32(tau), p["ke"], p["v"], t_inf=t_inf)
        auc24 = float(ex["auc24"]); peak = float(ex["peak"]); trough = float(ex["trough"])
        chk = G.check_regimen(d_snap, tau, peak, trough, auc24, crcl)
        candidates.append({
            "tau": tau, "dose_continuous": d_cont, "auc_continuous": auc_cont,
            "dose": d_snap, "t_inf": t_inf,
            "auc24": auc24, "peak": peak, "trough": trough,
            "auc_err": abs(auc24 - target), "guard": chk,
        })
    feasible = [c for c in candidates if c["guard"]["ok"]]
    pool = feasible if feasible else candidates
    best = min(pool, key=lambda c: c["auc_err"])
    return {
        "crcl": crcl, "ke": ke, "v": v, "cl": cl,
        "ibw": float(p["ibw"]), "adjbw": float(p["adjbw"]),
        "loading": load, "mic": float(mic), "mic_flag": mic_flag,
        "best": best, "candidates": candidates,
    }


# ===========================================================================
# 2-COMPARTMENT STRETCH PATH — separate, selectable; never replaces 1-comp core
# ===========================================================================
def verify_2c(T, cov, dose=1000.0, tau=12.0):
    """Verify the 2-comp path: end-to-end gradient (dose -> loss) vs finite diff,
    NaN check, and the linear-PK AUC identity AUC24 == daily_dose / CL."""
    p = ckd_params(T, cov)
    pr = priors_2c(p, cov["weight"])
    cl, v1, q, v2 = pr["cl"], pr["v1"], pr["q"], pr["v2"]

    def fdose(d):
        ex = pk_exposure_2c(T, d, f32(tau), f32(cl), f32(v1), f32(q), f32(v2))
        return exposure_loss(T, ex["auc24"], ex["peak"], ex["trough"])["loss"]

    d0 = f32(dose)
    gd = float(jax.grad(fdose)(d0))
    fdd = (float(fdose(d0 + 1.0)) - float(fdose(d0 - 1.0))) / 2.0
    ex = pk_exposure_2c(T, d0, f32(tau), f32(cl), f32(v1), f32(q), f32(v2))
    auc = float(ex["auc24"])
    auc_identity = (dose * (24.0 / tau)) / cl
    return {
        "cl": cl, "v1": v1, "q": q, "v2": v2,
        "dloss_ddose": gd, "dloss_ddose_fd": fdd,
        "dloss_ddose_relerr": abs(gd - fdd) / (abs(fdd) + 1e-9),
        "dose_nan": bool(jnp.isnan(jnp.float32(gd))),
        "auc24": auc, "auc_identity": auc_identity,
        "auc_relerr": abs(auc - auc_identity) / (abs(auc_identity) + 1e-9),
        "peak": float(ex["peak"]), "trough": float(ex["trough"]),
    }


def _optimize_dose_2c(T, cl, v1, q, v2, tau, target=500.0, mic=1.0, steps=400):
    """Bounded gradient descent on dose (Adam) through the 2-comp path."""
    lo, hi = 250.0, 2000.0

    def dose_of(theta):
        return lo + (hi - lo) * jax.nn.sigmoid(theta)

    def loss_of(theta):
        d = dose_of(theta)
        ex = pk_exposure_2c(T, d, f32(tau), f32(cl), f32(v1), f32(q), f32(v2))
        return exposure_loss(T, ex["auc24"], ex["peak"], ex["trough"], target, mic)["loss"]

    grad = jax.grad(loss_of)
    theta = f32(0.0)
    m, vv, b1, b2, eps, lr = 0.0, 0.0, 0.9, 0.999, 1e-8, 0.15
    for t in range(1, steps + 1):
        gt = float(grad(theta))
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * gt * gt
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        theta = theta - lr * mhat / (jnp.sqrt(vhat) + eps)
    return float(dose_of(theta))


def optimize_regimen_2c(T, cov, target=500.0, mic=1.0):
    """2-comp regimen optimization (stretch). Same structure as optimize_regimen."""
    p = ckd_params(T, cov)
    pr = priors_2c(p, cov["weight"])
    cl, v1, q, v2, crcl = pr["cl"], pr["v1"], pr["q"], pr["v2"], pr["crcl"]
    candidates = []
    for tau in G.feasible_intervals(crcl):
        d_cont = _optimize_dose_2c(T, cl, v1, q, v2, tau, target, mic)
        ex_c = pk_exposure_2c(T, f32(d_cont), f32(tau), f32(cl), f32(v1), f32(q), f32(v2),
                              t_inf=G.infusion_time(d_cont))
        auc_cont = float(ex_c["auc24"])
        d_snap = G.snap_dose(d_cont)
        t_inf = G.infusion_time(d_snap)
        ex = pk_exposure_2c(T, f32(d_snap), f32(tau), f32(cl), f32(v1), f32(q), f32(v2),
                            t_inf=t_inf)
        auc24 = float(ex["auc24"]); peak = float(ex["peak"]); trough = float(ex["trough"])
        chk = G.check_regimen(d_snap, tau, peak, trough, auc24, crcl)
        candidates.append({
            "tau": tau, "dose_continuous": d_cont, "auc_continuous": auc_cont,
            "dose": d_snap, "t_inf": t_inf, "auc24": auc24, "peak": peak,
            "trough": trough, "auc_err": abs(auc24 - target), "guard": chk,
        })
    feasible = [c for c in candidates if c["guard"]["ok"]]
    pool = feasible if feasible else candidates
    best = min(pool, key=lambda c: c["auc_err"])
    return {"crcl": crcl, "cl": cl, "v1": v1, "q": q, "v2": v2,
            "best": best, "candidates": candidates}
