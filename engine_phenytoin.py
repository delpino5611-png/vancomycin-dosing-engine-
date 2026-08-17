# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Phenytoin generality demo: the SAME machinery, a DIFFERENT drug structure.

This composes the SAME CKD physiology Tesseract (T1, reused unchanged) with the
NEW saturable phenytoin PK Tesseract (T2, Michaelis-Menten) and the NEW
concentration-target loss Tesseract (T3), and drives them with the SAME
gradient-descent optimizer and the SAME MAP-Bayesian fitter algorithm used for
vancomycin. The reusable part is the MACHINERY; only the two drug-specific
Tesseracts (model structure + clinical target) change.

    covariates --[T1 ckd_physiology, SAME]--> body size --> vmax,v
              --[T2 phenytoin_pk, saturable MM]--> trough,peak,cavg
              --[T3 phenytoin_loss, level target]--> loss

Published params (Spruill 2001 PMID 11714214; Winter, standard TDM; Kane 2016):
  Vmax ~7 mg/kg/day (population 5 to 9), Km ~4 to 6 mg/L, V ~0.65 L/kg,
  therapeutic total level 10 to 20 mg/L (free 1 to 2, fu ~0.1).
"""
import jax
import jax.numpy as jnp
from tesseract_jax import apply_tesseract

import guardrails_phenytoin as G
from engine import connect, ckd_params, f32  # connect + the SAME CKD module call

VMAX_MGKGDAY = 7.0    # population Vmax (mg/kg/day) [Spruill 2001]
KM_DEFAULT = 5.0      # population Km (mg/L) [Spruill 2001]
VD_LKG = 0.65         # phenytoin volume of distribution (L/kg) [Winter]
VANCO_VD_LKG = 0.7    # the coefficient inside ckd_physiology's differentiable v output


# ---------------------------------------------------------------------------
# Covariate -> phenytoin params, THROUGH the reused CKD Tesseract (T1)
# ---------------------------------------------------------------------------
def pheny_params(T, cov):
    """T1 (SAME module) -> body size -> phenytoin Vmax, Km, V.

    The CKD Tesseract's differentiable output v = 0.7*weight is used as the body
    size carrier, so weight_eff = v / 0.7. Phenytoin clearance is hepatic, so CrCl
    is informational here (returned but not a driver); the 2025 CKD literature
    effect on phenytoin enters through protein binding (documented extension).
    vmax_scale models metabolizer status (age / CYP2C9 genotype)."""
    p = ckd_params(T, cov)
    weight_eff = float(p["v"]) / VANCO_VD_LKG
    scale = float(cov.get("vmax_scale", 1.0))
    vmax = (VMAX_MGKGDAY / 24.0) * weight_eff * scale   # mg/h
    v = VD_LKG * weight_eff                             # L
    return {"crcl": float(p["crcl"]), "vmax": vmax, "km": KM_DEFAULT, "v": v,
            "weight": weight_eff, "vmax_scale": scale,
            "vmax_mgday": vmax * 24.0}


def pheny_exposure(T, dose, tau, vmax, km, v, t_inf=1.0):
    """T2: (dose,tau,vmax,km,v) -> saturable SS trough,peak,cavg,auc24,conc curve."""
    return apply_tesseract(
        T["pk"],
        {"dose": dose, "tau": tau, "vmax": vmax, "km": km, "v": v, "t_inf": f32(t_inf)},
        vmap_method="sequential",
    )


def pheny_loss(T, trough, peak, target=15.0):
    """T3: level metrics -> scalar loss (targets total trough level)."""
    return apply_tesseract(
        T["loss"],
        {"trough": trough, "peak": peak, "target_level": f32(target)},
        vmap_method="sequential",
    )


# ---------------------------------------------------------------------------
# End-to-end gradient verification (SAME rigor as the vanco path)
# ---------------------------------------------------------------------------
def loss_from_covariate_weight(T, cov, dose, tau, target=15.0):
    """End-to-end scalar: weight -> (T1) -> vmax,v -> (T2) -> (T3) -> loss.
    Differentiable in weight THROUGH the reused CKD Tesseract."""
    scale = float(cov.get("vmax_scale", 1.0))

    def f(weight):
        p = apply_tesseract(
            T["ckd"],
            {"age": f32(cov["age"]), "weight": weight,
             "height_in": f32(cov["height_in"]), "scr": f32(cov["scr"]),
             "sex": f32(cov["sex"])},
            vmap_method="sequential",
        )
        weight_eff = p["v"] / VANCO_VD_LKG            # differentiable T1 output
        vmax = (VMAX_MGKGDAY / 24.0) * weight_eff * scale
        v = VD_LKG * weight_eff
        ex = pheny_exposure(T, f32(dose), f32(tau), vmax, f32(KM_DEFAULT), v)
        return pheny_loss(T, ex["trough"], ex["peak"], target)["loss"]

    return f


def verify_gradients(T, cov, dose=150.0, tau=12.0, target=15.0):
    """Gradient checks: autodiff vs central finite differences, plus NaN checks.
    (1) d(loss)/d(dose)   through T2 -> T3
    (2) d(loss)/d(Vmax)   through T2 -> T3   (the saturable parameter)
    (3) d(loss)/d(weight) through T1 -> T2 -> T3 (the reused CKD module composes)
    Also the steady-state mass-balance identity (nonlinear analog of AUC=dose/CL)."""
    results = {}
    p = pheny_params(T, cov)
    vmax, km, v = p["vmax"], p["km"], p["v"]

    # (1) grad w.r.t dose
    def fdose(d):
        ex = pheny_exposure(T, d, f32(tau), f32(vmax), f32(km), f32(v))
        return pheny_loss(T, ex["trough"], ex["peak"], target)["loss"]

    d0 = f32(dose)
    gd = float(jax.grad(fdose)(d0))
    fdd = (float(fdose(d0 + 1.0)) - float(fdose(d0 - 1.0))) / 2.0
    results["dloss_ddose"] = gd
    results["dloss_ddose_fd"] = fdd
    results["dloss_ddose_relerr"] = abs(gd - fdd) / (abs(fdd) + 1e-9)
    results["dose_nan"] = bool(jnp.isnan(jnp.float32(gd)))

    # (2) grad w.r.t Vmax (the nonlinear elimination parameter)
    def fvmax(vm):
        ex = pheny_exposure(T, f32(dose), f32(tau), vm, f32(km), f32(v))
        return pheny_loss(T, ex["trough"], ex["peak"], target)["loss"]

    vm0 = f32(vmax)
    gv = float(jax.grad(fvmax)(vm0))
    eps_v = 0.05
    fdv = (float(fvmax(vm0 + eps_v)) - float(fvmax(vm0 - eps_v))) / (2 * eps_v)
    results["dloss_dvmax"] = gv
    results["dloss_dvmax_fd"] = fdv
    results["dloss_dvmax_relerr"] = abs(gv - fdv) / (abs(fdv) + 1e-9)
    results["vmax_nan"] = bool(jnp.isnan(jnp.float32(gv)))

    # (3) grad w.r.t weight, THROUGH the reused CKD Tesseract T1
    fw = loss_from_covariate_weight(T, cov, dose, tau, target)
    w0 = f32(cov["weight"])
    gw = float(jax.grad(fw)(w0))
    fdw = (float(fw(w0 + 0.5)) - float(fw(w0 - 0.5))) / 1.0
    results["dloss_dweight"] = gw
    results["dloss_dweight_fd"] = fdw
    results["dloss_dweight_relerr"] = abs(gw - fdw) / (abs(fdw) + 1e-9)
    results["weight_nan"] = bool(jnp.isnan(jnp.float32(gw)))

    # (4) steady-state mass-balance identity: eliminated per interval == dose
    ex = pheny_exposure(T, f32(dose), f32(tau), f32(vmax), f32(km), f32(v))
    conc = jnp.asarray(ex["conc"])
    times = jnp.asarray(ex["times"])
    elim = vmax * jnp.maximum(conc, 0.0) / (km + jnp.maximum(conc, 0.0))
    dt = float(times[1] - times[0])
    eliminated = float(dt * (jnp.sum(elim) - 0.5 * (elim[0] + elim[-1])))
    results["ss_eliminated"] = eliminated
    results["ss_dose"] = float(dose)
    results["ss_massbal_relerr"] = abs(eliminated - dose) / (abs(dose) + 1e-9)
    # textbook average-concentration check Css ~= Km*R/(Vmax-R)
    r_rate = dose / tau
    css_analytic = km * r_rate / max(vmax - r_rate, 1e-6)
    results["css_model"] = float(ex["cavg"])
    results["css_analytic"] = float(css_analytic)
    results["trough"] = float(ex["trough"])
    results["peak"] = float(ex["peak"])
    return results


# ---------------------------------------------------------------------------
# The SAME optimizer: bounded sigmoid reparam + Adam (identical to engine._optimize_dose)
# ---------------------------------------------------------------------------
def _optimize_dose(T, vmax, km, v, tau, target=15.0, steps=800):
    """Bounded gradient descent on the per-administration dose. Identical algorithm
    and hyperparameters to the vancomycin optimizer; only the composed PK+loss and
    the [lo,hi] dose window differ. The sigmoid reparam keeps dose in [25,400] so it
    can never cross into the saturation asymptote and produce a NaN."""
    lo, hi = 25.0, 400.0

    def dose_of(theta):
        return lo + (hi - lo) * jax.nn.sigmoid(theta)

    def loss_of(theta):
        d = dose_of(theta)
        ex = pheny_exposure(T, d, f32(tau), f32(vmax), f32(km), f32(v))
        return pheny_loss(T, ex["trough"], ex["peak"], target)["loss"]

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


def optimize_regimen(T, cov, target=15.0):
    """Full patient optimization: T1 params -> per-interval dose search (SAME optimizer)
    -> phenytoin guardrails -> best regimen (closest trough to target)."""
    p = pheny_params(T, cov)
    vmax, km, v = p["vmax"], p["km"], p["v"]
    candidates = []
    for tau in G.feasible_intervals():
        d_cont = _optimize_dose(T, vmax, km, v, tau, target)
        ex_c = pheny_exposure(T, f32(d_cont), f32(tau), f32(vmax), f32(km), f32(v))
        trough_cont = float(ex_c["trough"])
        d_snap = G.snap_dose(d_cont)
        ex = pheny_exposure(T, f32(d_snap), f32(tau), f32(vmax), f32(km), f32(v))
        trough = float(ex["trough"]); peak = float(ex["peak"]); cavg = float(ex["cavg"])
        chk = G.check_regimen(d_snap, tau, peak, trough, cavg, vmax)
        candidates.append({
            "tau": tau, "dose_continuous": d_cont, "trough_continuous": trough_cont,
            "dose": d_snap, "daily": d_snap * 24.0 / tau,
            "trough": trough, "peak": peak, "cavg": cavg,
            "trough_err": abs(trough - target), "guard": chk,
        })
    feasible = [c for c in candidates if c["guard"]["ok"]]
    pool = feasible if feasible else candidates
    best = min(pool, key=lambda c: c["trough_err"])
    return {"crcl": p["crcl"], "vmax": vmax, "km": km, "v": v,
            "vmax_mgday": p["vmax_mgday"], "vmax_scale": p["vmax_scale"],
            "best": best, "candidates": candidates}


# ---------------------------------------------------------------------------
# The SAME MAP-Bayesian fitter (penalized weighted least squares, Adam),
# fitting the saturable parameters from a sparse level.
# ---------------------------------------------------------------------------
SIGMA_PROP = 0.12
SIGMA_ADD = 1.0
OMEGA_VMAX = 0.40   # loose: Vmax is the clinically variable, identifiable parameter
OMEGA_KM = 0.12     # TIGHT: Km is weakly identifiable from sparse levels (Vmax/Km
                    # collinear), so the population prior carries it -- exactly what
                    # the MAP prior term is for. This is the honest fix to the
                    # documented identifiability landmine.


def _combined_sigma(pred, sp, sa):
    return jnp.sqrt(sa ** 2 + (sp * pred) ** 2)


def map_update(T, vmax_prior, km_prior, v, dose, tau, level_obs,
               omega_vmax=OMEGA_VMAX, omega_km=OMEGA_KM,
               sigma_prop=SIGMA_PROP, sigma_add=SIGMA_ADD, t_inf=1.0, steps=800):
    """MAP fit of (Vmax, Km) from one measured steady-state level. Same objective as
    the vanco fitter (combined-error residual + lognormal population prior), same
    Adam loop; V held at its covariate value (weakly informed by a single level)."""
    l_vmax0 = jnp.log(f32(vmax_prior))
    l_km0 = jnp.log(f32(km_prior))

    def phi(params):
        vmax = jnp.exp(params[0]); km = jnp.exp(params[1])
        ex = pheny_exposure(T, f32(dose), f32(tau), vmax, km, f32(v), t_inf=t_inf)
        s = _combined_sigma(ex["trough"], sigma_prop, sigma_add)
        resid = ((ex["trough"] - level_obs) / s) ** 2
        prior = (((params[0] - l_vmax0) / omega_vmax) ** 2
                 + ((params[1] - l_km0) / omega_km) ** 2)
        return resid + prior

    grad = jax.grad(phi)
    params = jnp.array([l_vmax0, l_km0], dtype=jnp.float32)
    m = jnp.zeros(2); vv = jnp.zeros(2)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.03
    for t in range(1, steps + 1):
        gt = grad(params)
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * gt * gt
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        params = params - lr * mhat / (jnp.sqrt(vhat) + eps)

    return {"vmax_prior": float(vmax_prior), "km_prior": float(km_prior),
            "vmax_post": float(jnp.exp(params[0])), "km_post": float(jnp.exp(params[1])),
            "final_obj": float(phi(params)),
            "nan": bool(jnp.any(jnp.isnan(params)))}


def synthetic_demo(T, vmax_prior, km_prior, v, vmax_true, km_true,
                   dose=200.0, tau=12.0):
    """Generate a 'measured' steady-state level from a TRUE patient (different Vmax),
    then MAP-recover the parameters from that single level."""
    ex = pheny_exposure(T, f32(dose), f32(tau), f32(vmax_true), f32(km_true), f32(v))
    level_obs = float(ex["trough"])
    res = map_update(T, vmax_prior, km_prior, v, dose, tau, f32(level_obs))
    res.update({"vmax_true": vmax_true, "km_true": km_true, "level_obs": level_obs,
                "dose": dose, "tau": tau})
    res["vmax_err_prior"] = abs(vmax_prior - vmax_true)
    res["vmax_err_post"] = abs(res["vmax_post"] - vmax_true)
    return res
