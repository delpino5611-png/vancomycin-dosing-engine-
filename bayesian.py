# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""MAP-Bayesian individualization — the finale (formalized objective).

Population Matzke/CG prior (from covariates via T1) + sparse measured levels
-> posterior PK parameters by autodiff THROUGH the served vanco_pk Tesseract (T2).
This is the InsightRX-style workflow: the population prior is good (Onor: r>0.7),
sparse noisy levels sharpen it WITHOUT overfitting.

Explicit MAP (penalized weighted least squares) objective:

    Phi(theta) = SUM_j (C_obs_j - C_pred_j)^2 / sigma_j^2        [residual / Sigma]
               + SUM_i (theta_i - theta_hat_i)^2 / omega_i^2     [pop prior / Omega]

- theta_i = log-transformed PK parameters (log Ke, log V); positivity for free and
  the standard lognormal interindividual-variability (IIV) form. omega_i is the
  log-scale prior SD, ~= the published vancomycin PopPK coefficient of variation
  (CV ~30-50%): omega_ke = 0.40, omega_v = 0.30 here [Radica Z.; Rybak 2020].
- sigma_j is the assay/residual error, a COMBINED proportional + additive model
  sigma_j = sqrt(sigma_add^2 + (sigma_prop * C_pred_j)^2), sigma_prop ~= 12%,
  sigma_add ~= 2 mg/L. Proportional error down-weights high concentrations; the
  additive floor keeps low troughs identifiable.

The Omega term is what prevents overfitting a single noisy/outlier level: the
posterior is pulled toward the levels only as far as the population prior allows.
Gradient descent (Adam) differentiates the level predictions through the PK
Tesseract over HTTP. This is the inverse problem the framework's composable
autodiff is built for, and the same objective drives the 4-parameter 2-compartment
stretch (see bayesian_2c.map_update_2c).
"""
import jax
import jax.numpy as jnp

from engine import pk_exposure, pk_exposure_2c

f32 = lambda x: jnp.float32(x)

# default residual (Sigma) and prior (Omega) hyperparameters
SIGMA_PROP = 0.12   # proportional assay error (~12%)
SIGMA_ADD = 2.0     # additive assay error (mg/L)
OMEGA_KE = 0.40     # log-scale IIV SD on Ke (~40% CV)
OMEGA_V = 0.30      # log-scale IIV SD on V  (~30% CV)


def _combined_sigma(pred, sigma_prop, sigma_add):
    """Combined proportional + additive residual SD at a predicted concentration."""
    return jnp.sqrt(sigma_add ** 2 + (sigma_prop * pred) ** 2)


def map_update(T, ke_prior, v_prior, dose, tau, peak_obs, trough_obs,
               omega_ke=OMEGA_KE, omega_v=OMEGA_V,
               sigma_prop=SIGMA_PROP, sigma_add=SIGMA_ADD,
               t_inf=1.0, steps=800, prior_weight=1.0):
    """Return prior, posterior (Ke,V) and the objective, minimizing Phi above.

    prior_weight scales the Omega term: 1.0 = full MAP; 0.0 = pure MLE (no prior,
    used only to demonstrate that the prior is what resists outlier overfitting).
    """
    l_ke_prior = jnp.log(f32(ke_prior))
    l_v_prior = jnp.log(f32(v_prior))
    pw = f32(prior_weight)

    def phi(params):
        l_ke, l_v = params[0], params[1]
        ke = jnp.exp(l_ke)
        v = jnp.exp(l_v)
        ex = pk_exposure(T, f32(dose), f32(tau), ke, v, t_inf=t_inf)
        # residual term (combined error model), evaluated at the model prediction
        s_peak = _combined_sigma(ex["peak"], sigma_prop, sigma_add)
        s_trough = _combined_sigma(ex["trough"], sigma_prop, sigma_add)
        resid = (((ex["peak"] - peak_obs) / s_peak) ** 2
                 + ((ex["trough"] - trough_obs) / s_trough) ** 2)
        # population-prior term (Omega), lognormal IIV
        prior = (((l_ke - l_ke_prior) / omega_ke) ** 2
                 + ((l_v - l_v_prior) / omega_v) ** 2)
        return resid + pw * prior

    grad = jax.grad(phi)
    params = jnp.array([l_ke_prior, l_v_prior], dtype=jnp.float32)
    m = jnp.zeros(2); vv = jnp.zeros(2)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.03
    for t in range(1, steps + 1):
        gt = grad(params)
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * gt * gt
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        params = params - lr * mhat / (jnp.sqrt(vhat) + eps)

    ke_post = float(jnp.exp(params[0]))
    v_post = float(jnp.exp(params[1]))
    return {
        "ke_prior": float(ke_prior), "v_prior": float(v_prior),
        "ke_post": ke_post, "v_post": v_post,
        "cl_prior": float(ke_prior) * float(v_prior),
        "cl_post": ke_post * v_post,
        "final_obj": float(phi(params)),
    }


def synthetic_demo(T, ke_prior, v_prior, ke_true, v_true, dose=1250.0, tau=12.0,
                   noise=0.0, seed=0):
    """Generate 'measured' peak+trough from a TRUE patient, then MAP-recover params."""
    ex_true = pk_exposure(T, f32(dose), f32(tau), f32(ke_true), f32(v_true))
    peak_obs = float(ex_true["peak"])
    trough_obs = float(ex_true["trough"])
    if noise > 0:
        key = jax.random.PRNGKey(seed)
        n = jax.random.normal(key, (2,)) * noise
        peak_obs += float(n[0]); trough_obs += float(n[1])

    res = map_update(T, ke_prior, v_prior, dose, tau, f32(peak_obs), f32(trough_obs))
    res.update({"ke_true": ke_true, "v_true": v_true,
                "peak_obs": peak_obs, "trough_obs": trough_obs,
                "dose": dose, "tau": tau})
    # distance-to-truth improvement
    res["ke_err_prior"] = abs(ke_prior - ke_true)
    res["ke_err_post"] = abs(res["ke_post"] - ke_true)
    res["v_err_prior"] = abs(v_prior - v_true)
    res["v_err_post"] = abs(res["v_post"] - v_true)
    return res


def outlier_robustness_demo(T, ke_prior, v_prior, ke_true, v_true,
                            dose=1250.0, tau=12.0, trough_mult=0.5, seed=0):
    """Show the prior (Omega) resists a single outlier level.

    Generate the TRUE peak+trough, then corrupt the TROUGH into a single spurious
    LOW level (x trough_mult, e.g. a level drawn too early or a lab error). A low
    trough falsely implies fast clearance (high Ke). Fit twice:
      - MAP  (prior_weight=1): the Omega term anchors the posterior near truth.
      - MLE  (prior_weight=0): no prior, the fit chases the outlier.
    Returns both, plus the clean MAP fit, with distance-to-truth for each so the
    demo can show MAP stays near truth while MLE overfits the bad level.
    """
    ex_true = pk_exposure(T, f32(dose), f32(tau), f32(ke_true), f32(v_true))
    peak_true = float(ex_true["peak"])
    trough_true = float(ex_true["trough"])
    trough_outlier = trough_true * trough_mult  # a single spurious low level

    clean = map_update(T, ke_prior, v_prior, dose, tau,
                       f32(peak_true), f32(trough_true), prior_weight=1.0)
    mapfit = map_update(T, ke_prior, v_prior, dose, tau,
                        f32(peak_true), f32(trough_outlier), prior_weight=1.0)
    mlefit = map_update(T, ke_prior, v_prior, dose, tau,
                        f32(peak_true), f32(trough_outlier), prior_weight=0.0)

    def err(fit):
        return (abs(fit["ke_post"] - ke_true), abs(fit["v_post"] - v_true))

    ke_clean, v_clean = err(clean)
    ke_map, v_map = err(mapfit)
    ke_mle, v_mle = err(mlefit)
    return {
        "ke_true": ke_true, "v_true": v_true,
        "ke_prior": ke_prior, "v_prior": v_prior,
        "peak_true": peak_true, "trough_true": trough_true,
        "trough_outlier": trough_outlier, "trough_mult": trough_mult,
        "clean": clean, "map": mapfit, "mle": mlefit,
        "ke_err_clean": ke_clean, "v_err_clean": v_clean,
        "ke_err_map": ke_map, "v_err_map": v_map,
        "ke_err_mle": ke_mle, "v_err_mle": v_mle,
        # prior error (distance of the population prior itself from truth)
        "ke_err_prior": abs(ke_prior - ke_true),
        "v_err_prior": abs(v_prior - v_true),
    }


# ===========================================================================
# 2-COMPARTMENT MAP (STRETCH) — the 4-parameter inverse problem where the
# gradient decisively beats grid search (the design-review "necessity rests on
# dimensionality" point). Same Phi objective, four log-parameters.
# ===========================================================================
def map_update_2c(T, prior, dose, tau, peak_obs, trough_obs,
                  omega=(0.40, 0.30, 0.50, 0.40),
                  sigma_prop=SIGMA_PROP, sigma_add=SIGMA_ADD,
                  t_inf=1.0, steps=1500):
    """4-parameter (CL, V1, Q, V2) MAP fit via autodiff through the 2-comp Tesseract.

    prior = dict(cl, v1, q, v2). omega = log-scale IIV SDs for (CL,V1,Q,V2).
    Same Phi as map_update: combined-error residual + lognormal population prior.
    """
    theta0 = jnp.array([jnp.log(f32(prior["cl"])), jnp.log(f32(prior["v1"])),
                        jnp.log(f32(prior["q"])), jnp.log(f32(prior["v2"]))],
                       dtype=jnp.float32)
    om = jnp.array(omega, dtype=jnp.float32)

    def phi(theta):
        cl, v1, q, v2 = [jnp.exp(theta[i]) for i in range(4)]
        ex = pk_exposure_2c(T, f32(dose), f32(tau), cl, v1, q, v2, t_inf=t_inf)
        s_peak = _combined_sigma(ex["peak"], sigma_prop, sigma_add)
        s_trough = _combined_sigma(ex["trough"], sigma_prop, sigma_add)
        resid = (((ex["peak"] - peak_obs) / s_peak) ** 2
                 + ((ex["trough"] - trough_obs) / s_trough) ** 2)
        prior_term = jnp.sum(((theta - theta0) / om) ** 2)
        return resid + prior_term

    grad = jax.grad(phi)
    theta = theta0
    m = jnp.zeros(4); vv = jnp.zeros(4)
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.02
    for t in range(1, steps + 1):
        gt = grad(theta)
        m = b1 * m + (1 - b1) * gt
        vv = b2 * vv + (1 - b2) * gt * gt
        mhat = m / (1 - b1 ** t)
        vhat = vv / (1 - b2 ** t)
        theta = theta - lr * mhat / (jnp.sqrt(vhat) + eps)

    post = {k: float(jnp.exp(theta[i])) for i, k in enumerate(("cl", "v1", "q", "v2"))}
    return {"prior": dict(prior), "post": post, "final_obj": float(phi(theta)),
            "nan": bool(jnp.any(jnp.isnan(theta)))}


def synthetic_demo_2c(T, prior, cl_true, v1_true, q_true, v2_true,
                      dose=1250.0, tau=12.0):
    """Generate 'measured' peak+trough from a TRUE 2-comp patient, then MAP-recover."""
    ex = pk_exposure_2c(T, f32(dose), f32(tau), f32(cl_true), f32(v1_true),
                        f32(q_true), f32(v2_true))
    peak_obs = float(ex["peak"]); trough_obs = float(ex["trough"])
    res = map_update_2c(T, prior, dose, tau, f32(peak_obs), f32(trough_obs))
    truth = {"cl": cl_true, "v1": v1_true, "q": q_true, "v2": v2_true}
    res["truth"] = truth
    res["peak_obs"] = peak_obs; res["trough_obs"] = trough_obs
    # distance-to-truth for the two identifiable params (CL, V1)
    for k in ("cl", "v1", "q", "v2"):
        res[f"{k}_err_prior"] = abs(prior[k] - truth[k])
        res[f"{k}_err_post"] = abs(res["post"][k] - truth[k])
    return res
