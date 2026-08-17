# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Vanco Tesseract #2b (STRETCH): TWO-compartment IV PK, analytic bi-exponential.

A SELECTABLE alternative to the locked one-compartment core (vanco_pk). Same
input/output contract (dose, tau -> conc curve, AUC24, peak, trough), so it slots
into the identical composition, optimizer and MAP-Bayesian stack. The one-comp
model remains the always-shippable entry; this path is promoted only after it
verifies (gradients no-NaN, AUC identity holds, optimizer + MAP work on it).

Parameters: CL (clearance), V1 (central volume), Q (inter-compartmental clearance),
V2 (peripheral volume). Micro rate constants:

    k10 = CL/V1     k12 = Q/V1     k21 = Q/V2
    sum = k10 + k12 + k21
    alpha, beta = (sum +/- sqrt(sum^2 - 4*k10*k21)) / 2        (alpha > beta > 0)

Unit-bolus central disposition is bi-exponential  c(t) = A e^-alpha t + B e^-beta t
with A = (alpha - k21)/(V1(alpha-beta)),  B = (k21 - beta)/(V1(alpha-beta)).
A constant infusion at rate R = D/t_inf integrates this impulse response:

    during infusion (0 <= e <= t_inf):
        C(e) = R [ A/alpha (1 - e^-alpha e) + B/beta (1 - e^-beta e) ]
    after infusion (e > t_inf):
        C(e) = R [ A/alpha (e^-alpha(e-t_inf) - e^-alpha e)
                 + B/beta (e^-beta (e-t_inf) - e^-beta e) ]

Steady state is built by SUPERPOSITION of N_DOSES prior doses spaced by tau, then
the last interval is read. Two-compartment kinetics capture the DISTRIBUTION phase
that one-compartment collapses, so the true post-infusion peak (and its decline
into the tissue compartment) is represented, which matters for AUC-guided dosing
in ICU / renal-impaired patients [Tesfamariam 2024; Radica Z.].

Linear-PK identity for verification: AUC24_ss = daily_dose / CL exactly (holds for
any linear compartment count).
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

N_DOSES = 40
N_T = 97


def _macro(cl, v1, q, v2):
    """Micro -> macro (alpha, beta) and bi-exponential coefficients A, B."""
    k10 = cl / v1
    k12 = q / v1
    k21 = q / v2
    s = k10 + k12 + k21
    disc = jnp.sqrt(jnp.maximum(s * s - 4.0 * k10 * k21, 1e-12))
    alpha = 0.5 * (s + disc)
    beta = 0.5 * (s - disc)
    denom = v1 * (alpha - beta)
    A = (alpha - k21) / denom
    B = (k21 - beta) / denom
    return alpha, beta, A, B


def _ss_conc_at(s, dose, cl, v1, q, v2, tau, t_inf):
    """SS 2-comp central concentration at time s into the last interval."""
    R = dose / t_inf
    alpha, beta, A, B = _macro(cl, v1, q, v2)
    j = jnp.arange(N_DOSES, dtype=jnp.float32)
    t_abs = (N_DOSES - 1) * tau + s
    e = t_abs - j * tau  # elapsed since each prior dose (>= 0)

    during = R * (A / alpha * (1.0 - jnp.exp(-alpha * e))
                  + B / beta * (1.0 - jnp.exp(-beta * e)))
    post = R * (A / alpha * (jnp.exp(-alpha * (e - t_inf)) - jnp.exp(-alpha * e))
                + B / beta * (jnp.exp(-beta * (e - t_inf)) - jnp.exp(-beta * e)))
    contrib = jnp.where(e <= t_inf, during, post)
    return jnp.sum(contrib)


class InputSchema(BaseModel):
    dose: Differentiable[Array[..., Float32]] = Field(
        description="Single dose (mg)", default=np.float32(1000.0)
    )
    tau: Differentiable[Array[..., Float32]] = Field(
        description="Dosing interval (h)", default=np.float32(12.0)
    )
    cl: Differentiable[Array[..., Float32]] = Field(
        description="Clearance (L/h)", default=np.float32(5.0)
    )
    v1: Differentiable[Array[..., Float32]] = Field(
        description="Central volume V1 (L)", default=np.float32(7.5)
    )
    q: Differentiable[Array[..., Float32]] = Field(
        description="Inter-compartmental clearance Q (L/h)", default=np.float32(7.0)
    )
    v2: Differentiable[Array[..., Float32]] = Field(
        description="Peripheral volume V2 (L)", default=np.float32(42.0)
    )
    t_inf: Array[..., Float32] = Field(
        description="Infusion duration (h) — not differentiated (AUC is t_inf-invariant)",
        default=np.float32(1.0),
    )


class OutputSchema(BaseModel):
    conc: Differentiable[Array[..., Float32]] = Field(
        description="Steady-state concentration curve over one interval (mg/L)"
    )
    times: Array[..., Float32] = Field(description="Time grid within interval (h)")
    auc24: Differentiable[Array[..., Float32]] = Field(
        description="Steady-state AUC0-24 (mg*h/L)"
    )
    peak: Differentiable[Array[..., Float32]] = Field(
        description="SS peak = conc at end of infusion (mg/L)"
    )
    trough: Differentiable[Array[..., Float32]] = Field(
        description="SS trough = conc at end of interval (mg/L)"
    )


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    dose = inputs["dose"]
    tau = inputs["tau"]
    cl = inputs["cl"]
    v1 = inputs["v1"]
    q = inputs["q"]
    v2 = inputs["v2"]
    t_inf = inputs["t_inf"]

    grid = jnp.linspace(0.0, 1.0, N_T) * tau
    conc = jax.vmap(lambda s: _ss_conc_at(s, dose, cl, v1, q, v2, tau, t_inf))(grid)

    dt = tau / (N_T - 1)
    auc_interval = dt * (jnp.sum(conc) - 0.5 * (conc[0] + conc[-1]))
    auc24 = auc_interval * (24.0 / tau)

    peak = _ss_conc_at(t_inf, dose, cl, v1, q, v2, tau, t_inf)
    trough = _ss_conc_at(tau, dose, cl, v1, q, v2, tau, t_inf)
    return {"conc": conc, "times": grid, "auc24": auc24, "peak": peak, "trough": trough}


def apply(inputs: InputSchema) -> OutputSchema:
    """Two-compartment IV vancomycin steady-state forward model (stretch path)."""
    return apply_jit(inputs.model_dump())


def abstract_eval(abstract_inputs):
    is_shapedtype_dict = lambda x: type(x) is dict and (x.keys() == {"shape", "dtype"})
    is_shapedtype_struct = lambda x: isinstance(x, jax.ShapeDtypeStruct)
    jaxified_inputs = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(**x) if is_shapedtype_dict(x) else x,
        abstract_inputs.model_dump(),
        is_leaf=is_shapedtype_dict,
    )
    dynamic_inputs, static_inputs = eqx.partition(
        jaxified_inputs, filter_spec=is_shapedtype_struct
    )

    def wrapped_apply(dynamic_inputs):
        return apply_jit(eqx.combine(static_inputs, dynamic_inputs))

    jax_shapes = jax.eval_shape(wrapped_apply, dynamic_inputs)
    return jax.tree.map(
        lambda x: (
            {"shape": x.shape, "dtype": str(x.dtype)} if is_shapedtype_struct(x) else x
        ),
        jax_shapes,
        is_leaf=is_shapedtype_struct,
    )


def jacobian_vector_product(inputs: InputSchema, jvp_inputs, jvp_outputs, tangent_vector):
    return jvp_jit(inputs.model_dump(), tuple(jvp_inputs), tuple(jvp_outputs), tangent_vector)


def vector_jacobian_product(inputs: InputSchema, vjp_inputs, vjp_outputs, cotangent_vector):
    return vjp_jit(inputs.model_dump(), tuple(vjp_inputs), tuple(vjp_outputs), cotangent_vector)


def jacobian(inputs: InputSchema, jac_inputs, jac_outputs):
    return jac_jit(inputs.model_dump(), tuple(jac_inputs), tuple(jac_outputs))


@eqx.filter_jit
def jvp_jit(inputs, jvp_inputs, jvp_outputs, tangent_vector):
    filtered_apply = filter_func(apply_jit, inputs, jvp_outputs)
    return jax.jvp(
        filtered_apply,
        [flatten_with_paths(inputs, include_paths=jvp_inputs)],
        [tangent_vector],
    )[1]


@eqx.filter_jit
def vjp_jit(inputs, vjp_inputs, vjp_outputs, cotangent_vector):
    filtered_apply = filter_func(apply_jit, inputs, vjp_outputs)
    _, vjp_func = jax.vjp(
        filtered_apply, flatten_with_paths(inputs, include_paths=vjp_inputs)
    )
    return vjp_func(cotangent_vector)[0]


@eqx.filter_jit
def jac_jit(inputs, jac_inputs, jac_outputs):
    filtered_apply = filter_func(apply_jit, inputs, jac_outputs)
    return jax.jacfwd(filtered_apply)(
        flatten_with_paths(inputs, include_paths=jac_inputs)
    )
