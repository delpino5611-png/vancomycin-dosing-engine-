# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Vanco Tesseract #2: one-compartment IV PK, multi-dose superposition to steady state.

Intermittent IV infusion, one-compartment linear kinetics (Ke, V; CL = Ke*V).
Single-dose concentration for a dose given at t=0, infused over t_inf at rate R=D/t_inf:

    during infusion (0 <= s <= t_inf):  C(s) = (R/CL) * (1 - exp(-Ke*s))
    post infusion    (s >  t_inf):      C(s) = (R/CL) * (1 - exp(-Ke*t_inf)) * exp(-Ke*(s - t_inf))

Steady state is built by SUPERPOSITION: sum the residual contribution of N_DOSES
prior doses spaced by the interval tau, then read the LAST interval (which is at
steady state for N_DOSES large). Outputs the SS concentration-time curve over one
interval, the (differentiable, trapezoidal) AUC24, and SS peak/trough.

Structured so a 2-compartment variant (CL, V1, Q, V2 via a diffrax ODE) can slot in
behind the same input/output contract later (stretch). One-compartment is the
LOCKED core (matches bedside vanco + finishable).

Linear-PK identity used for verification: AUC24_ss = daily_dose / CL exactly.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

N_DOSES = 40  # doses in the superposition (>= 5 half-lives even for ESRD q24 -> SS)
N_T = 97  # points on the one-interval SS curve grid


def _ss_conc_at(s, dose, ke, v, tau, t_inf):
    """SS concentration at time s into the last dosing interval (0 <= s <= tau)."""
    cl = ke * v
    R = dose / t_inf
    j = jnp.arange(N_DOSES, dtype=jnp.float32)
    t_abs = (N_DOSES - 1) * tau + s
    elapsed = t_abs - j * tau  # >= 0 for every prior dose at this evaluation point
    during = (R / cl) * (1.0 - jnp.exp(-ke * elapsed))
    post = (R / cl) * (1.0 - jnp.exp(-ke * t_inf)) * jnp.exp(-ke * (elapsed - t_inf))
    contrib = jnp.where(elapsed <= t_inf, during, post)
    return jnp.sum(contrib)


class InputSchema(BaseModel):
    dose: Differentiable[Array[..., Float32]] = Field(
        description="Single dose (mg)", default=np.float32(1000.0)
    )
    tau: Differentiable[Array[..., Float32]] = Field(
        description="Dosing interval (h)", default=np.float32(12.0)
    )
    ke: Differentiable[Array[..., Float32]] = Field(
        description="Elimination rate (1/h)", default=np.float32(0.09)
    )
    v: Differentiable[Array[..., Float32]] = Field(
        description="Volume of distribution (L)", default=np.float32(56.0)
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
    ke = inputs["ke"]
    v = inputs["v"]
    t_inf = inputs["t_inf"]

    grid = jnp.linspace(0.0, 1.0, N_T) * tau  # 0..tau
    conc = jax.vmap(lambda s: _ss_conc_at(s, dose, ke, v, tau, t_inf))(grid)

    dt = tau / (N_T - 1)
    auc_interval = dt * (jnp.sum(conc) - 0.5 * (conc[0] + conc[-1]))  # trapezoid
    auc24 = auc_interval * (24.0 / tau)

    peak = _ss_conc_at(t_inf, dose, ke, v, tau, t_inf)  # end of infusion
    trough = _ss_conc_at(tau, dose, ke, v, tau, t_inf)  # end of interval
    return {"conc": conc, "times": grid, "auc24": auc24, "peak": peak, "trough": trough}


def apply(inputs: InputSchema) -> OutputSchema:
    """One-compartment IV vancomycin steady-state forward model."""
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
