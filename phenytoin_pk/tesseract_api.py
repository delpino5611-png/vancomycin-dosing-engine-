# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Phenytoin Tesseract (generality demo): one-compartment IV with SATURABLE
(Michaelis-Menten) elimination -- a genuinely DIFFERENT model STRUCTURE from the
vancomycin linear-superposition PK, served behind the SAME input/output contract.

Vancomycin elimination is first-order (linear): rate = Ke*C, so steady state is an
analytic sum of exponentials. Phenytoin is capacity-limited (nonlinear):

    V dC/dt = R_in(t) - Vmax * C / (Km + C)

There is NO constant half-life and NO closed-form superposition. Near saturation
(daily input rate approaching Vmax) the steady-state level rises hyperbolically --
the famous phenytoin trap where a small dose bump triples the level. This is where
a gradient-based optimizer is genuinely necessary, not a convenience.

Params (all published):
  Vmax  maximum metabolic rate (mg/h)    ~7 mg/kg/day (population 5 to 9)   [Spruill 2001]
  Km    concentration at half Vmax (mg/L) ~4 to 6 (4.3 Caucasian, 5.7 age 20 to 39) [Spruill 2001]
  V     volume of distribution (L)         ~0.65 L/kg (0.5 to 0.8)          [Winter, standard TDM]
Therapeutic total level 10 to 20 mg/L (free 1 to 2, fu ~0.1) [Kane 2016 / MDCalc].

NUMERICAL STABILITY (addresses the two documented Michaelis-Menten landmines):
  1. "dose > Vmax -> conc -> infinity -> stiff ODE -> NaN gradients."
     - The elimination term uses a clamped concentration (max(C,0)) and Km+C is
       strictly positive, so the vector field is finite and smooth everywhere.
     - Integration is a FIXED-STEP RK4 (lax.scan), never an adaptive stiff solver,
       so there is no solver blow-up. Over a finite horizon the level stays finite
       even above saturation (the trap is represented, not a crash). The optimizer's
       bounded reparameterization and the guardrail layer keep the search below Vmax.
  2. Warm start: integration begins at the analytic average-concentration fixed
     point Css ~= Km*R/(Vmax-R) (clamped), so a short horizon reaches the periodic
     steady state; the true nonlinear ODE is still what is integrated.

Steady-state identity used for verification (the nonlinear analog of vanco's
AUC = dose/CL): at steady state the mass ELIMINATED over one interval equals the
dose administered over that interval (mass balance).
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

N_INT = 12  # dosing intervals integrated to reach the periodic steady state
SPI = 48    # RK4 steps per interval (dt = tau / SPI)
N_T = SPI + 1  # samples on the reported last-interval steady-state curve


def _mm_rate(c, vmax, km):
    """Saturable (Michaelis-Menten) elimination rate (mg/h). Clamped, finite."""
    cpos = jnp.maximum(c, 0.0)
    return vmax * cpos / (km + cpos)


def _rate_in(t, dose, tau, t_inf):
    """Zero-order infusion input (mg/h): dose/t_inf during the first t_inf of each
    interval, else 0. dose enters only the magnitude, so gradients in dose are clean."""
    ti = jnp.mod(t, tau)
    return jnp.where(ti < t_inf, dose / t_inf, 0.0)


def _simulate(dose, tau, vmax, km, v, t_inf):
    """Integrate the nonlinear one-compartment MM ODE to steady state (fixed-step RK4,
    lax.scan) and return the last-interval concentration samples (mg/L, total)."""
    dt = tau / SPI

    # analytic average-concentration warm start (clamped so it is always finite/positive)
    r_rate = dose / tau  # mean input rate (mg/h)
    denom = jnp.maximum(vmax - r_rate, 0.05 * vmax)
    css0 = jnp.clip(km * r_rate / denom, 0.1, 200.0)
    a0 = css0 * v  # amount (mg)

    def field(t, a):
        c = a / v
        return _rate_in(t, dose, tau, t_inf) - _mm_rate(c, vmax, km)

    def step(a, i):
        t = i.astype(jnp.float32) * dt
        k1 = field(t, a)
        k2 = field(t + 0.5 * dt, a + 0.5 * dt * k1)
        k3 = field(t + 0.5 * dt, a + 0.5 * dt * k2)
        k4 = field(t + dt, a + dt * k3)
        a_next = a + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        a_next = jnp.maximum(a_next, 0.0)
        return a_next, a_next

    idx = jnp.arange(N_INT * SPI, dtype=jnp.float32)
    a_last, a_traj = jax.lax.scan(step, a0, idx)
    a_all = jnp.concatenate([a0[None], a_traj])  # length N_INT*SPI + 1
    c_all = a_all / v
    c_last = c_all[-(SPI + 1):]  # the final interval = the steady-state curve
    return c_last, dt


class InputSchema(BaseModel):
    dose: Differentiable[Array[..., Float32]] = Field(
        description="Dose per administration (mg)", default=np.float32(150.0)
    )
    tau: Differentiable[Array[..., Float32]] = Field(
        description="Dosing interval (h)", default=np.float32(12.0)
    )
    vmax: Differentiable[Array[..., Float32]] = Field(
        description="Max metabolic rate Vmax (mg/h)", default=np.float32(20.4)
    )
    km: Differentiable[Array[..., Float32]] = Field(
        description="Michaelis constant Km (mg/L, total)", default=np.float32(5.0)
    )
    v: Differentiable[Array[..., Float32]] = Field(
        description="Volume of distribution (L)", default=np.float32(45.5)
    )
    t_inf: Array[..., Float32] = Field(
        description="Infusion duration (h) — not differentiated", default=np.float32(1.0)
    )


class OutputSchema(BaseModel):
    conc: Differentiable[Array[..., Float32]] = Field(
        description="Steady-state total-concentration curve over one interval (mg/L)"
    )
    times: Array[..., Float32] = Field(description="Time grid within interval (h)")
    auc24: Differentiable[Array[..., Float32]] = Field(
        description="Steady-state AUC0-24 of total level (mg*h/L)"
    )
    peak: Differentiable[Array[..., Float32]] = Field(
        description="SS peak total level over the interval (mg/L)"
    )
    trough: Differentiable[Array[..., Float32]] = Field(
        description="SS trough total level = end of interval (mg/L)"
    )
    cavg: Differentiable[Array[..., Float32]] = Field(
        description="SS average total level over the interval (mg/L)"
    )


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    dose = inputs["dose"]
    tau = inputs["tau"]
    vmax = inputs["vmax"]
    km = inputs["km"]
    v = inputs["v"]
    t_inf = inputs["t_inf"]

    c_last, _ = _simulate(dose, tau, vmax, km, v, t_inf)
    grid = jnp.linspace(0.0, 1.0, N_T) * tau

    dt = tau / (N_T - 1)
    auc_interval = dt * (jnp.sum(c_last) - 0.5 * (c_last[0] + c_last[-1]))  # trapezoid
    auc24 = auc_interval * (24.0 / tau)
    cavg = auc_interval / tau

    peak = jnp.max(c_last)
    trough = c_last[-1]
    return {"conc": c_last, "times": grid, "auc24": auc24,
            "peak": peak, "trough": trough, "cavg": cavg}


def apply(inputs: InputSchema) -> OutputSchema:
    """One-compartment IV phenytoin with saturable Michaelis-Menten elimination."""
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
