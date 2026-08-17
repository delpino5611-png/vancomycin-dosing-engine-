# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Vanco Tesseract #3: exposure objective (differentiable dosing loss).

Converts steady-state exposure metrics into the scalar loss the optimizer
minimizes. Modern vanco target is AUC24/MIC 400-600 [Rybak 2020]. The efficacy
term now scales with MIC explicitly: it drives the ratio AUC24/MIC to a target
(default 500 = center of the 400-600 band). Trough-only dosing misses true AUC
exposure, so this loss targets the AUC/MIC ratio directly and adds smooth safety
penalties on the ABSOLUTE trough/peak concentrations (toxicity is not scaled by
MIC).

    ratio = AUC24 / MIC
    L = (ratio - target)^2
        + W_TROUGH_HI * relu(trough - TROUGH_CEIL)^2     (nephrotoxicity > 20)
        + W_TROUGH_LO * relu(TROUGH_FLOOR - trough)^2    (subtherapeutic < 10)
        + W_PEAK_HI  * relu(peak - PEAK_CEIL)^2          (excessive peak)

MIC default is 1.0 per Rybak 2020 ("assume MIC = 1 mg/L"); with MIC=1 the loss
is identical to the AUC-targeting form. Allowed alternatives 0.5 / 2.0. Note the
clinical caveat handled upstream (engine/guardrails): at MIC >= 2 the guideline
response is CONSIDER AN ALTERNATIVE AGENT, not chase a higher (futile and
nephrotoxic) AUC [Rybak 2020].

relu(x)^2 is C^1 and smooth-enough for gradient descent; penalties are one-sided
so they only bite when a bound is violated (a differentiable soft-constraint,
not a hard clip — the guardrail layer handles hard blocks).
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

TROUGH_CEIL = 20.0  # mg/L nephrotoxicity ceiling (spec)
TROUGH_FLOOR = 10.0  # mg/L subtherapeutic floor (legacy trough band)
PEAK_CEIL = 50.0  # mg/L excessive-peak soft ceiling
W_TROUGH_HI = 50.0
W_TROUGH_LO = 5.0
W_PEAK_HI = 20.0


def _relu(x):
    return jnp.maximum(x, 0.0)


class InputSchema(BaseModel):
    auc24: Differentiable[Array[..., Float32]] = Field(
        description="Steady-state AUC0-24 (mg*h/L)"
    )
    peak: Differentiable[Array[..., Float32]] = Field(description="SS peak (mg/L)")
    trough: Differentiable[Array[..., Float32]] = Field(description="SS trough (mg/L)")
    target_auc: Differentiable[Array[..., Float32]] = Field(
        description="Target AUC24/MIC ratio (mg*h/L per mg/L); center of 400-600 band",
        default=np.float32(500.0),
    )
    mic: Differentiable[Array[..., Float32]] = Field(
        description="MIC (mg/L); default 1.0 per Rybak 2020 assume-MIC=1",
        default=np.float32(1.0),
    )


class OutputSchema(BaseModel):
    loss: Differentiable[Array[..., Float32]] = Field(description="Dosing objective")
    auc_term: Differentiable[Array[..., Float32]] = Field(description="(AUC-target)^2")
    trough_penalty: Differentiable[Array[..., Float32]] = Field(
        description="Trough safety penalty"
    )
    peak_penalty: Differentiable[Array[..., Float32]] = Field(description="Peak penalty")


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    auc24 = inputs["auc24"]
    peak = inputs["peak"]
    trough = inputs["trough"]
    target = inputs["target_auc"]
    mic = inputs["mic"]

    ratio = auc24 / mic  # AUC24/MIC — the efficacy exposure metric (Rybak 2020)
    auc_term = (ratio - target) ** 2
    trough_penalty = (
        W_TROUGH_HI * _relu(trough - TROUGH_CEIL) ** 2
        + W_TROUGH_LO * _relu(TROUGH_FLOOR - trough) ** 2
    )
    peak_penalty = W_PEAK_HI * _relu(peak - PEAK_CEIL) ** 2
    loss = auc_term + trough_penalty + peak_penalty
    return {
        "loss": loss,
        "auc_term": auc_term,
        "trough_penalty": trough_penalty,
        "peak_penalty": peak_penalty,
    }


def apply(inputs: InputSchema) -> OutputSchema:
    """Differentiable dosing loss from steady-state exposure metrics."""
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
