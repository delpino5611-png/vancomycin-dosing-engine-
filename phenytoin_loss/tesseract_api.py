# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Phenytoin Tesseract (generality demo): exposure objective for a CONCENTRATION
target -- a DIFFERENT clinical loss from the vancomycin AUC24/MIC objective.

Vancomycin drives an exposure RATIO (AUC24/MIC into 400 to 600). Phenytoin is
monitored by the measured serum LEVEL: total 10 to 20 mg/L (free 1 to 2, fu ~0.1)
[Kane 2016 / MDCalc]. So the loss targets the steady-state trough LEVEL to the
center of the band and adds smooth one-sided safety penalties:

    L = (trough - target_level)^2
        + W_TOX * relu(trough - TOX_CEIL)^2       (nystagmus/ataxia above ~20)
        + W_SUB * relu(SUBTHER_FLOOR - trough)^2  (breakthrough seizures below ~10)
        + W_PEAK_TOX * relu(peak - PEAK_TOX)^2     (transient peak toxicity)

relu(x)^2 is C^1, so the objective is smooth for gradient descent; penalties are
one-sided (they bite only when a bound is crossed). The guardrail layer holds the
hard discrete blocks. Same structure as the vanco loss, different target semantics
(a concentration, not an exposure ratio) -- the reusable optimizer minimizes either.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

TOX_CEIL = 20.0       # mg/L total, toxicity ceiling (nystagmus ~20, ataxia ~30)
SUBTHER_FLOOR = 10.0  # mg/L total, subtherapeutic floor
PEAK_TOX = 25.0       # mg/L transient peak toxicity soft ceiling
W_TOX = 5.0
W_SUB = 5.0
W_PEAK_TOX = 2.0


def _relu(x):
    return jnp.maximum(x, 0.0)


class InputSchema(BaseModel):
    trough: Differentiable[Array[..., Float32]] = Field(
        description="SS trough total phenytoin level (mg/L) — the monitored value"
    )
    peak: Differentiable[Array[..., Float32]] = Field(
        description="SS peak total phenytoin level (mg/L)"
    )
    target_level: Differentiable[Array[..., Float32]] = Field(
        description="Target total level (mg/L); center of the 10 to 20 band",
        default=np.float32(15.0),
    )


class OutputSchema(BaseModel):
    loss: Differentiable[Array[..., Float32]] = Field(description="Dosing objective")
    level_term: Differentiable[Array[..., Float32]] = Field(
        description="(trough-target)^2"
    )
    tox_penalty: Differentiable[Array[..., Float32]] = Field(
        description="Toxicity penalty (trough + peak)"
    )
    sub_penalty: Differentiable[Array[..., Float32]] = Field(
        description="Subtherapeutic penalty"
    )


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    trough = inputs["trough"]
    peak = inputs["peak"]
    target = inputs["target_level"]

    level_term = (trough - target) ** 2
    tox_penalty = (W_TOX * _relu(trough - TOX_CEIL) ** 2
                   + W_PEAK_TOX * _relu(peak - PEAK_TOX) ** 2)
    sub_penalty = W_SUB * _relu(SUBTHER_FLOOR - trough) ** 2
    loss = level_term + tox_penalty + sub_penalty
    return {"loss": loss, "level_term": level_term,
            "tox_penalty": tox_penalty, "sub_penalty": sub_penalty}


def apply(inputs: InputSchema) -> OutputSchema:
    """Differentiable dosing loss for a phenytoin concentration target."""
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
