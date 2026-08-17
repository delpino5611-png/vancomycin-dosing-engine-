# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Vanco Tesseract #1: CKD physiology / covariate -> PK-parameter map.

Maps patient covariates (age, weight, height, serum creatinine, sex) to the
vancomycin one-compartment PK parameters via published, differentiable formulas:

    IBW    (Devine)        male   50   + 2.3*(height_in - 60)
                           female 45.5 + 2.3*(height_in - 60)
    AdjBW                  IBW + 0.4*(ABW - IBW)                 [Onor 2020, best correlate]
    CrCl   (Cockcroft-Gault, AdjBW):
                           (140-age)*AdjBW / (72*SCr) * (0.85 if female)
    Ke     (Matzke prior)  0.00083*CrCl + 0.0044   (1/h)        [Matzke 1984 / Onor 2020]
    V                      0.7 * weight (L)                      [Rybak/Bauer]
    CL                     Ke * V (L/h)

Vancomycin is ~90% renally cleared, so CrCl is the dominant, central covariate:
the renal-impairment effect flows straight into Ke/CL. This is the CKD physiology
module that makes the downstream dose optimization patient-specific.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, Field
from tesseract_core.runtime import Array, Differentiable, Float32
from tesseract_core.runtime.tree_transforms import filter_func, flatten_with_paths

# physiologic clamp on CrCl (mL/min) — a guardrail against the CG formula
# exploding at very low SCr (keeps gradients finite; Gemini NaN caution)
CRCL_MIN = np.float32(5.0)
CRCL_MAX = np.float32(160.0)


class InputSchema(BaseModel):
    age: Differentiable[Array[..., Float32]] = Field(
        description="Age (years), adults 18-89", default=np.float32(40.0)
    )
    weight: Differentiable[Array[..., Float32]] = Field(
        description="Actual body weight ABW (kg)", default=np.float32(80.0)
    )
    height_in: Differentiable[Array[..., Float32]] = Field(
        description="Height (inches)", default=np.float32(70.0)
    )
    scr: Differentiable[Array[..., Float32]] = Field(
        description="Serum creatinine (mg/dL)", default=np.float32(1.0)
    )
    sex: Array[..., Float32] = Field(
        description="Sex flag: 0=male, 1=female (not differentiated)",
        default=np.float32(0.0),
    )


class OutputSchema(BaseModel):
    crcl: Differentiable[Array[..., Float32]] = Field(
        description="Cockcroft-Gault CrCl on AdjBW (mL/min)"
    )
    ke: Differentiable[Array[..., Float32]] = Field(
        description="Matzke elimination-rate prior (1/h)"
    )
    v: Differentiable[Array[..., Float32]] = Field(
        description="Volume of distribution (L)"
    )
    cl: Differentiable[Array[..., Float32]] = Field(description="Clearance CL=Ke*V (L/h)")
    ibw: Array[..., Float32] = Field(description="Ideal body weight (kg)")
    adjbw: Array[..., Float32] = Field(description="Adjusted body weight (kg)")


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    age = inputs["age"]
    weight = inputs["weight"]
    height_in = inputs["height_in"]
    scr = inputs["scr"]
    sex = inputs["sex"]  # 0=male, 1=female

    is_female = sex > 0.5
    ibw_base = jnp.where(is_female, 45.5, 50.0)
    ibw = ibw_base + 2.3 * (height_in - 60.0)
    adjbw = ibw + 0.4 * (weight - ibw)

    sex_factor = jnp.where(is_female, 0.85, 1.0)
    crcl_raw = (140.0 - age) * adjbw / (72.0 * scr) * sex_factor
    crcl = jnp.clip(crcl_raw, CRCL_MIN, CRCL_MAX)

    ke = 0.00083 * crcl + 0.0044  # Matzke prior (1/h)
    v = 0.7 * weight  # L
    cl = ke * v  # L/h
    return {"crcl": crcl, "ke": ke, "v": v, "cl": cl, "ibw": ibw, "adjbw": adjbw}


def apply(inputs: InputSchema) -> OutputSchema:
    """Covariate -> vancomycin PK-parameter map (CKD physiology module)."""
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
