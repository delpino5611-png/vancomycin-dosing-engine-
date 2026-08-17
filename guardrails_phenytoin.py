# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Guardrails layer for a PHENYTOIN regimen -- the discrete, non-differentiable
safety wrapper on top of the smooth optimizer, mirroring guardrails.py for vanco
but with phenytoin-specific rules.

Key structural difference from vancomycin: phenytoin is HEPATICALLY metabolized
(CYP2C9/2C19), so the interval is NOT chosen by CrCl. Intervals are driven by the
formulation and by keeping the level flat (long half-life -> q12 or q24 typical;
q8 for divided immediate-release). The dominant landmine is saturation: as the
daily input rate approaches Vmax the level runs away, so the guardrail flags a
daily rate that crosses a safe fraction of Vmax.

  HARD BLOCK:
    - single dose        > 400 mg           (typical max per administration)
    - total daily dose   > 600 mg           (usual adult ceiling; higher only w/ levels)
    - SS trough          > 20 mg/L           (toxicity)
    - daily input rate   > 0.9 * Vmax        (into the saturation asymptote)
  SOFT WARN:
    - SS trough          < 10 mg/L           (subtherapeutic)
    - daily input rate   > 0.8 * Vmax        (approaching saturation; small bumps jump)
"""
MAX_DOSE = 400.0        # mg per administration
MAX_DAILY = 600.0       # mg/day
TROUGH_CEIL = 20.0      # mg/L
TROUGH_FLOOR = 10.0     # mg/L
SAT_HARD = 0.90         # fraction of Vmax (daily input rate) that is a hard block
SAT_WARN = 0.80         # fraction of Vmax that is a soft warning

# per-administration dose grid on a 30 mg granularity. Phenytoin is supplied in
# 30 mg capsules precisely BECAUSE its saturable kinetics demand fine titration:
# near Vmax a 50 mg step can move the level by tens of mg/L. This fine grid is the
# clinical answer to the saturation trap the model exposes.
DOSE_CHOICES = [30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0,
                300.0, 330.0, 360.0, 390.0, 400.0]
INTERVAL_SET = [8.0, 12.0, 24.0]


def infusion_time(dose):
    """IV phenytoin/fosphenytoin infusion duration (h). Max IV rate 50 mg/min is far
    above these doses, so a nominal 1 h (or shorter) is always safe. Kept for the
    shared PK contract; the drug is usually oral, modeled as a short input."""
    return 1.0


def feasible_intervals(crcl=None):
    """Interval choices for phenytoin. Unlike vanco, NOT a function of CrCl (hepatic
    drug). crcl is accepted and ignored so the call site matches the vanco engine."""
    return [8.0, 12.0, 24.0]


def check_regimen(dose, tau, peak, trough, cavg, vmax):
    """Return {'ok', 'hard_block', 'soft_warn', 'daily', 'sat_frac'}.

    dose = mg per administration, tau = h, vmax = mg/h. daily input rate = dose/tau,
    daily dose = dose*24/tau."""
    hard, soft = [], []
    daily_dose = dose * 24.0 / tau
    rate = dose / tau                 # mean input rate (mg/h)
    sat_frac = rate / vmax if vmax > 0 else float("inf")

    if dose > MAX_DOSE + 1e-6:
        hard.append(f"single dose {dose:.0f} mg > {MAX_DOSE:.0f} mg cap")
    if daily_dose > MAX_DAILY + 1e-6:
        hard.append(f"daily dose {daily_dose:.0f} mg/day > {MAX_DAILY:.0f} mg/day cap")
    if trough > TROUGH_CEIL + 1e-6:
        hard.append(f"SS trough {trough:.1f} > {TROUGH_CEIL:.0f} mg/L (toxicity)")
    if sat_frac > SAT_HARD:
        hard.append(f"daily input {rate:.1f} mg/h is {sat_frac*100:.0f}% of Vmax "
                    f"({vmax:.1f}) - past the safe saturation limit")

    if trough < TROUGH_FLOOR:
        soft.append(f"SS trough {trough:.1f} < {TROUGH_FLOOR:.0f} mg/L (subtherapeutic)")
    if SAT_WARN < sat_frac <= SAT_HARD:
        soft.append(f"daily input {rate:.1f} mg/h is {sat_frac*100:.0f}% of Vmax - "
                    f"near saturation, small dose bumps cause large level jumps")

    return {"ok": len(hard) == 0, "hard_block": hard, "soft_warn": soft,
            "daily": daily_dose, "sat_frac": sat_frac}


def snap_dose(dose):
    """Snap a continuous optimized per-administration dose to the practical grid."""
    dose = min(dose, MAX_DOSE)
    return min(DOSE_CHOICES, key=lambda d: abs(d - dose))
