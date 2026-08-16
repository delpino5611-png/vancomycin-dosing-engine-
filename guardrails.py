# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Guardrails layer — hard caps + soft warnings for a vancomycin regimen.

Hard blocks (regimen is clinically unsafe / off-protocol) vs soft warnings
(permissible but flagged). Per vanco_engine_spec.md:

  HARD BLOCK:
    - single dose  > 2000 mg
    - infusion rate > 1000 mg/hr  (>=10 mg/min -> Red Man / infusion reaction)
    - SS trough    > 20 mg/L      (nephrotoxicity ceiling)
    - AUC24        > 600 mg*h/L   (exposure ceiling)
    - interval not allowed for the patient's CrCl band
  SOFT WARN:
    - SS trough < 10 mg/L         (legacy subtherapeutic)
    - AUC24     < 400 mg*h/L      (below AUC/MIC target)
    - CrCl      < 30 mL/min       (consider extended interval / level-guided dosing)

The optimizer differentiates through the smooth PK+loss stack; this layer is the
DISCRETE, non-differentiable safety wrapper applied to the resulting regimen.
"""
import math

MAX_DOSE = 2000.0  # mg
MAX_RATE = 1000.0  # mg/hr
TROUGH_CEIL = 20.0  # mg/L
AUC_CEIL = 600.0  # mg*h/L
AUC_FLOOR = 400.0  # mg*h/L
TROUGH_FLOOR = 10.0  # mg/L
DOSE_CHOICES = [250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0]
INTERVAL_SET = [8.0, 12.0, 24.0]


def infusion_time(dose):
    """Spec infusion durations (h): 1h<=1000, 1.5h<=1500, 2h<=2000, 3h<=3000."""
    if dose <= 1000.0:
        return 1.0
    if dose <= 1500.0:
        return 1.5
    if dose <= 2000.0:
        return 2.0
    return 3.0


def feasible_intervals(crcl):
    """Interval-by-CrCl (spec): >=80 -> q8-12; 50-79 -> q12-24; 30-49 -> q24; <30 -> q24 (warn)."""
    if crcl >= 80.0:
        return [8.0, 12.0]
    if crcl >= 50.0:
        return [12.0, 24.0]
    if crcl >= 30.0:
        return [24.0]
    return [24.0]  # <30: longest in the discrete set; a warning is attached separately


def check_regimen(dose, tau, peak, trough, auc24, crcl):
    """Return {'ok': bool, 'hard_block': [...], 'soft_warn': [...], 'rate': mg/hr}."""
    hard, soft = [], []
    t_inf = infusion_time(dose)
    rate = dose / t_inf

    if dose > MAX_DOSE:
        hard.append(f"single dose {dose:.0f} mg > {MAX_DOSE:.0f} mg cap")
    if rate > MAX_RATE + 1e-6:
        hard.append(f"infusion rate {rate:.0f} mg/hr > {MAX_RATE:.0f} mg/hr (Red Man risk)")
    if trough > TROUGH_CEIL + 1e-6:
        hard.append(f"SS trough {trough:.1f} > {TROUGH_CEIL:.0f} mg/L (nephrotoxicity)")
    if auc24 > AUC_CEIL + 1e-6:
        hard.append(f"AUC24 {auc24:.0f} > {AUC_CEIL:.0f} mg*h/L ceiling")
    if tau not in feasible_intervals(crcl):
        hard.append(
            f"interval q{tau:.0f}h not allowed for CrCl {crcl:.0f} "
            f"(allowed: {['q%.0f' % x for x in feasible_intervals(crcl)]})"
        )

    if trough < TROUGH_FLOOR:
        soft.append(f"SS trough {trough:.1f} < {TROUGH_FLOOR:.0f} mg/L (subtherapeutic)")
    if auc24 < AUC_FLOOR:
        soft.append(f"AUC24 {auc24:.0f} < {AUC_FLOOR:.0f} mg*h/L (below AUC/MIC target)")
    if crcl < 30.0:
        soft.append(
            f"CrCl {crcl:.0f} < 30 mL/min - consider extended interval / level-guided dosing"
        )

    return {"ok": len(hard) == 0, "hard_block": hard, "soft_warn": soft, "rate": rate,
            "t_inf": t_inf}


def snap_dose(dose):
    """Snap a continuous optimized dose to the nearest practical dose, capped at MAX_DOSE."""
    dose = min(dose, MAX_DOSE)
    return min(DOSE_CHOICES, key=lambda d: abs(d - dose))


# --- loading dose (Rybak 2020: 25-30 mg/kg actual body weight, caution > 3 g) ---
LOADING_CAP = 3000.0  # mg
LOADING_CHOICES = [500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0,
                   2250.0, 2500.0, 2750.0, 3000.0]


def snap_dose_loading(dose):
    """Snap a loading dose to the 250 mg grid, capped at 3000 mg (loading allows > 2 g)."""
    dose = min(dose, LOADING_CAP)
    return min(LOADING_CHOICES, key=lambda d: abs(d - dose))


def mic_alternative_agent_flag(mic):
    """Rybak 2020: at MIC >= 2 mg/L, chasing AUC/MIC 400-600 demands an AUC that is
    futile and nephrotoxic — the guideline response is to CONSIDER AN ALTERNATIVE
    AGENT, not to dose vancomycin higher. Return a flag string (or None)."""
    if mic >= 2.0:
        return (f"MIC {mic:.1f} mg/L >= 2: guideline response is CONSIDER ALTERNATIVE "
                f"AGENT, not a higher vancomycin AUC (target AUC/MIC 400-600 would need "
                f"AUC {int(400 * mic)}-{int(600 * mic)}, futile + nephrotoxic).")
    return None
