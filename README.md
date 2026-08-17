# Vancomycin Differentiable Dosing Engine

A model-informed precision-dosing engine for IV vancomycin, built from **three
composed Tesseracts** with **end-to-end JAX gradients** flowing across the HTTP
boundaries between them. Covariates -> PK parameters -> steady-state exposure ->
dosing loss, differentiated as one function; `jax.grad` optimizes the regimen and
runs a MAP-Bayesian individualization from sparse measured levels.

Built to `vanco_engine_spec.md` (LOCKED). All formulas are published:
Cockcroft-Gault 1976, Matzke 1984, Onor 2020, Rybak 2020. This is the real
engine, not the day-1 toy prototype.

---

## Architecture

```
 covariates ─►[T1 ckd_physiology]─► ke,v ─►[T2 vanco_pk]─► auc24,peak,trough ─►[T3 exposure_loss]─► loss
 (age,wt,ht,SCr,sex)                (dose,tau)                                   (AUC/MIC target)
        │                                                                              ▲
        └──────────────────── jax.grad flows end-to-end across all three ─────────────┘
```

**T1 `ckd_physiology/`** — the CKD physiology module. Covariates ->
IBW (Devine) -> AdjBW -> Cockcroft-Gault CrCl (AdjBW, sex-adjusted, physiologically
clamped 5-160) -> Matzke Ke prior `Ke = 0.00083*CrCl + 0.0044` -> `V = 0.7*wt`,
`CL = Ke*V`. Vancomycin is ~90% renally cleared, so CrCl is the dominant, central
driver — the renal-impairment effect is the core story, not a side adjustment.

**T2 `vanco_pk/`** — one-compartment IV PK, **multi-dose superposition to steady
state**. Intermittent infusion (during/post piecewise), sums `N_DOSES=40` prior
doses spaced by `tau`, reads the last (steady-state) interval. Outputs the SS
concentration curve, a differentiable trapezoidal **AUC24**, SS **peak** (end of
infusion) and **trough** (end of interval). Structured behind a clean input/output
contract so a **2-compartment variant (CL,V1,Q,V2 via a diffrax ODE) can slot in
later** (stretch). Linear-PK identity used for verification: `AUC24_ss = daily_dose/CL`.

**T3 `exposure_loss/`** — the differentiable dosing objective:
`L = (AUC24/MIC - 500)^2 + 50*relu(trough-20)^2 + 5*relu(10-trough)^2 + 20*relu(peak-50)^2`.
Targets the AUC/MIC ratio 400-600 directly (MIC an explicit input, default 1.0 per
Rybak 2020 "assume MIC=1"; allowed 0.5/2.0) — the modern Rybak-2020 shift
("trough-based dosing misses true AUC"). Safety penalties are on ABSOLUTE peak/trough
(toxicity is not MIC-scaled). `relu()^2` penalties are smooth one-sided soft-constraints;
hard blocks live in the guardrail layer. At MIC>=2 the engine flags "consider alternative
agent" rather than chasing a futile/nephrotoxic AUC (guardrails.mic_alternative_agent_flag).

**`guardrails.py`** — the discrete, non-differentiable safety wrapper.
Hard blocks: dose >2000 mg, rate >1000 mg/hr (Red Man), trough >20 mg/L, AUC >600,
interval not allowed for the CrCl band (>=80 -> q8/q12; 50-79 -> q12/q24; 30-49 ->
q24; <30 -> q24 + warn). Soft warns: trough <10, AUC <400, CrCl <30. Also the
spec infusion-duration table and 250 mg-grid dose snapping.

**T2b `vanco_pk_2c/`** — the two-compartment STRETCH path (CL, V1, Q, V2), an
analytic bi-exponential IV-infusion model with the same steady-state superposition
and the **identical input/output contract** as T2, so it slots into the same
composition, optimizer and MAP stack. Separate and selectable; the 1-comp core is
never replaced. Verified (gradients no-NaN, AUC identity exact, optimizer in-band,
4-param MAP runs) so it is marked **available**. Captures the distribution-phase
true peak the 1-comp collapses.

**`engine.py`** — composition + `jax.grad` optimizer. `verify_gradients()` checks
end-to-end autodiff vs finite differences; `optimize_regimen()` returns a **two-phase**
result: an empiric weight-based **loading dose** (`loading_dose()`: 25 mg/kg ABW, cap
3 g, Rybak 2020) plus the gradient-optimized maintenance regimen (bounded Adam on dose
per feasible interval, sigmoid reparam keeps dose in [250,2000] -> no NaN), then the
guardrail layer selects the best safe regimen. `verify_2c()`, `optimize_regimen_2c()`
drive the 2-comp path.

**`bayesian.py`** — the finale, with a formalized MAP objective
`Phi = SUM_j (C_obs-C_pred)^2/sigma_j^2 + SUM_i (theta_i-theta_hat_i)^2/omega_i^2`:
combined proportional+additive residual error (Sigma ~ 12% + 2 mg/L) and a lognormal
population-prior penalty (Omega ~ CV 30-50% on log Ke/V) — the prior is what prevents
overfitting a single noisy level (`outlier_robustness_demo` shows MAP < MLE under a bad
trough). `map_update` fits (Ke,V) for the 1-comp core; `map_update_2c` fits the 4-param
(CL,V1,Q,V2) 2-comp inverse — the high-dim problem where autodiff beats grid search.
All by autodiff **through the served PK Tesseract**.

**`servers.py`** — launches/tears down the 3 Docker-free `tesseract-runtime serve`
processes (ports 8031-8033), waits on `/health`. **`run_demo.py`** — runs the full
verification + 3-patient demo self-contained.

---

## How to run

Python 3.10 or newer. Install into a fresh virtual environment, then run the demo:

```
python -m venv venv
# Windows:        venv\Scripts\activate
# macOS / Linux:  source venv/bin/activate
pip install -r requirements.txt
python run_demo.py
```

`run_demo.py` launches the Tesseract servers itself (subprocess, including the
optional 2-comp path), runs the full verification suite (gradient, optimizer, MIC
flag, guardrails, formalized MAP + outlier robustness, 2-comp) and the two-phase
3-adult-patient demo, and tears the servers down. No Docker required (uses the
`tesseract-runtime` Docker-free serve path). CPU-only; runs in about 3 to 4 minutes.

To serve a module standalone (for `curl` / inspection):
```
# macOS / Linux:  export TESSERACT_API_PATH=vanco_pk/tesseract_api.py
# Windows:        set TESSERACT_API_PATH=vanco_pk\tesseract_api.py
tesseract-runtime serve --port 8032
```

---

## Verification numbers (from `run_demo.py`)

**1. End-to-end gradient** (autodiff across all 3 composed Tesseracts, no NaN):

| gradient | autodiff | finite-diff | rel. err |
|---|---|---|---|
| d(loss)/d(SCr)  through T1->T2->T3 | -82248.60 | -82252.05 | 4.2e-05 |
| d(loss)/d(dose) through T2->T3     | -86.35 | -86.35 | 7.5e-05 |

**2. Optimizer** — continuous `jax.grad` optimum hits AUC24=500 for all patients:

| patient | CrCl | Ke (1/h) | CL (L/h) | continuous optimum | clinical regimen | AUC24 | peak | trough |
|---|---|---|---|---|---|---|---|---|
| young-normal | 105.3 | 0.0918 | 5.14 | 1285 mg q12 (AUC 500) | 1250 mg q12 | 486 | 31.2 | 11.9 |
| elderly-CKD  | 22.5  | 0.0231 | 0.97 | 485 mg q24 (AUC 500)  | 500 mg q24  | 515 | 27.6 | 16.2 |
| AKI-ESRD     | 12.3  | 0.0146 | 0.87 | 436 mg q24 (AUC 500)  | 500 mg q24* | 574 | 28.1 | 20.1 |

\*AKI-ESRD: the continuous optimum (436 mg) is exact; snapping to the 250 mg
clinical grid forces 500 mg -> AUC 574, trough 20.1 -> **guardrail blocks it**
(trough >20) and warns to escalate to extended-interval / level-guided dosing.
This is correct behavior: fixed q8/q12/q24 dosing genuinely fails in ESRD.

**3. Guardrails** — a deliberately unsafe regimen (3000 mg q8, CrCl 23) is hard-blocked:
dose >2000, trough 364 >20, AUC 9266 >600, interval q8 not allowed for CrCl 23. 4/4 blocks fire.

**3b. MIC parameter** — at MIC=1 the maintenance dose is unchanged; at MIC=2 the
engine flags "consider alternative agent" (AUC/MIC 400-600 would demand AUC 800-1200,
futile + nephrotoxic) rather than chasing dose.

**4. Bayesian MAP update** (formalized Phi = combined-error residual + lognormal
population prior) — true patient clears 40% faster / 15% smaller V than covariates
predict; 2 levels sharpen the population prior toward truth:

| param | prior | posterior | truth | \|err\| prior -> post |
|---|---|---|---|---|
| Ke | 0.0918 | 0.1151 | 0.1285 | 0.0367 -> 0.0133 |
| V  | 56.0   | 52.3   | 47.6  | 8.40 -> 4.70 |

**4b. MAP outlier robustness** — a single spurious LOW trough (x0.5). The population
prior makes the posterior a shrinkage estimate that resists the bad level:

| fit | Ke \|err\| | V \|err\| |
|---|---|---|
| population prior      | 0.0367 | 8.40 |
| MAP (with prior)      | 0.0202 | 0.20 |
| MLE (no prior)        | 0.0630 | 7.26 |

MLE chases the outlier past the prior; MAP does not overfit.

**5. Two-compartment stretch (available)** — separate selectable path, verifies:

| check | result |
|---|---|
| d(loss)/d(dose) 2c->T3 vs finite-diff | -14.78 vs -14.75, relerr 1.8e-03, no NaN |
| AUC identity (model vs daily_dose/CL) | 389.13 vs 389.13, relerr 2.0e-07 |
| distribution phase (true peak vs 1-comp) | 74.2 vs 26 mg/L for same CL |
| optimizer continuous optimum | AUC 465 (in 400-600 band) |
| 4-param MAP inverse (no NaN) | CL err 2.06->0.04, V1 err 2.25->1.69 |

All checks **PASS**; the 1-comp core stays the always-shippable entry.

---

## Spec deviations / notes

- **Infusion time in the gradient path is fixed at 1.0 h** (differentiable path).
  This is exact for AUC24 (`AUC = daily_dose/CL` is infusion-time-invariant for
  linear PK) and only mildly affects peak/trough. The guardrail/reporting layer
  applies the spec-correct infusion duration (1.0-3.0 h by dose) for the final
  regimen. A step-function `t_inf(dose)` would be non-differentiable, so this is
  the clean split.
- **Discrete dose grid = 250 mg increments**, intervals = {8,12,24} h (spec set).
  The differentiable optimizer works in continuous dose then snaps; where the
  discrete grid cannot reach target (ESRD), the guardrail flags it rather than
  shipping an unsafe rounded regimen.
- **Bayesian fits Ke,V** (2-param) for the 1-comp core, and CL,V1,Q,V2 (4-param)
  for the 2-comp path — the high-dim inverse where gradient-based inference decisively
  beats grid search (the design-review "necessity rests on dimensionality" point). The
  forward 1-comp dose is near closed-form; the gradient earns its keep on the Bayesian
  inverse and the differentiable composition, not the forward dose (stated honestly in
  the writeup).
- **2-compartment PK** is now BUILT and verified as a separate selectable module
  (`vanco_pk_2c/`, same input/output contract); one-compartment remains the LOCKED,
  always-shippable core. Serve the 2-comp path with `serve_all(include_2c=True)`.
- **Loading dose** (`engine.loading_dose`): 25 mg/kg ABW, cap 3 g (Rybak 2020),
  reported as phase 1 before the maintenance regimen (phase 2).
- CrCl is clamped to [5,160] mL/min inside T1 to keep gradients finite at very low
  SCr (the design-review NaN caution).

## Files

- `ckd_physiology/tesseract_api.py` — T1 (covariates -> PK params)
- `vanco_pk/tesseract_api.py` — T2 (1-comp IV, SS superposition, AUC24) [LOCKED core]
- `vanco_pk_2c/tesseract_api.py` — T2b (2-comp analytic bi-exponential) [stretch, available]
- `exposure_loss/tesseract_api.py` — T3 (AUC/MIC differentiable loss, MIC input)
- `guardrails.py` — hard-block / soft-warn safety layer + loading-dose snap + MIC flag
- `engine.py` — composition, gradient verification, regimen optimizer, loading dose, 2-comp path
- `bayesian.py` — MAP-Bayesian individualization (formalized Phi; 1-comp + 4-param 2-comp)
- `servers.py` — Docker-free server launch/teardown
- `run_demo.py` — full verification + 3-patient demo
- `recovery_experiment.py` — parameter-recovery validation (200 virtual patients; recovers Ke/V/AUC from 2 noisy levels, quantifies prior-vs-no-prior robustness)
- `*/tesseract_config.yaml`, `*/tesseract_requirements.txt` — per-module packaging

### Phenytoin generality demo (same engine, saturable Michaelis-Menten PK)

- `phenytoin_pk/tesseract_api.py` — saturable (Michaelis-Menten) PK module: nonlinear ODE integrated with a fixed-step solver, in place of vanco's linear superposition
- `phenytoin_loss/tesseract_api.py` — clinical loss on total serum level (10-20 mg/L target) in place of the AUC/MIC exposure ratio
- `engine_phenytoin.py` — composition + gradient verification + regimen optimizer for phenytoin (reuses the shared physiology module, optimizer, and Bayesian fitter unchanged)
- `guardrails_phenytoin.py` — discrete safety wrapper for phenytoin dosing
- `servers_phenytoin.py` — Docker-free server launch/teardown for the phenytoin modules
- `demo_phenytoin.py` — full phenytoin verification + normal/fast/slow-metabolizer demo + sparse-data individualization
- `make_figure_phenytoin.py` — renders the saturable-PK figure
