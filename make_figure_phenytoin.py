# Copyright 2026. SPDX-License-Identifier: Apache-2.0
"""Generate the phenytoin generality-demo figure from REAL served-engine output.

Two panels, theme-aware (light + dark), matching the existing figures/ style:
  A. The saturable-kinetics TRAP: steady-state trough vs daily dose (the hyperbola),
     therapeutic band shaded, the naive linear-extrapolation overshoot vs the
     gradient-optimized dose, and the Vmax asymptote.
  B. The SAME engine individualizing three metabolizer phenotypes onto the 10 to 20
     mg/L band (real optimized steady-state curves over 48 h).

No fabricated numbers: every point comes from the served phenytoin Tesseract.
"""
import os

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine import connect
from engine_phenytoin import optimize_regimen, pheny_params, pheny_exposure
from servers_phenytoin import serve_all

jax.config.update("jax_platform_name", "cpu")
f32 = lambda x: jnp.float32(x)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
os.makedirs(OUT, exist_ok=True)

CAT = dict(blue_l="#2a78d6", blue_d="#3987e5", orange_l="#eb6834", orange_d="#d95926",
           aqua_l="#1baf7a", aqua_d="#199e70", violet_l="#4a3aa7", violet_d="#9085e9")
STATUS = dict(good="#0ca30c", warning="#fab219", serious="#ec835a", critical="#d03b3b")

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", band="#0ca30c",
                  blue=CAT["blue_l"], orange=CAT["orange_l"], aqua=CAT["aqua_l"],
                  violet=CAT["violet_l"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", band="#3fb63f",
                 blue=CAT["blue_d"], orange=CAT["orange_d"], aqua=CAT["aqua_d"],
                 violet=CAT["violet_d"]),
}


def style(th):
    t = THEMES[th]
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["ink"],
        "axes.labelcolor": t["ink"], "axes.edgecolor": t["axis"],
        "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.titlecolor": t["ink"], "grid.color": t["grid"],
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.grid": True, "grid.linewidth": 1.0, "axes.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return t


def ss_curve_over(T, dose, tau, vmax, km, v, hours=48.0):
    """Tile the real steady-state one-interval curve across `hours`."""
    ex = pheny_exposure(T, f32(dose), f32(tau), f32(vmax), f32(km), f32(v))
    grid = np.array(ex["times"]); conc = np.array(ex["conc"])
    n = int(np.ceil(hours / tau))
    ts, cs = [], []
    for k in range(n):
        ts.append(grid + k * tau); cs.append(conc)
    t = np.concatenate(ts); c = np.concatenate(cs)
    m = t <= hours + 1e-6
    return t[m], c[m], float(ex["trough"])


def collect(T):
    """Run the real engine and gather every number the figure plots."""
    cov = dict(age=40, weight=70, height_in=68, scr=1.0, sex=0, vmax_scale=1.0)
    p = pheny_params(T, cov)
    vmax, km, v = p["vmax"], p["km"], p["v"]
    tau = 12.0

    # dose-response curve: trough vs DAILY dose (real model), sweeping per-admin dose
    daily, troughs = [], []
    for dose in np.linspace(60.0, 235.0, 40):  # per q12 admin
        ex = pheny_exposure(T, f32(float(dose)), f32(tau), f32(vmax), f32(km), f32(v))
        daily.append(float(dose) * 24.0 / tau)
        troughs.append(float(ex["trough"]))

    anchor = 180.0
    ex_a = pheny_exposure(T, f32(anchor), f32(tau), f32(vmax), f32(km), f32(v))
    naive = anchor * 1.2
    ex_n = pheny_exposure(T, f32(naive), f32(tau), f32(vmax), f32(km), f32(v))

    patients = {
        "adult-normal":     cov,
        "fast-metabolizer": dict(age=30, weight=75, height_in=70, scr=0.9, sex=0, vmax_scale=1.3),
        "slow-metabolizer": dict(age=75, weight=62, height_in=64, scr=1.2, sex=1, vmax_scale=0.65),
    }
    curves = {}
    b = None
    for name, c in patients.items():
        rr = optimize_regimen(T, c, target=15.0)
        bb = rr["best"]
        if name == "adult-normal":
            b = bb  # reuse for panel A (avoids a second full optimizer run)
        t, cc, tr = ss_curve_over(T, bb["dose"], bb["tau"], rr["vmax"], rr["km"], rr["v"])
        curves[name] = dict(t=t, c=cc, dose=bb["dose"], tau=bb["tau"],
                            daily=bb["daily"], trough=tr, vmax_mgday=rr["vmax_mgday"])

    return dict(vmax_mgday=p["vmax_mgday"], daily=np.array(daily), troughs=np.array(troughs),
                anchor_daily=anchor * 2, anchor_tr=float(ex_a["trough"]),
                naive_daily=naive * 2, naive_tr=float(ex_n["trough"]),
                eng_daily=b["daily"], eng_dose=b["dose"], eng_tau=b["tau"],
                eng_tr=b["trough"], curves=curves)


def render(D, th):
    t = style(th)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    # -------- Panel A: the saturable trap --------
    axA.axhspan(10, 20, color=t["band"], alpha=0.12, lw=0, zorder=0)
    axA.axhline(20, color=t["band"], lw=1.0, ls=":", alpha=0.6)
    axA.axhline(10, color=t["band"], lw=1.0, ls=":", alpha=0.6)
    axA.axvline(D["vmax_mgday"], color=t["muted"], lw=1.2, ls="--", alpha=0.8)
    axA.text(D["vmax_mgday"] - 6, 46, "Vmax asymptote\n(clearance saturates)",
             ha="right", va="top", fontsize=8.5, color=t["ink2"])

    axA.plot(D["daily"], D["troughs"], color=t["blue"], lw=2.6, zorder=3,
             label="true saturable response")

    # naive linear extrapolation from the anchor
    slope = (D["anchor_tr"]) / D["anchor_daily"]  # crude linear intuition through origin
    lin_x = np.array([D["anchor_daily"], D["naive_daily"]])
    lin_y = slope * lin_x
    axA.plot(lin_x, lin_y, color=t["muted"], lw=1.8, ls="--", zorder=2,
             label="what linear intuition expects")

    axA.scatter([D["anchor_daily"]], [D["anchor_tr"]], s=70, color=t["aqua"],
                zorder=5, edgecolor=t["surface"], linewidth=1.2)
    axA.annotate(f"anchor {D['anchor_daily']:.0f} mg/day\ntrough {D['anchor_tr']:.0f} mg/L",
                 (D["anchor_daily"], D["anchor_tr"]), textcoords="offset points",
                 xytext=(-6, -34), fontsize=8.5, color=t["ink2"], ha="right")

    axA.scatter([D["naive_daily"]], [D["naive_tr"]], s=90, color=STATUS["critical"],
                zorder=6, edgecolor=t["surface"], linewidth=1.2, marker="X")
    axA.annotate(f"+20% dose -> {D['naive_tr']:.0f} mg/L\nTOXIC (not +20%)",
                 (D["naive_daily"], D["naive_tr"]), textcoords="offset points",
                 xytext=(8, -6), fontsize=8.5, color=STATUS["critical"], ha="left",
                 fontweight="bold")

    axA.scatter([D["eng_daily"]], [D["eng_tr"]], s=90, color=t["orange"],
                zorder=6, edgecolor=t["surface"], linewidth=1.2)
    axA.annotate(f"engine: {D['eng_dose']:.0f} mg q{D['eng_tau']:.0f}h\ntrough {D['eng_tr']:.0f} mg/L",
                 (D["eng_daily"], D["eng_tr"]), textcoords="offset points",
                 xytext=(8, 14), fontsize=8.5, color=t["orange"], ha="left", fontweight="bold")

    axA.set_ylim(0, 48)
    axA.set_xlim(D["daily"].min(), D["daily"].max())
    axA.set_xlabel("phenytoin daily dose (mg/day)")
    axA.set_ylabel("steady state trough (mg/L, total)")
    axA.set_title("A. The saturable trap: a small dose bump runs away",
                  fontsize=11, loc="left", pad=10)
    axA.legend(loc="upper left", fontsize=8.5, frameon=False)

    # -------- Panel B: three metabolizers, same engine --------
    axB.axhspan(10, 20, color=t["band"], alpha=0.12, lw=0, zorder=0)
    axB.axhline(20, color=t["band"], lw=1.0, ls=":", alpha=0.6)
    axB.axhline(10, color=t["band"], lw=1.0, ls=":", alpha=0.6)
    axB.text(0.4, 20.6, "therapeutic band 10 to 20 mg/L", fontsize=8.5,
             color=t["band"], va="bottom")

    colors = {"adult-normal": t["blue"], "fast-metabolizer": t["violet"],
              "slow-metabolizer": t["orange"]}
    disp = {"adult-normal": "normal metabolizer", "fast-metabolizer": "fast metabolizer",
            "slow-metabolizer": "slow metabolizer"}
    for name, cv in D["curves"].items():
        lab = (f"{disp[name]}: {cv['dose']:.0f} mg q{cv['tau']:.0f}h "
               f"({cv['daily']:.0f} mg/day, Vmax {cv['vmax_mgday']:.0f})")
        axB.plot(cv["t"], cv["c"], color=colors[name], lw=2.4, label=lab)

    axB.set_ylim(0, 26)
    axB.set_xlim(0, 48)
    axB.set_xlabel("time (h)")
    axB.set_ylabel("phenytoin level (mg/L, total)")
    axB.set_title("B. Same engine, three metabolizer phenotypes on target",
                  fontsize=11, loc="left", pad=10)
    axB.legend(loc="lower right", fontsize=8.0, frameon=False)

    fig.suptitle("Phenytoin (saturable Michaelis and Menten kinetics) through the same dosing engine",
                 fontsize=12.5, x=0.012, ha="left", y=0.99, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT, f"fig_phenytoin_saturable_{th}.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main():
    with serve_all() as urls:
        T = connect(urls)
        D = collect(T)
        for th in ("light", "dark"):
            print("wrote", render(D, th))


if __name__ == "__main__":
    main()
