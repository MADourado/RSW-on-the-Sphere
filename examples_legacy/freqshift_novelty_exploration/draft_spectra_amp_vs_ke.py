"""Item 4 open question: KE (|A|^2) vs. amplitude-envelope (|A|) as the
FFT input signal. Compares both for the clearest shared-mode case,
RH(4,5) (full quartet vs. sub-triad0), quartet_rossby_gravity_influence.

Draft/exploratory -- not the final metric implementation.

Run:

    python examples/freqshift_novelty/draft_spectra_amp_vs_ke.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.plotting.style import apply_house_style
from run_dynamics import run_dynamics

WAVE_SET_KEY = "quartet_rossby_gravity_influence"
TARGET_LABEL = "RH(4,5)"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "freqshift_novelty",
                               "draft_spectra_amp_vs_ke.png")


def spectrum_of(signal, t_days):
    n = len(t_days)
    dt = t_days[1] - t_days[0]
    sig = signal - np.mean(signal)
    window = np.hanning(n)
    spec = np.fft.rfft(sig * window)
    freqs = np.fft.rfftfreq(n, d=dt)  # cycles/day
    power = np.abs(spec) ** 2
    periods = np.zeros_like(freqs)
    periods[1:] = 1.0 / freqs[1:]
    return periods, power


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    specs = load_wave_set_specs()
    spec = specs[WAVE_SET_KEY]
    config = RunConfig.from_wave_set(spec, plot=False)
    results = run_dynamics(config, write_table=False)
    full = results["full"]
    triad0_name = next(name for name in results if name != "full" and TARGET_LABEL in results[name]["labels"])
    triad0 = results[triad0_name]

    Y_full = np.load(full["trajectory_path"])["Y"]
    Y_sub = np.load(triad0["trajectory_path"])["Y"]
    j_full = full["labels"].index(TARGET_LABEL)
    j_sub = triad0["labels"].index(TARGET_LABEL)

    t_full, t_sub = full["t"], triad0["t"]
    E_full, E_sub = full["E"][:, j_full], triad0["E"][:, j_sub]
    A_full, A_sub = Y_full[:, j_full], Y_sub[:, j_sub]

    p_full_ke, pow_full_ke = spectrum_of(E_full, t_full)
    p_sub_ke, pow_sub_ke = spectrum_of(E_sub, t_sub)
    p_full_amp, pow_full_amp = spectrum_of(np.abs(A_full), t_full)
    p_sub_amp, pow_sub_amp = spectrum_of(np.abs(A_sub), t_sub)

    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(p_full_ke, pow_full_ke, color="black", label="full quartet")
    axes[0].plot(p_sub_ke, pow_sub_ke, color="tab:orange", ls="--", label=triad0_name)
    axes[0].set_title("KE = |A|^2 spectrum")
    axes[0].set_xlim(0, 3); axes[0].set_xlabel("Period (days)"); axes[0].legend(fontsize=8)

    axes[1].plot(p_full_amp, pow_full_amp, color="black", label="full quartet")
    axes[1].plot(p_sub_amp, pow_sub_amp, color="tab:orange", ls="--", label=triad0_name)
    axes[1].set_title("|A| (amplitude envelope) spectrum")
    axes[1].set_xlim(0, 3); axes[1].set_xlabel("Period (days)"); axes[1].legend(fontsize=8)

    fig.suptitle(f"{TARGET_LABEL}: KE spectrum vs. amplitude-envelope spectrum, full vs. {triad0_name}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    fig.savefig(args.path, dpi=150, bbox_inches="tight")
    print(f"wrote {os.path.abspath(args.path)}")


if __name__ == "__main__":
    main()
