"""§3.3.1 headline numbers for Quartet C (EG(1,1)), using the same
methodology outlined below (period shift + peak-KE difference, both
tf-independent; $\\mathcal{F}_2$ at the registered tf_days; Hilbert phase
lag).

NOTE (2026-09-03): the Quartet D (EG(7,9), i.e. `quartet_gravity_79`)
headline this script used to also compute is no longer cited anywhere in
JFM-template.tex -- superseded by Quartet E once that topology was added
(git: "new quartet E"). `quartet_gravity_79`'s own `display_label` was
updated to flag it as retired; this script no longer calls `headline()`
on it. The Quartet C (EG(1,1)) numbers this script computes were
similarly retired along with the old `fig: cap4ex1` paragraph (30 m/s IC)
once that figure/section was rebuilt around a symmetric 40 m/s IC and a
novelty-frequency framing instead
(`paper_figure008_quartet_rossby_kelvin_panel.py`) -- kept here rather
than deleted since neither computation is currently cited, not because
either is still used.

Target mode is b=RH(3,4) for both quartets (index 1 in the registry's
own a,b,c,d ordering -- shared edge RH(4,5)+RH(3,4) for both Quartet C
and D, confirmed in `wave_sets_default.yaml`).

Run:

    python examples/tables/paper_headlines_sec3.3.py
"""
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from scipy.signal import find_peaks

from rsw_sphere.physics import (gamma_from_he, days_from_nondim_time,
                                 air_density_from_equivalent_depth, A, G)
from rsw_sphere.dynamics.integrators import RK44 as RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs


def headline(spec, label, tf_days_check=(20, 40)):
    gamma = gamma_from_he(spec.h_e, g=G)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_full = WaveSet(gamma, list(spec.modes), [spec.triad_indices(i) for i in range(spec.n_triads())],
                           N=10, deg=300)
        ws_triad1 = WaveSet(gamma, list(spec.modes[:3]), [spec.triad_indices(0)], N=10, deg=300)

    i_b = 1
    A0_full = ws_full.amplitudes_from_velocities(list(spec.velocities), spec.h_e, g=G)
    A0_triad = A0_full[:3]
    h = spec.settings['h']

    t_f_long = 400.0 * 4 * np.pi
    Yf, Tf = RK33(ws_full, 0, t_f_long, h, A0_full)
    Yt, _ = RK33(ws_triad1, 0, t_f_long, h, A0_triad)
    days = days_from_nondim_time(Tf)
    KEf = np.abs(Yf[:, i_b]) ** 2
    KEt = np.abs(Yt[:, i_b]) ** 2
    pf, _ = find_peaks(KEf)
    pt, _ = find_peaks(KEt)
    T_full = np.mean(np.diff(days[pf]))
    T_triad = np.mean(np.diff(days[pt]))
    period_shift_pct = 100 * (T_full - T_triad) / T_triad

    rho = air_density_from_equivalent_depth(spec.h_e, g=G)
    prefactor = G * spec.h_e ** 2 * A ** 2 * np.pi * rho
    peak_diff_pct = 100 * (KEf.max() - KEt.max()) / KEt.max()
    EK_full_peak = prefactor * KEf.max()
    EK_triad_peak = prefactor * KEt.max()

    print(f"\n=== {label} ({spec.key if hasattr(spec, 'key') else ''}) ===")
    print(f"Period shift (tf-independent, 400d window) = {period_shift_pct:+.2f}% "
          f"(T_full={T_full:.4f}d, T_triad={T_triad:.4f}d)")
    print(f"Peak-KE difference (tf-independent) = {peak_diff_pct:+.2f}% "
          f"(full={EK_full_peak:.3e} J, triad={EK_triad_peak:.3e} J)")

    for tf_days in tf_days_check:
        t_f = tf_days * 4 * np.pi
        Yf_c, _ = RK33(ws_full, 0, t_f, h, A0_full)
        Yt_c, _ = RK33(ws_triad1, 0, t_f, h, A0_triad)
        af, at = np.abs(Yf_c[:, i_b]), np.abs(Yt_c[:, i_b])
        F2 = np.sqrt(np.mean((af - at) ** 2)) / np.sqrt(np.mean(at ** 2))
        print(f"  F2 (tf={tf_days}d) = {F2 * 100:.2f}%")

    return dict(period_shift_pct=period_shift_pct, peak_diff_pct=peak_diff_pct,
                EK_full_peak=EK_full_peak, EK_triad_peak=EK_triad_peak)


if __name__ == "__main__":
    specs = load_wave_set_specs()
    r_c = headline(specs['quartet_rossby_kelvin'], "Quartet C (EG(1,1))")

    print(f"\nQuartet C: period {r_c['period_shift_pct']:+.2f}%, peak-KE {r_c['peak_diff_pct']:+.2f}%")
