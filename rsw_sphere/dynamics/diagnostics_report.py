"""Pure computation behind run_dynamics.py's console tables/CSVs --
factored out of its own main() so a sweep can call the exact same
diagnostics engine per grid point without going through the CLI.

compute_diagnostics_report() always computes everything (cheap -- FFT and
a handful of numpy calls per mode/unit, no integration); callers decide
what to print/write. write_diagnostics_files() writes the CSV/figure
side effects (same filenames run_dynamics.py --diagnostics always wrote).
"""
import os

import numpy as np

from rsw_sphere.utilities.periods import dominant_periods, low_frequency_power, DEFAULT_EXCLUSION_FRAC
from rsw_sphere.utilities.efficiency import wave_set_efficiency
from rsw_sphere.utilities.pmeasure import (
    pairwise_target_diagnostics, efficiency_variation_combined_for_all_targets, _default_triad_index_for_mode)
from rsw_sphere.utilities.novelty_frequency import novelty_combined_for_all_targets
from rsw_sphere.utilities.tables import write_csv
from rsw_sphere.physics import total_energy_joules
from rsw_sphere.dynamics.trajectory_cache import _mode_slug
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.novelty_frequency_panel import novelty_frequency_figures


def compute_diagnostics_report(results: dict, spec, novelty_exclusion_frac: float = DEFAULT_EXCLUSION_FRAC,
                                novelty_min_prominence: float = 0.02,
                                efficiency_drift_max: float = 0.1,
                                low_freq_threshold_frac: float = 1.0) -> dict:
    """Every diagnostic run_dynamics.py --diagnostics computes for one
    already-integrated ``results`` dict (run_dynamics.run_dynamics()'s own
    return shape).

    low_freq_threshold_frac : fraction beyond each (mode, unit)'s own
        slowest detected peak period that counts as "low frequency"
        (default 1.0 = 100%, i.e. the cutoff is twice that peak's own
        period -- a factor-of-2 scale separation, tunable if a tighter or
        looser separation is wanted).

    Returns
    -------
    dict
        per_mode_unit : {unit_name: {mode_label: {dEK, efficiency,
            linear_period_days, linear_freq_cpd, top_peaks, period_global,
            freq_global_cpd, low_freq_power, insufficient_cycles}}} --
            always computed, one entry per (mode, unit) pair, mirrors
            run_dynamics.py's always-on frequency table. `low_freq_power`
            is relative to that SAME (mode, unit)'s own slowest detected
            peak (`max(p['period_days'] for p in top_peaks)`, not
            necessarily the dominant one), not a fixed absolute-day
            cutoff -- "low frequency" means a period at least
            `low_freq_threshold_frac` beyond whatever periods this
            trajectory's own native oscillations already contain, so the
            same relative definition applies whether the mode's own
            period is 0.15d or 5d. Previously a fixed 10-day cutoff,
            which was arbitrary relative to a fast mode (a ~66x
            separation) and barely separated at all for a slow one
            (found 2026-08-28).
        unit_energy : {unit_name: {mean_total_energy_nondim,
            mean_total_energy_joules}} -- one entry per unit (``full`` and
            every sub-triad), ``E_total``'s own time-mean converted to
            Joules via ``rsw_sphere.physics.total_energy_joules``. Lets a
            caller check whether a wave set's own total-energy budget
            actually changes across candidates/sweep points -- useful
            context alongside ``efficiency_var`` (e.g. a driving velocity
            that also changes how much energy the wave set carries), even
            though ``efficiency_var`` itself no longer normalizes by it
            (see ``rsw_sphere.utilities.pmeasure``'s own module
            docstring): the per-unit ``efficiency`` field below still
            does, since it's a standalone, single-configuration quantity
            (paper eq. `effgen`), not a cross-configuration comparison.
        precession : {triad_label: {freq_full_cpd, freq_alone_cpd (or
            None), phase_variation_pct (nan if no alone comparison)}} --
            empty if spec has no sub-triads.
        pairwise : list of {mode, vs, efficiency_var_pct, spectral_dev_pct,
            novelty_period_days, novelty_relevance_pct} -- one row per
            (target mode, containing sub-triad) pair. ``efficiency_var_pct``
            is the raw energy-variation percent change, full wave set vs.
            this one named sub-triad (``rsw_sphere.utilities.pmeasure.
            pairwise_target_diagnostics``'s own ``efficiency_var`` value).
        final : list of {mode, efficiency_var_final_pct, spectral_dev_final_pct,
            vs, novelty_period_final_days, novelty_freq_final_cpd,
            novelty_relevance_final_pct} -- one row per mode in the full
            wave set, combining every containing sub-triad at once (``vs``
            is whichever containing sub-triad gives the target its own
            largest raw energy swing, i.e. ``rsw_sphere.utilities.pmeasure.
            efficiency_variation_final``'s own reference choice). Empty if
            spec has no sub-triads (undefined without a "with vs. without"
            comparison).
    """
    per_mode_unit = {}
    eff_cache = {}
    unit_energy = {}
    t_f_days_by_unit = {}
    for name, r in results.items():
        per_mode_unit[name] = {}
        mean_e_nondim = float(np.mean(r['E_total']))
        unit_energy[name] = {
            'mean_total_energy_nondim': mean_e_nondim,
            'mean_total_energy_joules': total_energy_joules(mean_e_nondim, spec.h_e),
        }
        t_f_days_by_unit[name] = float(r['t'][-1] - r['t'][0])
        for j, (lbl, w, dEK) in enumerate(zip(r['labels'], r['omega'], r['dEK'])):
            pr = dominant_periods(r['t'], r['E'][:, j])
            eff = wave_set_efficiency(r['E'][:, j], r['E_total'], r['drift'], drift_max=efficiency_drift_max)
            eff_cache[(name, lbl)] = eff
            lin_period = 1.0 / (2 * abs(w))
            lin_freq = 2 * abs(w)
            top_peaks = pr['top_peaks']
            insufficient_cycles = any(p['period_days'] * 4 > t_f_days_by_unit[name] for p in top_peaks[:2])
            # "Low frequency" is relative to this (mode, unit)'s own
            # slowest detected peak (not necessarily the dominant, highest
            # -power one), not a fixed absolute-day cutoff -- a period
            # already present in the mode's own native dynamics shouldn't
            # count as a newly-generated low-frequency feature just
            # because it happens to be slower than some fixed number of
            # days.
            if top_peaks:
                own_slowest_period = max(p['period_days'] for p in top_peaks)
                low_freq_cutoff_days = own_slowest_period * (1.0 + low_freq_threshold_frac)
                low_freq_power = low_frequency_power(
                    r['t'], r['E'][:, j] / r['E_total'], period_cutoff_days=low_freq_cutoff_days)
            else:
                low_freq_power = 0.0
            per_mode_unit[name][lbl] = {
                'dEK': float(dEK), 'efficiency': eff,
                'linear_period_days': lin_period, 'linear_freq_cpd': lin_freq,
                'top_peaks': top_peaks,
                'period_global': top_peaks[0]['period_days'] if top_peaks else float('nan'),
                'freq_global_cpd': 1.0 / top_peaks[0]['period_days'] if top_peaks else float('nan'),
                'low_freq_power': low_freq_power,
                'insufficient_cycles': insufficient_cycles,
            }

    precession = {}
    pairwise = []
    final = []

    if spec.has_subtriads():
        full = results['full']

        for triad_label, prec_full in full['precession_freq'].items():
            prec_alone = None
            for name, r in results.items():
                if name != 'full' and triad_label in r['precession_freq']:
                    prec_alone = r['precession_freq'][triad_label]
                    break
            if prec_alone is not None and abs(prec_alone) > 1e-12:
                phase_variation = 100 * (abs(prec_full) - abs(prec_alone)) / abs(prec_alone)
            else:
                phase_variation = float('nan')
            precession[triad_label] = {
                'freq_full_cpd': prec_full / (2 * np.pi),
                'freq_alone_cpd': prec_alone / (2 * np.pi) if prec_alone is not None else None,
                'phase_variation_pct': phase_variation,
            }

        for j, label in enumerate(full['labels']):
            amp_full = np.sqrt(full['E'][:, j])
            for name, r in results.items():
                if name == 'full' or label not in r['labels']:
                    continue
                j_sub = r['labels'].index(label)
                amp_sub = np.sqrt(r['E'][:, j_sub])
                d = pairwise_target_diagnostics(
                    full['t'], amp_full, amp_sub, full['E_total'], r['E_total'],
                    novelty_exclusion_frac=novelty_exclusion_frac,
                    novelty_min_prominence=novelty_min_prominence)
                pairwise.append({
                    'mode': label, 'vs': name, 'efficiency_var_pct': d['efficiency_var'],
                    'spectral_dev_pct': d['spectral_deviation'],
                    'novelty_period_days': d['novelty_period'], 'novelty_relevance_pct': d['novelty_relevance'],
                })

        pfinal = efficiency_variation_combined_for_all_targets(results)
        novelty_final = novelty_combined_for_all_targets(
            results, min_prominence=novelty_min_prominence, exclusion_frac=novelty_exclusion_frac)
        for label in full['labels']:
            pf = pfinal[label]
            ref = pf['reference']

            peaks = novelty_final[label]['novel_peaks']
            novelty_period = peaks[0]['period_days'] if peaks else float('nan')
            final.append({
                'mode': label, 'efficiency_var_final_pct': pf['efficiency_var'],
                'spectral_dev_final_pct': pf['spectral_deviation'],
                'vs': ref, 'novelty_period_final_days': novelty_period,
                'novelty_freq_final_cpd': 1.0 / novelty_period if peaks else float('nan'),
                'novelty_relevance_final_pct': peaks[0]['relevance_pct'] if peaks else float('nan'),
            })

    return {
        'per_mode_unit': per_mode_unit, 'eff_cache': eff_cache, 'unit_energy': unit_energy,
        'precession': precession, 'pairwise': pairwise, 'final': final,
    }


def pairwise_value_for_target(report: dict, spec, target_idx: int, reference_triad: int, field: str):
    """One row of report['pairwise'] -- the same (target, sub-triad) pair the
    now-deleted ``pmeasure.wave_set_diagnostics_sweep`` engine's own
    single-fixed-reference-triad diagnostics used to compare against
    (``pmeasure._default_triad_index_for_mode``'s own selection:
    ``reference_triad`` if the target is one of its members, else the first
    triad containing it). Returns NaN if the target belongs to no sub-triad
    at all (e.g. a plain triad).

    Lets a caller (``run_sweep_sets.py``) reproduce that old engine's exact
    pairwise ``efficiency_var``/``novelty_period`` values from this module's
    own report, instead of a separate integration pass -- same formula either
    way (``pairwise_target_diagnostics``), just read from a different,
    already-computed place.

    target_idx : positional index into spec.modes (as the old engine's own
        `target_indices` took), not a registry mode key or label.
    field : 'efficiency_var_pct', 'spectral_dev_pct', 'novelty_period_days',
        or 'novelty_relevance_pct' (report['pairwise']'s own row keys).
    """
    triads = [spec.triad_indices(i) for i in range(spec.n_triads())]
    t_idx = _default_triad_index_for_mode(triads, reference_triad, target_idx)
    if t_idx is None:
        return float('nan')
    member_p, member_q, _ = spec.sub_triad_modes(t_idx)
    unit_name = f"triad_{_mode_slug(*member_p)}_{_mode_slug(*member_q)}"
    target_label = _mode_label(*spec.modes[target_idx])
    for row in report['pairwise']:
        if row['mode'] == target_label and row['vs'] == unit_name:
            return row[field]
    return float('nan')


def write_diagnostics_files(results: dict, report: dict, run_dir: str, run_label: str, spec,
                             diagnostics: bool = True, novelty_exclusion_frac: float = DEFAULT_EXCLUSION_FRAC,
                             novelty_min_prominence: float = 0.02, novelty_xmax: float = None) -> dict:
    """Write diag_evol_*.csv (always) and, if ``diagnostics`` and spec has
    sub-triads, diag_prec_freq_*.csv/diag_pairwise_*.csv/diag_final_*.csv +
    diag_freq_novel_*.png -- same filenames run_dynamics.py --diagnostics
    already wrote, factored out for reuse by a sweep's own per-point
    diagnostics bundle (which always wants the full bundle when writing
    anything at all -- diagnostics=True there).

    novelty_xmax : upper period (days) shown/searched in the novelty
        spectrum figures. Default (None): ``round(sqrt(tf_days / 2))`` --
        nonlinear so it doesn't balloon for long runs; hits tf_days=20 ->
        3d (the good short-run default) and tf_days=200 -> 10d exactly.
        A long tf_days run's own novelty search window (and plot) scales
        with it automatically instead of staying pinned at the short-run
        default of 3 days, which used to cut off any genuinely long-period
        novel content a long run was specifically integrated to look for
        (found 2026-08-28, a tf_days=200 quartet_rossby_kelvin run whose
        spectrum plot had
        nothing to show past 3 days).

    Returns dict of paths written (keys: diag_evol, diag_prec_freq,
    diag_pairwise, diag_final, novelty -- the last a list; a key is
    omitted if nothing was written for it).
    """
    os.makedirs(run_dir, exist_ok=True)
    paths = {}

    diag_evol_rows = []
    for unit_name, per_mode in report['per_mode_unit'].items():
        for label, m in per_mode.items():
            row = {
                'wave_set': spec.key, 'unit': unit_name, 'mode': label,
                'dEK': m['dEK'], 'efficiency': m['efficiency'],
                'linear_period_days': m['linear_period_days'], 'linear_freq_cpd': m['linear_freq_cpd'],
                'low_freq_power': m['low_freq_power'],
            }
            for k in range(3):
                if k < len(m['top_peaks']):
                    p = m['top_peaks'][k]
                    row[f'peak{k + 1}_period_days'] = p['period_days']
                    row[f'peak{k + 1}_freq_cpd'] = 1.0 / p['period_days']
                    row[f'peak{k + 1}_power_pct'] = p['power_frac']
                else:
                    row[f'peak{k + 1}_period_days'] = ''
                    row[f'peak{k + 1}_freq_cpd'] = ''
                    row[f'peak{k + 1}_power_pct'] = ''
            diag_evol_rows.append(row)
    paths['diag_evol'] = os.path.join(run_dir, f'diag_evol_{run_label}.csv')
    write_csv(diag_evol_rows, paths['diag_evol'])

    if not diagnostics or not spec.has_subtriads():
        return paths

    diag_prec_freq_rows = [
        {'triad': triad_label, 'precession_freq_full_cpd': p['freq_full_cpd'],
         'precession_freq_alone_cpd': p['freq_alone_cpd'] if p['freq_alone_cpd'] is not None else '',
         'phase_variation_pct': p['phase_variation_pct']}
        for triad_label, p in report['precession'].items()
    ]
    paths['diag_prec_freq'] = os.path.join(run_dir, f'diag_prec_freq_{run_label}.csv')
    write_csv(diag_prec_freq_rows, paths['diag_prec_freq'])

    diag_pairwise_rows = [
        {'mode': d['mode'], 'vs': d['vs'],
         'efficiency_var_pct': d['efficiency_var_pct'], 'spectral_dev_pct': d['spectral_dev_pct'],
         'novelty_period_days': d['novelty_period_days'] if np.isfinite(d['novelty_period_days']) else '',
         'novelty_relevance_pct': d['novelty_relevance_pct'] if np.isfinite(d['novelty_period_days']) else ''}
        for d in report['pairwise']
    ]
    paths['diag_pairwise'] = os.path.join(run_dir, f'diag_pairwise_{run_label}.csv')
    write_csv(diag_pairwise_rows, paths['diag_pairwise'])

    diag_final_rows = [
        {'mode': d['mode'],
         'efficiency_var_final_pct': d['efficiency_var_final_pct'],
         'spectral_dev_final_pct': d['spectral_dev_final_pct'], 'vs': d['vs'] or 'none',
         'novelty_period_final_days': d['novelty_period_final_days'] if np.isfinite(d['novelty_period_final_days']) else '',
         'novelty_freq_final_cpd': d['novelty_freq_final_cpd'] if np.isfinite(d['novelty_freq_final_cpd']) else '',
         'novelty_relevance_final_pct': d['novelty_relevance_final_pct'] if np.isfinite(d['novelty_relevance_final_pct']) else ''}
        for d in report['final']
    ]
    paths['diag_final'] = os.path.join(run_dir, f'diag_final_{run_label}.csv')
    write_csv(diag_final_rows, paths['diag_final'])

    paths['novelty'] = novelty_frequency_figures(
        results, run_dir, xmax=novelty_xmax, filename_suffix=run_label, min_prominence=novelty_min_prominence,
        exclusion_frac=novelty_exclusion_frac)

    return paths
