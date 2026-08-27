"""Shared (efficiency map, energy integration) row builder for the two
single-triad composite figures in "Resonant triads" (``fig: rossby_only``,
``fig: combined``) -- paper_figure003/004 both need the same pattern (one
triad, one target mode held at rest, the other two swept for an
efficiency map alongside a fixed-velocity energy time series), so it is
factored here rather than duplicated.

Built on run_sweep.py's own unified 2D engine (``compute_2d_grid`` --
the same one ``run_sweep.py --wave-set KEY`` uses for its own 2D sweeps,
one ``run_dynamics()`` call per grid point) for the efficiency map, and
``rsw_sphere.plotting.energy_evolution.wave_set_energy_evolution`` for
the energy panel -- not on the retired ``triad_efficiency.py``/
``triad_dynamics.py`` toolchain, nor on the older, separate
``rsw_sphere.utilities.registry.sweep_2d`` engine this used before
(migrated 2026-08-27; ``plot_efficiency_map``'s own rendering is
unchanged, only where its input array comes from).
"""
import dataclasses

import numpy as np

from rsw_sphere.dynamics.run_config import RunConfig, SweepAxis, SweepConfig
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.functional_map import plot_efficiency_map
from rsw_sphere.plotting.energy_evolution import wave_set_energy_evolution
from rsw_sphere.utilities.efficiency import default_velocity_range
from run_sweep import compute_2d_grid


def triad_row(spec, target: int, ax_eff, ax_energy, n_grid: int = 15,
              tf_days: float = None, h: float = None,
              energy_velocities=None):
    """Fill one (efficiency map, energy integration) row for a 1-triad
    ``WaveSetSpec`` at target-mode index ``target`` (0/1/2 -> a/b/c, held
    at rest for both panels unless ``energy_velocities`` overrides the
    energy panel).
    """
    settings = spec.settings
    tf_days = tf_days if tf_days is not None else settings.get("tf_days", 10)
    h = h if h is not None else settings.get("h", 0.01)

    triads = [spec.triad_indices(0)]
    swept = [i for i in range(3) if i != target]
    idx1, idx2 = swept
    u1_range = default_velocity_range(spec.modes[idx1][2])
    u2_range = default_velocity_range(spec.modes[idx2][2])

    velocities = list(spec.velocities)
    velocities[target] = 0.0
    point_spec = dataclasses.replace(spec, velocities=tuple(velocities))
    sweep = SweepConfig(
        axes=(SweepAxis(mode=spec.mode_keys[idx1], min=u1_range[0], max=u1_range[1]),
              SweepAxis(mode=spec.mode_keys[idx2], min=u2_range[0], max=u2_range[1])),
        n_grid=n_grid)
    config = RunConfig.from_wave_set(point_spec, tf_days=tf_days, h=h, sweep=sweep)
    U1, U2, grid_results = compute_2d_grid(config, plot_per_point=False)

    target_label = _mode_label(*spec.modes[target])
    # raw 0-1 fraction, matching wave_set_efficiency's own convention --
    # plot_efficiency_map does its own *100 for display, so this must NOT
    # be pre-scaled (unlike run_sweep.py's own per-point compact dict,
    # which pre-scales for its own line/heatmap plots).
    efficiency = np.array([[grid_results[i, j]['per_mode_unit']['full'][target_label]['efficiency'] / 100
                             for j in range(n_grid)] for i in range(n_grid)])

    label1 = _mode_label(*spec.modes[idx1])
    label2 = _mode_label(*spec.modes[idx2])
    plot_efficiency_map(
        U1, U2, efficiency,
        xlabel=f"{label1} - zonal velocity (m/s)",
        ylabel=f"{label2} - zonal velocity (m/s)",
        title=f"{spec.display_label}: target {target_label} -- efficiency",
        ax=ax_eff)

    if energy_velocities is None:
        energy_velocities = list(spec.velocities)
        energy_velocities[target] = 0.0
    energy_result = wave_set_energy_evolution(
        spec.modes, triads, energy_velocities, h_e=spec.h_e,
        tf_days=tf_days, h=h, highlight=target, ax=ax_energy)
    ax_energy.set_title(f"{spec.display_label}: {spec.label} -- energy integration")

    return {'U1': U1, 'U2': U2, 'Efficiency': efficiency}, energy_result
