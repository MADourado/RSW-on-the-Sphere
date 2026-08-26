"""Shared (efficiency map, energy integration) row builder for the two
single-triad composite figures in "Resonant triads" (``fig: rossby_only``,
``fig: combined``) -- paper_figure003/004 both need the same pattern (one
triad, one target mode held at rest, the other two swept for an
efficiency map alongside a fixed-velocity energy time series), so it is
factored here rather than duplicated.

Built entirely on ``rsw_sphere.utilities.registry.sweep_2d`` (the same
engine ``run_sweep.py`` calls for its own 2D sweeps) and
``rsw_sphere.plotting.energy_evolution.wave_set_energy_evolution`` --
not on the retired ``triad_efficiency.py``/``triad_dynamics.py`` toolchain.
"""
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.functional_map import plot_efficiency_map
from rsw_sphere.plotting.energy_evolution import wave_set_energy_evolution
from rsw_sphere.utilities.efficiency import default_velocity_range
from rsw_sphere.utilities.registry import sweep_2d


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

    result = sweep_2d(
        spec.modes, triads, spec.h_e, (idx1, idx2), {target: 0.0}, [target],
        diagnostics=("efficiency",), u1_range=u1_range, u2_range=u2_range,
        n_grid=n_grid, tf_days=tf_days, h=h, verbose=True, progress_label=spec.key)

    label1 = _mode_label(*spec.modes[idx1])
    label2 = _mode_label(*spec.modes[idx2])
    target_label = _mode_label(*spec.modes[target])
    plot_efficiency_map(
        result["U1"], result["U2"], result["Efficiency"][..., 0],
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

    return result, energy_result
