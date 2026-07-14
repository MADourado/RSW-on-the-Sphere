# Hough mode visualization

Two scripts in `utils/` visualize a single Hough mode `(m, n, α)` — the
normal modes (eigenvectors) diagonalized in `Hough_Harmonics/` and used as
building blocks for the triad/quartet/quintet dynamics elsewhere in the
repository. Both take the same mode indices; they differ in what they plot.

| Script | Shows | Domain |
|--------|-------|--------|
| `utils/Hough_and_derivatives.py` | `u, v, h` and their latitudinal derivatives `du/dφ, dv/dφ, dh/dφ` | latitude only, `φ ∈ [−90°, 90°]` |
| `utils/Hough_spatial_ev.py` | `h` as a filled contour, `(u, v)` as a quiver | full sphere, `(λ, φ)` |

For the eigenvalue problem and the meaning of `(m, n, α)` and the
equivalent height `h_e`, see [`README_dispersion.md`](README_dispersion.md)
— both scripts reuse the same `γ`/`ε` non-dimensionalisation and the same
`Hough_Harmonics.Eigenvalues_and_eigenvectors.Eigenvectors.Hough_harmonic`
solver.

---

## 1. `Hough_and_derivatives.py` — latitudinal profile

Plots the (normalized) latitudinal Hough functions `u(φ), v(φ), h(φ)` and
their derivatives, for a single mode at a given equivalent height. This is
the quickest sanity check of a mode's meridional structure — used to
generate the `mode_*/Hough_harmonic.png` and `mode_*/derivatives.png`
figures under `Testes_Marco/figures/`.

```python
from utils.Hough_and_derivatives import hough_and_derivatives

hough_and_derivatives(m=1, n=2, alpha=3, h_e=10000, path='Testes_Marco/figures/mode_a')
```

`alpha`: `1` = EIG, `2` = WIG, `3` = Rossby-Haurwitz (thesis EG/WG/RH — see
[`README_dispersion.md`](README_dispersion.md) §2.4). The function has no
CLI; call it directly or edit the `if __name__ == "__main__":` example at
the bottom of the file.

---

## 2. `Hough_spatial_ev.py` — full spatial pattern

Reconstructs and plots the **full 2D spatial structure** of a Hough mode
over the sphere: `h(λ, φ)` as a filled contour, `(u(λ,φ), v(λ,φ))` as a
quiver, on a `PlateCarree` (lon-lat) map with gridlines and axis labels.

### 2.1 Usage

```bash
# save to a file (positional path argument)
python utils/Hough_spatial_ev.py output.png --m 3 --n 7 --alpha 3

# choose a different equivalent height (metres)
python utils/Hough_spatial_ev.py output.png --m 3 --n 7 --alpha 3 --he 5000

# recenter the map on a different longitude
python utils/Hough_spatial_ev.py output.png --m 3 --n 7 --central-longitude 60

# run as a module from the repository root
python -m utils.Hough_spatial_ev output.png --m 1 --n 2 --alpha 3

# show interactively instead of saving (omit the path)
python utils/Hough_spatial_ev.py --m 1 --n 2 --alpha 3
```

`python utils/Hough_spatial_ev.py --help` prints the full argument list.
`-m` here is Python's own module flag (`python -m utils.Hough_spatial_ev`)
— unrelated to the script's `--m` (zonal wavenumber); the two are parsed by
different programs (the `python` interpreter vs. this script's `argparse`).

From another script:

```python
from utils.Hough_spatial_ev import hough_spatial_ev

hough_spatial_ev(m=3, n=7, alpha=3, h_e=10000, path="output.png")   # save
hough_spatial_ev(m=1, n=2, alpha=3)                                  # show interactively
```

### 2.2 Arguments

| Argument | Type | Default | Meaning |
|----------|------|---------|---------|
| `path` | str | `None` | Output image path. If omitted / `None`, the figure is shown with `plt.show()` instead of being saved. |
| `--m` | int | `1` | Zonal wavenumber. |
| `--n` | int | `2` | Meridional mode index. |
| `--alpha` | int | `3` | Mode type: `1`=EIG, `2`=WIG, `3`=Rossby-Haurwitz. |
| `--he` (`h_e`) | float | `10000` | Equivalent height (equivalent depth) in metres. |
| `--central-longitude` | float | `0` | Longitude at the centre of the map, degrees. |
| `N` | int | `10` | Spectral truncation of the Hough-mode expansion (function-only, not exposed on the CLI). |
| `nlat`, `nlon` | int | `91`, `181` | Latitude/longitude grid resolution (function-only). |
| `quiver_step` | int | `6` | Subsampling stride for the `(u, v)` arrows (function-only). |

### 2.3 How the spatial field is built

The eigensolver (`Hough_harmonic`) only returns the **latitudinal** part
`U(φ), V(φ), Z(φ)` of a mode. The full field follows from the standard
separable-mode ansatz used throughout the thesis (§2.2, `H = Θ(φ)
exp[i(mλ − ωt)]`): a static snapshot at zero phase,

```
h(λ, φ) = Re[ Z(φ) · exp(i m λ) ]
u(λ, φ) = Re[ U(φ) · exp(i m λ) ]
v(λ, φ) = Re[ V(φ) · exp(i m λ) ]
```

`U, V, Z` are normalized by the same Gaussian-quadrature norm used in
`Hough_Harmonics/Normalization.py::norm_Hough`, and follow its sign
convention (`Z` negated relative to the raw `Hough_harmonic` output, `U, V`
not) so the two visualization scripts and `norm_Hough`-based code stay
consistent.

### 2.4 Requirements

`Hough_spatial_ev.py` additionally requires **cartopy** (map projection +
gridlines), now listed in `requirements.txt`. The first run downloads and
caches Natural Earth data if coastlines are ever re-enabled (currently off
— see below); an internet connection is needed the first time cartopy's
`gridlines()`/projection machinery initializes.

### 2.5 Known quirks (worked around in the code)

- **Coastlines are off.** They were removed on request — this is an
  idealized-sphere mode plot, not a geographic map.
- **`bbox_inches='tight'` is not used when saving.** Combining cartopy's
  `draw_labels=True` gridlines with a `matplotlib` colorbar triggers a
  known `matplotlib`/`cartopy` bug where the axes' tight bounding box comes
  out `NaN`/`inf`, silently cropping the map out of the saved PNG (only the
  colorbar survives). Figure margins are set explicitly via
  `fig.subplots_adjust(...)` instead.
- **The title is a plain `ax.text()`, not `ax.set_title()`.** The same
  `NaN`/`inf` bug corrupts `matplotlib`'s automatic title-position update
  (it inspects the gridliner's hidden label extents), pushing the title to
  `y = inf` and making it invisible. A manually positioned axes-fraction
  text sidesteps the automatic positioning logic entirely.

If a future `matplotlib`/`cartopy` release fixes the underlying bug, both
workarounds are safe to simplify back to `ax.set_title()` +
`bbox_inches='tight'`.

---

## 3. References

See [`README_dispersion.md`](README_dispersion.md) §4 and the main
[`README.md`](README.md) for the underlying eigenvalue problem and
literature (Swarztrauber & Kasahara 1985; Longuet-Higgins 1968; Dourado
2025).
