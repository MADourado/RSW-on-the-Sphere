# Dispersion relation of the RSW normal modes on the sphere

`rsw_sphere/plotting/dispersion_relation_fancy.py` computes and plots the **dispersion relation
of the normal modes (Hough harmonics) of the linear rotating shallow-water
(RSW) equations on a sphere**. The output is a publication-quality diagram of
frequency `ω` versus zonal wavenumber `m`, showing the three wave families
— eastward inertia-gravity (EIG), westward inertia-gravity (WIG) and
Rossby-Haurwitz (RH) — together with the special Kelvin and mixed
Rossby-gravity (Yanai) modes.

---

## 1. Usage

The script is standalone and callable in four equivalent ways.

### Command line

```bash
# save to a file (positional path argument)
python rsw_sphere/plotting/dispersion_relation_fancy.py dispersion_exp.png

# choose a different equivalent height (metres)
python rsw_sphere/plotting/dispersion_relation_fancy.py dispersion_exp.png --he 5000

# run as a module from the repository root
python -m rsw_sphere.plotting.dispersion_relation_fancy dispersion_exp.png

# show interactively instead of saving (omit the path)
python rsw_sphere/plotting/dispersion_relation_fancy.py
```

`python rsw_sphere/plotting/dispersion_relation_fancy.py --help` prints the full argument list.

### From another script

```python
from rsw_sphere.plotting.dispersion_relation_fancy import dispersion_relation

dispersion_relation(h_e=10000, path="dispersion_exp.png")   # save
dispersion_relation()                                        # show interactively
```

### Arguments

| Argument | Type | Default | Meaning |
|----------|------|---------|---------|
| `path`   | str  | `None`  | Output image path. If omitted / `None`, the figure is shown with `plt.show()` instead of being saved. |
| `--he` (`h_e`) | float | `10000` | Equivalent height (equivalent depth) in metres. |

The script requires the repository root to be importable (so that
`rsw_sphere` is found). It handles this automatically — direct execution,
`-m` module execution and `import` all work.

### Requirements

See `requirements.txt` (pinned for API compatibility):
`numpy<2.0`, `scipy<1.15`, `matplotlib`, `pyyaml`.

---

## 2. The mathematics

### 2.1 Governing equations

The starting point is the linearised rotating shallow-water model on a sphere of
radius `a` rotating at rate `Ω` — **Laplace's tidal equations**. Following the
thesis (§1.3) and Swarztrauber & Kasahara (1985), one non-dimensionalises with
the mean/equivalent height `h₀` as vertical scale, the gravity-wave speed
`c = √(g h₀)` as velocity scale and `(2Ω)⁻¹` as time scale. Dropping primes, the
dimensionless tidal equations (thesis eqs 1.33–1.35) are

```
∂u/∂t − sinφ · v + (γ / cosφ) ∂h/∂λ            = 0
∂v/∂t + sinφ · u +  γ ∂h/∂φ                    = 0
∂h/∂t + (γ / cosφ) [ ∂u/∂λ + ∂(v cosφ)/∂φ ]    = 0
```

with longitude `λ`, latitude `φ`, zonal wavenumber `m ∈ ℤ`, and the single
parameter `γ` (below). The Coriolis parameter is `f = 2Ω sinφ`, which in these
units reduces to `sinφ`. (Notation: the code calls the equivalent height `h_e`;
the thesis calls it `h₀` — they are the same quantity.)

### 2.2 Eigenvalue problem and non-dimensional parameters

Seeking normal modes `H(λ, φ, t) = Θ(φ) · exp[ i (m λ − ω t) ]` reduces the tidal
equations to the eigenvalue problem `(L − i ω) H = 0` (thesis eq 2.2). The
eigenvalue `ω` is the **dimensionless frequency** — the physical frequency scaled
by `2Ω`. To avoid clashing with the physical `ω` used in the plot, this README
writes the dimensionless eigenvalue as `σ` (so `σ = ω_physical / 2Ω`, and the
thesis's `ω` is our `σ`).

The single controlling parameter is `γ` (thesis eq after 1.11) and, equivalently,
**Lamb's parameter** (Lamb's number, thesis eq 1.12):

```
γ = √(g h₀) / (2 Ω a) = c / (2 Ω a) ,       c = √(g h₀),
ε = 1 / γ² = 4 a² Ω² / (g h₀) = (2 Ω a / c)² .
```

`c` is the shallow-water gravity-wave speed; `ε` measures the relative influence
of rotation (`2Ω a`) versus gravity waves (`c`). For `h₀ = 10 km`:
`c ≈ 313 m/s`, `ε ≈ 8.8`, `γ ≈ 0.34`.

### 2.3 Spectral (Hough) discretisation

Expanding `û, v̂, Φ̂` in vector spherical harmonics of zonal order `m` and using
the recurrence relations for `sinφ ·` and `d/dφ ·` turns the tidal equations into
a **generalised algebraic eigenvalue problem** for `σ` (the infinite system of
Swarztrauber & Kasahara 1985, truncated). Modes separate by parity about the
equator, so for each zonal wavenumber `m` one assembles two matrices:

- `A(m, γ, N)` — the *symmetric* structure (`u`, `h` symmetric about the equator,
  `v` antisymmetric),
- `B(m, γ, N)` — the *antisymmetric* structure,

where — per thesis §2.1 — inertia-gravity modes are symmetric when `n + m` is
**even**, and Rossby-Haurwitz modes are symmetric when `n + m` is **odd**
(`utils`→`rsw_sphere/hough_harmonics/eigenvalues_and_eigenvectors/matrix_system.py`).
For the special case `m = 0`:

- `C(γ, N)`, `D(γ, N)`  (`.../Matrix_m0.py`).

`N` is the spectral truncation; the script uses `N = 3`, giving `6N = 18`
eigenvalues per `m`. The matrix entries are built from three coefficients:

```
q(n,m) = m / [ n (n+1) ]                                   (rotation / Coriolis)
p(n,m) = √{ (n−1)(n+1)(n−m)(n+m) / [ n² (2n−1)(2n+1) ] }   (meridional coupling)
r(n,γ) = γ √[ n (n+1) ]                                    (gravity, scaled by γ)
```

Diagonalising `A` and `B` (and `C`, `D` at `m = 0`) yields the dimensionless
frequencies `σ`. The plot converts them to physical units through

```
ω = 2Ω · σ ,   shown in units of 10⁻⁴ rad s⁻¹  (scale factor s = 2Ω × 10⁴).
```

### 2.4 The three wave families and the meridional index

A mode is labelled `(m, n, α)` — zonal wavenumber `m`, meridional wavenumber `n`
(the spherical-harmonic degree), and type `α ∈ {1, 2, 3}` (thesis §2.1). For
every `m` the eigenvalues fall into three families, indexed in the dispersion
diagram by the **meridional index** `l = n − m ≥ 0`:

| Family (thesis abbrev.) | code `alpha` | sign of `σ` | description |
|--------|:---:|:---:|-------------|
| **EIG** (thesis **EG**) | 1 | `σ > 0` (large) | eastward inertia-gravity |
| **WIG** (thesis **WG**) | 2 | `σ < 0` (large) | westward inertia-gravity |
| **RH**  | 3 | `|σ| ≪ 1`      | Rossby-Haurwitz (rotational, westward) |

(The thesis abbreviates the gravity families **EG**/**WG**; this figure and the
code use **EIG**/**WIG** for the same "Eastward/Westward Inertia-Gravity" waves.)

Two families contain **special modes** — the `n = m` (`l = 0`) curves that the
thesis (Fig. 2.1 caption) highlights:

- **Kelvin wave** = EIG with `l = 0` (`n = m`); asymptotically `ω ≈ c k`. In
  thesis notation this is the mode `(1,1,EG)`.
- **Mixed Rossby-gravity (MRG / Yanai)** = the `l = 0` rotational curve, i.e.
  RH with `n = m` (the "mixed mode" of the thesis's RH panel). On the combined
  diagram this single physical mode appears as two branches split by propagation
  direction — a westward branch (RH `l = 0`) and an eastward, gravity-like
  branch (EIG `l = 1`) — both drawn in green.

When the 18 eigenvalues per `m` are sorted ascending, the columns map to
`(family, l)` as follows (verified against the Hough-function builder
`Eigenvectors.py`, stable for all `m`):

```
idx 0–5   : WIG,  l = 5,4,3,2,1,0     (most negative → highest l)
idx 6–11  : RH,   l = 0,1,2,3,4,5     (idx 6 = MRG-west, l = 0)
idx 12–17 : EIG,  l = 0,1,2,3,4,5     (idx 12 = Kelvin, idx 13 = MRG-east)
```

### 2.5 Relation to the Matsuno equatorial β-plane

In the strong-trapping limit (`ε → ∞`) the spherical modes converge to the
classical Matsuno equatorial β-plane modes, whose signed meridional index
`n_M` relates to `l = n − m` by

```
EIG:  n_M = l − 1        (Kelvin l=0 → n_M=−1;  MRG-east l=1 → n_M=0)
WIG:  n_M = l + 1
RH :  n_M = l            (MRG-west l=0 → n_M=0)
```

The negative Matsuno indices (`n_M = −1` for Kelvin) are a β-plane artefact of
counting meridional nodes of `v`; on the sphere `l = n − m ≥ 0` always, since
`n ≥ m`. (This correspondence was checked numerically: at `ε ≈ 880` the sphere
frequencies match the Matsuno cubic roots to ~1–4 %.)

### 2.6 Reference curves

- **`ω ≈ c k`** (grey dotted): the short-wave (large `|m|`) inertia-gravity
  asymptote, with `c = √(g h_e)` and `k = m / a`.
- **`2Ω`** (grey dashed horizontal): the rotation frequency, close to the
  low-`m` minimum of the inertia-gravity branches. (On the sphere the Coriolis
  parameter `f = 2Ω sinφ` varies with latitude, so `2Ω` is its polar maximum,
  **not** a constant `f₀` — the label is deliberately `2Ω`, not `f₀`.)

---

## 3. The output figure

A square diagram of `ω` (in `10⁻⁴ rad s⁻¹`) versus zonal wavenumber `m`,
`m ∈ [−10, 10]`, `ω ∈ [0, 3.5]`. Because RH and WIG have negative frequencies,
`−ω` is plotted for them, so the horizontal axis encodes direction:
**eastward** (`m > 0`) on the right, **westward** (`m < 0`) on the left.

Elements:

- **Blue solid** — inertia-gravity branches (EIG on the right, WIG on the left),
  the heavy blue line being the **Kelvin** wave.
- **Green dashed** — the **Mixed Rossby-Gravity (Yanai)** mode (eastward branch
  `l = 1`, westward branch `l = 0`); a single label points to it.
- **Red solid** — the **Rossby-Haurwitz** branches (thin lines near the bottom).
- **`n − m` labels** along the top edge (and, for RH, at the left edge) give the
  meridional index `l` of each curve.
- **Right-hand axis** — the corresponding **period** `T = 2π/ω` in days
  (ticked at round-number periods to keep the labels distinct).
- Grey reference curves `ω ≈ c k` and the `2Ω` line, as in §2.6.

Changing `--he` (equivalent height) rescales `ε`, `c`, and hence the whole
spectrum: smaller `h_e` ⇒ larger `ε` ⇒ stronger equatorial trapping and lower
gravity-wave speed.

**Relation to thesis Figure 2.1.** The thesis figure shows the same spectrum but
in a different presentation: three *separate* panels (EG, WG, RH), the vertical
axis in **dimensionless frequency** `σ` (not `10⁻⁴ rad s⁻¹`), `m` from 0 to 20,
and a truncation of `N = 30` (with Gauss–Legendre normalisation on 60 points).
This script instead draws a single **combined** panel in physical units, `m` from
−10 to 10, at `N = 3`, and adds the `ω ≈ c k` and `2Ω` reference curves and the
Matsuno correspondence (§2.5), which are not part of the thesis figure. The
underlying eigenvalues are the same.

---

## 4. References

Following the thesis (Dourado, M. A., *Nonlinear wave interactions in rotating
shallow water equations on the sphere*, MSc dissertation, IME-USP, 2025):

- Swarztrauber, P. N. & Kasahara, A. (1985). *The vector harmonic analysis of
  Laplace's tidal equations.* SIAM J. Sci. Stat. Comput., 6, 464–491. — the
  spectral method implemented here.
- Longuet-Higgins, M. S. (1968). *The eigenfunctions of Laplace's tidal
  equations over a sphere.* Phil. Trans. R. Soc. A, 262, 511–607.
- Kasahara, A. (1977, 1978). *…global barotropic primitive equations with Hough
  harmonic expansion.* J. Atmos. Sci. — the `m = 0` treatment.
- Matsuno, T. (1966). *Quasi-geostrophic motions in the equatorial area.*
  J. Meteor. Soc. Japan, 44, 25–43; Ripa, P. (1981, 1983) — the equatorial
  β-plane limit referenced in §2.5.
