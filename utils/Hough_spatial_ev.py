"""Spatial structure of a Hough mode (eigenvector) of the linear rotating
shallow-water equations on the sphere.

Plots the height field h(lambda, phi) as a filled contour and the velocity
field (u, v) as a quiver, both reconstructed from the mode's latitudinal
Hough functions via the zonal-wavenumber ansatz Re[(U,V,Z)(phi) * exp(i m
lambda)], on a PlateCarree map.

Run from the command line:

    python utils/Hough_spatial_ev.py output.png --m 3 --n 7 --alpha 3
    python utils/Hough_spatial_ev.py output.png --m 3 --n 7 --alpha 3 --he 5000
    python -m utils.Hough_spatial_ev output.png --m 1 --n 2 --alpha 3

or import and call it from another script:

    from utils.Hough_spatial_ev import hough_spatial_ev
    hough_spatial_ev(m=3, n=7, alpha=3, h_e=10000, path="output.png")
"""
import os
import sys

# Make the repository root importable so that "Hough_Harmonics" resolves
# whether this file is run directly (python utils/Hough_spatial_ev.py),
# run as a module (python -m utils.Hough_spatial_ev), or imported.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from Hough_Harmonics.Normalization import norm_Hough
from Hough_Harmonics.Eigenvalues_and_eigenvectors.Eigenvectors import Hough_harmonic


def label(m, n, alpha, height):

    l = ''
    if alpha == 1:
        l += 'EIG'
    elif alpha == 2:
        l += 'WIG'
    else:
        l += 'RH'

    l += f'({m},{n})  $h_e=${height}m'
    return l


def hough_spatial_ev(m, n, alpha, h_e: float = 10000, N: int = 10,
                      nlat: int = 91, nlon: int = 181,
                      central_longitude: float = 0,
                      quiver_step: int = 6, path: str = None):
    """Plot the spatial structure of a Hough mode on a PlateCarree map.

    Parameters
    ----------
    m, n, alpha : int
        Hough mode indices. ``alpha``: 1 = EIG, 2 = WIG, 3 = Rossby-Haurwitz.
    h_e : float, optional
        Equivalent height (equivalent depth) in metres. Default ``10000``.
    N : int, optional
        Truncation order of the Hough-mode expansion. Default ``10``.
    nlat, nlon : int, optional
        Number of latitude/longitude grid points for the reconstructed
        field. Default ``91``/``181`` (2-degree resolution).
    central_longitude : float, optional
        Longitude at the centre of the map. Default ``0``.
    quiver_step : int, optional
        Subsampling stride for the (u, v) quiver arrows. Default ``6``.
    path : str or None, optional
        If given, the figure is saved to this path (PNG, 200 dpi) and the
        figure is closed. If ``None`` (default), the figure is shown
        interactively with ``plt.show()``.

    Returns
    -------
    None
    """
    l = label(m, n, alpha, h_e)
    g = 9.8
    a = 6.38e+06
    omega = 2 * np.pi / 24 / 60 / 60
    eps = (4 * a * a * omega * omega) / (g * h_e)
    gamma = 1 / np.sqrt(eps)

    # Normalization constant, from the same quadrature-based routine used
    # in Hough_and_derivatives.py, so the two scripts stay consistent.
    _, _, _, _, _, _, _, norm, eigen = norm_Hough(m, n, alpha, gamma, N, 60)

    lat = np.linspace(-90, 90, nlat)
    phi = np.deg2rad(lat)

    U = np.empty(nlat, dtype=complex)
    V = np.empty(nlat, dtype=complex)
    Z = np.empty(nlat, dtype=complex)

    for i, p in enumerate(phi):
        u, v, z, du, dv, dz, _ = Hough_harmonic(m, n, alpha, gamma, p, N)
        U[i] = u
        V[i] = v
        Z[i] = z

    # Same sign convention as norm_Hough (Z is flipped there, U/V are not)
    U = U / np.sqrt(norm)
    V = V / np.sqrt(norm)
    Z = -Z / np.sqrt(norm)

    lon = np.linspace(-180, 180, nlon)
    LON, LAT = np.meshgrid(lon, lat)
    phase = np.exp(1j * m * np.deg2rad(LON))

    # Static snapshot at phase = 0: full spatial field is
    # Re[ (U,V,Z)(phi) * exp(i*m*lambda) ]
    H_field = np.real(Z[:, None] * phase)
    U_field = np.real(U[:, None] * phase)
    V_field = np.real(V[:, None] * phase)

    fig = plt.figure(figsize=(9, 5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=central_longitude))
    fig.subplots_adjust(left=0.07, right=0.92, top=0.93, bottom=0.09)
    ax.set_global()
    gl = ax.gridlines(linestyle='--', linewidth=0.4, draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    vmax = np.max(np.abs(H_field))
    cf = ax.contourf(LON, LAT, H_field, levels=30, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, transform=ccrs.PlateCarree())
    plt.colorbar(cf, ax=ax, orientation='vertical', shrink=0.7, label=r'$h$')

    ax.quiver(LON[::quiver_step, ::quiver_step], LAT[::quiver_step, ::quiver_step],
              U_field[::quiver_step, ::quiver_step], V_field[::quiver_step, ::quiver_step],
              transform=ccrs.PlateCarree(), regrid_shape=20, color='k')

    # NOTE: ax.set_title() is avoided here: matplotlib's automatic title-
    # position update reads the gridliner's (hidden) top-label text extents,
    # which come out NaN/inf for this projection + colorbar combination and
    # push the title itself to y=inf, making it invisible. A plain axes-
    # fraction text sidesteps that auto-positioning entirely.
    ax.text(0.5, 1.08, l, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=plt.rcParams['axes.titlesize'],
            fontweight=plt.rcParams['axes.titleweight'])

    if path:
        # NOTE: bbox_inches='tight' is avoided here: combining gridline
        # labels (draw_labels=True) with a colorbar triggers a known
        # matplotlib/cartopy bug where the axes tight-bbox comes out
        # NaN/inf, which silently crops the map out of the saved figure.
        # Margins are set explicitly via subplots_adjust instead.
        fig.savefig(path, dpi=200)
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot the spatial structure (h contour + (u,v) quiver) "
                    "of a Hough mode of the rotating shallow-water equations "
                    "on the sphere.")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="output image path (e.g. hough_mode.png). "
             "If omitted, the figure is shown interactively.")
    parser.add_argument(
        "--m", type=int, default=1, help="zonal wavenumber (default: 1).")
    parser.add_argument(
        "--n", type=int, default=2, help="meridional mode index (default: 2).")
    parser.add_argument(
        "--alpha", type=int, default=3, choices=[1, 2, 3],
        help="mode type: 1=EIG, 2=WIG, 3=Rossby-Haurwitz (default: 3).")
    parser.add_argument(
        "--he", "--equivalent-height", dest="he", type=float, default=10000,
        help="equivalent height in metres (default: 10000).")
    parser.add_argument(
        "--central-longitude", dest="central_longitude", type=float, default=0,
        help="central longitude of the map in degrees (default: 0).")
    args = parser.parse_args()

    hough_spatial_ev(args.m, args.n, args.alpha, h_e=args.he,
                      central_longitude=args.central_longitude, path=args.path)
