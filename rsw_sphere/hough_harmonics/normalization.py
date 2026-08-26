import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from .eigenvalues_and_eigenvectors.matrix_system import matriz_A
from .eigenvalues_and_eigenvectors.matrix_system import matriz_B
from .eigenvalues_and_eigenvectors.eigenvectors import Hough_harmonic
from .eigenvalues_and_eigenvectors.eigenvectors import symetry
from rsw_sphere.physics import gamma_from_he


def norm_Hough(m,n,alpha,gamma, N,deg):
    
    # Considering the inner product, here the norm of a Hough harmonic
    # is calculated, in order to define an orthonormal basis
    
    # Gaussian quadrature is used for the integral
    point, weight = np.polynomial.legendre.leggauss(deg)

    # NOTE: the symmetric/antisymmetric branching (Hough_coef_A vs.
    # Hough_coef_B) already happens *inside* Hough_harmonic, based on
    # symetry(m, n, alpha) -- see eigenvalues_and_eigenvectors/eigenvectors.py.
    # A second, symmetry-dependent overall sign flip of (U, V, Z) was
    # previously attempted here but left disabled (unreachable code below
    # an unconditional return). It has been removed: Hough_harmonic's own
    # branching already yields consistent relative signs and normalization
    # for both symmetric and antisymmetric modes, confirmed by reproducing
    # coupling coefficients for an antisymmetric-mode triad against an
    # independent reference. A caller that constructs coupling coefficients
    # directly from Hough_coef_A/Hough_coef_B (bypassing Hough_harmonic and
    # this branching) must replicate symetry()'s A/B choice itself, or
    # antisymmetric modes will silently use the wrong eigenvector.

    # Interval [-1,1] is changed to [-pi/2, pi/2]
    ang = np.pi/2 * point
    
    U   = []
    V   = []
    Z   = []
    DU  = []
    DV  = []
    DZ  = []
    
    for phi in ang:
        
        # Calculating the fields and their derivatives in each latitude
        # for the quadrature
        u,v,z,du,dv,dz, eigen = Hough_harmonic(m, n, alpha, gamma, phi, N)
        
        U   += [u]
        V   += [v]
        Z   += [z]
        
        DU  += [du]
        DV  += [dv]
        DZ  += [dz]
        
    U   = np.array(U)
    V   = np.array(V)
    Z   = np.array(Z)
    DU  = np.array(DU)
    DV  = np.array(DV)
    DZ  = np.array(DZ) 
    cos = np.cos(ang)
    
    # This is the integral, recalling that once the change of variables is
    # applyed, a factor of pi/2 must be considered
    norm = np.pi/2* sum(weight * (U*U + V*V + Z*Z)*cos)
    
    # Normalizing the fields
    U = 1/np.sqrt(norm)*U
    V = 1/np.sqrt(norm)*V
    Z = 1/np.sqrt(norm)*Z
    
    DU = 1/np.sqrt(norm)*DU
    DV = 1/np.sqrt(norm)*DV
    DZ = 1/np.sqrt(norm)*DZ
    
    return U,V,-Z,-DU,-DV,DZ,point, norm, eigen

    # DEAD CODE, kept for the record (unreachable -- the function already
    # returned above). This was an attempt at an additional symmetry-
    # dependent overall sign flip of (U, V, Z), on top of the
    # Hough_coef_A/Hough_coef_B branching already done inside
    # Hough_harmonic. Investigated and found unnecessary: the unconditional
    # return above already gives consistent signs/normalization for both
    # symmetric and antisymmetric modes (see the note above norm_Hough).
    # s = symetry(m, n, alpha)
    #
    # if s:
    #     return U, V, -Z, -DU, -DV, DZ, point, norm, eigen
    # else:
    #     return -U, -V, Z, DU, DV, -DZ, point, norm, eigen
    #
    # if s:
    #     return -U, -V, Z, DU, DV, -DZ, point, norm, eigen
    # else:
    #     return U, V, -Z, -DU, -DV, DZ, point, norm, eigen

def norm_component(u, deg = 300):

    # Norm of the zonal velocity component (u)
    # This norm is used to define the necessary amplitude to obtain a
    # specific zonal velocity.

    point, weight = np.polynomial.legendre.leggauss(deg)

    ang = np.pi/2 * point

    cos = np.cos(ang)

    norm = np.pi/2* sum(weight * (u*u)*cos)

    return np.sqrt(norm)


def velocity_to_amplitude(u_target, u_component, h_e, g=9.8):
    """Complex (here: real) amplitude giving a physical zonal velocity of
    ``u_target`` m/s for a real initial amplitude.

    **Includes the factor of 2 from the paper's own derivation**
    (``eq: Azonal`` in ``JFM-template.tex``): the physical field is
    ``A*(u,iv,h) + conj(A)*(u,-iv,h) = 2*Re(A)*u`` (for the u-component),
    so a real amplitude ``A`` produces a physical zonal velocity
    ``U = 2*A*u_component``, i.e. ``A = U / (2 * norm_component(u_component)
    * sqrt(g*h_e))``.

    This factor of 2 was missing from the (retired) §2.2 triad toolchain's
    own velocity-to-amplitude conversions until 2026-08-11 -- found while
    cross-checking the new
    quartet/quintet layer's harvested parameters against the dissertation's
    published ``tab: cap4ex`` amplitudes: every one of 4 independent
    (mode, velocity) pairs matched the published value to ~4-5 significant
    figures only with this factor included, and was consistently exactly
    2x too large without it. All velocity-labeled §2.2 figures/captions
    were regenerated after this fix;
    the underlying mode frequencies, periods, coupling coefficients and
    mismatches are entirely unaffected (they don't depend on amplitude).

    Parameters
    ----------
    u_target : float
        Desired physical zonal velocity, m/s.
    u_component : ndarray
        The mode's ``u`` field (e.g. ``TRIAD.uvh_a[0]``), as returned by
        ``norm_Hough``.
    h_e : float
        Equivalent height, m.
    g : float, optional
        Gravitational acceleration, m/s^2. Default 9.8.

    Returns
    -------
    float
        Real initial amplitude.
    """
    return u_target / (2 * norm_component(u_component) * np.sqrt(g * h_e))


#------------------
# PLOTS
#------------------

def label(m,n,alpha,height):

    l = ''
    if alpha == 1:
        l += 'EIG'
    elif alpha == 2:
        l += 'WIG'
    else:
        l += 'RH'
    
    l += f'({m},{n}) at {height}m'
    return l

def hough_and_derivatives(m,n,alpha, h_e:int = 10000):

    l = label(m,n,alpha, h_e)
    eps, gamma = gamma_from_he(h_e)

    U,V,Z,DU, DV, DZ, ANG,norm, eigen = norm_Hough(m, n, alpha, gamma, 10, 60)

    angle = np.pi/2*ANG

    U_1 = []
    V_1 = []
    Z_1 = []
    DU_1 = []
    DV_1 = []
    DZ_1 = []

    for phi in angle: 
        
        u_1,v_1,z_1,du1, dv1, dz1, eigen = Hough_harmonic(m,n,alpha,  gamma, phi, 30)
        
        U_1 += [u_1]    
        V_1 += [v_1]
        Z_1 += [z_1]  
        
        DU_1 += [du1]
        DV_1 += [dv1]
        DZ_1 += [dz1]
        
    U_1 = 1/np.sqrt(norm) * np.array(U_1) 
    V_1 = 1/np.sqrt(norm) * np.array(V_1)
    Z_1 = 1/np.sqrt(norm) * np.array(Z_1)

    DU_1 = 1/np.sqrt(norm) * np.array(DU_1)
    DV_1 = 1/np.sqrt(norm) * np.array(DV_1)
    DZ_1 = 1/np.sqrt(norm) * np.array(DZ_1)

    
    ANG = 90*ANG
    
    plt.plot(ANG,U,label = r'$u$')  
    plt.plot(ANG,V,label = r'$v$')
    plt.plot(ANG,Z,label = r'$h$')

    plt.ylim([-1.5,1.5])
    plt.xlim([0,90])
    x_ticks = np.linspace(0,90,7) 
    plt.xticks(x_ticks)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend()
    plt.title(l)
    plt.xlabel(r'Latitude ($\phi$) - deg')
    plt.show()

    plt.plot(ANG, DU, label = 'DU')
    plt.plot(ANG, DV, label = 'DV')
    plt.plot(ANG, DZ, label = 'DZ')

    plt.ylim([-1.5,1.5])
    plt.xlim([0,90])
    x_ticks = np.linspace(0,90,7) 
    plt.xticks(x_ticks)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend()
    plt.title('Derivative - ' + l)
    plt.xlabel(r'Latitude ($\phi$) - deg')
    plt.show()

if __name__ == "__main__":
    hough_and_derivatives(1,2,3,10000)

    


    
        
        
    
    