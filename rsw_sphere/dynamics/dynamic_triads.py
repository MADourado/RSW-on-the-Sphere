"""Single resonant triad: the three coupling coefficients, the frequency
mismatch, the cubic-energy integral, and the amplitude-equation RHS.

``WaveSet`` (``wave_sets.py``) generalizes this to any number of modes and
triads; ``TRIAD`` is kept as the independent reference implementation the
two are cross-checked against (see ``check_wave_set_physics.py``, C1-C3).
"""
import numpy as np

from rsw_sphere.dynamics.integrators import RK44  # re-exported for callers
from rsw_sphere.hough_harmonics.normalization import norm_Hough
from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.eigenvectors import symetry
from rsw_sphere.hough_harmonics.inner_product import inner_product
from rsw_sphere.hough_harmonics.inner_product import S_abc

import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)

def label(m,n,alpha):

    l = ''
    if alpha == 1:
        l += 'EIG'
    elif alpha == 2:
        l += 'WIG'
    else:
        l += 'RH '
    
    l += f'({m},{n})'
    return l

class TRIAD:
    
    def __init__(self, gamma, m_a, n_a, alpha_a, m_b, n_b, 
                 alpha_b, m_c,n_c, alpha_c, N=10, deg = 60):
        
        self.mode_a = np.array([m_a, n_a, alpha_a])
        self.mode_b = np.array([m_b, n_b, alpha_b])
        self.mode_c = np.array([m_c, n_c, alpha_c])
        
        A = norm_Hough(m_a,n_a,alpha_a,gamma, N,deg)
        eigen_a = A[-1]
        A = A[:-3]
        
        self.uvh_a = A
        self.label_a = label(m_a, n_a, alpha_a)
        
        B = norm_Hough(m_b,n_b,alpha_b,gamma, N,deg)
        eigen_b = B[-1]
        B = B[:-3]
        
        self.uvh_b = B
        self.label_b = label(m_b, n_b, alpha_b)
        
        C = norm_Hough(m_c,n_c,alpha_c,gamma, N,deg)
        eigen_c = C[-1]
        C = C[:-3]
        
        self.uvh_c = C
        self.label_c = label(m_c, n_c, alpha_c)

        if symetry(m_a, n_a, alpha_a) and symetry(m_b, n_b, alpha_b) and symetry(m_c, n_c, alpha_c):
            fat = -1
        else:
            fat = 1
        
        self.freq_a = eigen_a
        self.freq_b = eigen_b
        self.freq_c = eigen_c
        
        inner_ABC  = inner_product(A, m_a,B, m_b, C, m_c, deg, True)  # projection on mode a
        inner_BAC  = inner_product(B, m_b,A, m_a, C, m_c, deg, True)  # mode b
        inner_CAB  = inner_product(C, m_c,A, m_a, B, m_b, deg, False) # mode c
        
        self.coef_ABC = fat * gamma * inner_ABC
        self.coef_BAC = fat * gamma * inner_BAC
        self.coef_CAB = fat * gamma * inner_CAB
        
        self.mismatch = -self.freq_c+ self.freq_b + self.freq_a
        
        self.Sabc = -fat * S_abc(A,m_a,B,m_b,C,m_c,deg)
        
    def f(self, AMP):
        
        coef_ABC = self.coef_ABC
        coef_BAC = self.coef_BAC
        coef_CAB = self.coef_CAB
        
        A_a, A_b, A_c = AMP
        
        F1 = -1j * self.freq_a * A_a + 1j * coef_ABC * np.conj(A_b) * A_c
        F2 = -1j * self.freq_b * A_b + 1j * coef_BAC * np.conj(A_a) * A_c
        F3 = -1j * self.freq_c * A_c + 1j * coef_CAB * A_a * A_b
        
        
        return np.array([F1, F2, F3])


def Energy_0(Triad, A_0):
    
    A_a, A_b, A_c = A_0
    
    Energy_02  = (A_a * np.conj(A_a) + A_b * np.conj(A_b) + A_c * np.conj(A_c) )
    Energy_03 = (2*np.real(np.conj(A_a) * np.conj(A_b) * A_c) * Triad.Sabc)
    
    return  Energy_02, Energy_03
