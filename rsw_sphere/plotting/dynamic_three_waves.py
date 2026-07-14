import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.colors import ListedColormap

from rsw_sphere.hough_harmonics.normalization import norm_component

from rsw_sphere.dynamics.dynamic_triads import TRIAD
from rsw_sphere.dynamics.dynamic_triads import Triad_dynamics

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

def triad_evolution(h_e, m_a, n_a, alpha_a, m_b, n_b, 
                 alpha_b, m_c,n_c, alpha_c, u_a, u_b, u_c, t_0, t_f,h, path ):

    g = 9.8
    a = 6.38e+06
    omega = 2*np.pi/24/60/60
    eps = (4*a*a*omega*omega)/(g*h_e)
    gamma = 1/np.sqrt(eps)

    N = 10
    deg = 300

    Triad = TRIAD(gamma, m_a, n_a, alpha_a, m_b, n_b, 
                alpha_b, m_c,n_c, alpha_c, N, deg)

    nu_a = norm_component(Triad.uvh_a[0])*np.sqrt(g*h_e)
    nu_b = norm_component(Triad.uvh_b[0])*np.sqrt(g*h_e)
    nu_c = norm_component(Triad.uvh_c[0])*np.sqrt(g*h_e)

    A_a = u_a/nu_a
    A_b = u_b/nu_b
    A_c = u_c/nu_c #50
    A_0 = np.array([A_a,A_b,A_c])

    l_a = label(m_a, n_a, alpha_a)
    l_b = label(m_b, n_b, alpha_b)
    l_c = label(m_c, n_c, alpha_c)

    print('--------COUPLING COEFFICIENTS--------------')
    print(f'Coef {l_a}: {np.real(Triad.coef_ABC)}')
    print(f'Coef {l_b}: {np.real(Triad.coef_BAC)}')
    print(f'Coef {l_c}: {np.real(Triad.coef_CAB)}')
    print()
    print('Mismatch: ', np.real(Triad.mismatch))
    print()
    print('---------------FREQUENCIES-----------------')
    print(f'Freq {l_a}: {np.real(Triad.freq_a)} ' )
    print(f'Freq {l_b}: {np.real(Triad.freq_b)} ' )
    print(f'Freq {l_c}: {np.real(Triad.freq_c)} ' )
    print()
    print('E_total constraint: ', np.real(Triad.coef_BAC + Triad.coef_ABC - Triad.coef_CAB + Triad.mismatch*Triad.Sabc))

    Triad_dynamics(Triad, A_0, t_0, t_f, h, path)
    #eff_tri(Triad, A_0, u_a)
    #Triad_Precession(Triad, t_0, t_f, h, vel_a = 20, pri =False)

if __name__ == "__main__":
    triad_evolution(10000, 1,1,3,3,4,3,4,5,3,10,10,10,0,100,0.001)