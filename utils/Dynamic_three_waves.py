import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.colors import ListedColormap

from Hough_Harmonics.Normalization import norm_component

from Dynamic_Triads import TRIAD
from Dynamic_Triads import Triad_dynamics

def triad_evolution(h_e, m_a, n_a, alpha_a, m_b, n_b, 
                 alpha_b, m_c,n_c, alpha_c, u_a, u_b, u_c, t_0, t_f, ):

    g = 9.8
    h_e = 10000

    a = 6.38e+06
    omega = 2*np.pi/24/60/60
    eps = (4*a*a*omega*omega)/(g*h_e)
    #eps = 8.810572669756384
    gamma = 1/np.sqrt(eps)

    N = 10
    deg = 300

    t_0 = 0
    t_f = 100#25.2
    h   = 0.01

    m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,1,1, 3,4,3, 4,5,3

    Triad = TRIAD(gamma, m_a, n_a, alpha_a, m_b, n_b, 
                alpha_b, m_c,n_c, alpha_c, N, deg)

    nu_a = norm_component(Triad.uvh_a[0])*np.sqrt(g*h_e)
    nu_b = norm_component(Triad.uvh_b[0])*np.sqrt(g*h_e)
    nu_c = norm_component(Triad.uvh_c[0])*np.sqrt(g*h_e)

    A_a = 2*u_a/nu_a
    A_b = 2*u_b/nu_b
    A_c = 2*u_c/nu_c #50
    A_0 = np.array([A_a,A_b,A_c])

    Triad_dynamics(Triad, A_0, t_0, t_f=100, h=0.001, p = True)
    #eff_tri(Triad, A_0, u_a)
    #Triad_Precession(Triad, t_0, t_f, h, vel_a = 20, pri =False)

    print()
    print('Coef A = ', Triad.coef_ABC)
    print('Coef B = ', Triad.coef_BAC)
    print('Coef C = ', Triad.coef_CAB)
    print('a / c = ', Triad.coef_ABC/Triad.coef_CAB)
    print('Mismatch = ', Triad.mismatch)
    print()
    print(f'Freq {Triad.label_a} = {Triad.freq_a} ' )
    print(f'Freq {Triad.label_b} = {Triad.freq_b} ' )
    print(f'Freq {Triad.label_c} = {Triad.freq_c} ' )
    print()
    print('Triad = ', Triad.coef_BAC + Triad.coef_ABC - Triad.coef_CAB + Triad.mismatch*Triad.Sabc)

if __name__ == "__main__":
    triad_evolution(10000, 1,1,3,3,4,3,4,5,3,10,10,10,0,100)