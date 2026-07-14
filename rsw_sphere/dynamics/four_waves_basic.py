import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from rsw_sphere.hough_harmonics.normalization import norm_Hough
from rsw_sphere.hough_harmonics.normalization import norm_component
from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.eigenvectors import symetry
from rsw_sphere.hough_harmonics.inner_product import inner_product
from rsw_sphere.hough_harmonics.inner_product import S_abc


class FOUR_WAVES:

    def __init__(self, gamma, m_a, n_a, alpha_a, m_b, n_b,
                 alpha_b, m_c, n_c, alpha_c, m_d, n_d, alpha_d,
                 N=10, deg=300):

        self.mode_a = np.array([m_a, n_a, alpha_a])
        self.mode_b = np.array([m_b, n_b, alpha_b])
        self.mode_c = np.array([m_c, n_c, alpha_c])
        self.mode_d = np.array([m_d, n_d, alpha_d])

        A = norm_Hough(m_a, n_a, alpha_a, gamma, N, deg)
        eigen_a = A[-1]
        A = A[:-3]

        self.uvh_a = A

        B = norm_Hough(m_b, n_b, alpha_b, gamma, N, deg)
        eigen_b = B[-1]
        B = B[:-3]

        self.uvh_b = B

        C = norm_Hough(m_c, n_c, alpha_c, gamma, N, deg)
        eigen_c = C[-1]
        C = C[:-3]
        
        self.uvh_c = C

        D = norm_Hough(m_d, n_d, alpha_d, gamma, N, deg)
        eigen_d = D[-1]
        D = D[:-3]

        self.uvh_d = D

        self.freq_a = eigen_a
        self.freq_b = eigen_b
        self.freq_c = eigen_c
        self.freq_d = eigen_d

        inner_ABC = inner_product(A, m_a, B, m_b, C, m_c, deg, False)
        inner_ABD = inner_product(A, m_a, B, m_b, D, m_d, deg, False)

        inner_BAC = inner_product(B, m_b, C, m_c, A, m_a, deg, True)
        inner_BAD = inner_product(B, m_b, D, m_d, A, m_a, deg, True)

        inner_CAB = inner_product(C, m_c, B, m_b, A, m_a, deg, True)

        inner_DAB = inner_product(D, m_d, B, m_b, A, m_a, deg, True)

        self.coef_ABC = -gamma * inner_ABC
        self.coef_ABD = -gamma * inner_ABD
        self.coef_BAC = -gamma * inner_BAC
        self.coef_BAD = -gamma * inner_BAD
        self.coef_CAB = -gamma * inner_CAB
        self.coef_DAB = -gamma * inner_DAB

        self.mismatch_1 = self.freq_a - self.freq_c - self.freq_b
        self.mismatch_2 = self.freq_a - self.freq_b - self.freq_d

        self.S_abc = S_abc(C, m_c, B, m_b, A, m_a, deg)
        self.S_abd = S_abc(B, m_b, D, m_d, A, m_a, deg)

    def f(self, AMP):

        coef_ABC = self.coef_ABC
        coef_ABD = self.coef_ABD
        coef_BAC = self.coef_BAC
        coef_BAD = self.coef_BAD
        coef_CAB = self.coef_CAB
        coef_DAB = self.coef_DAB

        A_a, A_b, A_c, A_d = AMP

        F1 = -1j * self.freq_a * A_a + 1j * coef_ABC * A_b * A_c
        F1 += 1j * coef_ABD * A_b * A_d

        F2 = -1j * self.freq_b * A_b + 1j * coef_BAC * A_a * np.conj(A_c)
        F2 += 1j * coef_BAD * A_a * np.conj(A_d)

        F3 = -1j * self.freq_c * A_c + 1j * coef_CAB * A_a * np.conj(A_b)

        F4 = -1j * self.freq_d * A_d + 1j * coef_DAB * A_a * np.conj(A_b)

        return np.array([F1, F2, F3, F4])


'''
RUNGE KUTTA 33
'''


def RK33(Four_Waves, t_0, t_f, h, A_0):

    n = (t_f - t_0)/h
    n = int(n)

    y_0 = A_0

    Y = [y_0]

    for k in range(n):

        k1 = Four_Waves.f(Y[-1])
        k2 = Four_Waves.f(Y[-1] + h/2 * k1)
        k3 = Four_Waves.f(Y[-1] + h/2 * k2)
        k4 = Four_Waves.f(Y[-1] + h * k3)

        Y += [Y[-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)]

    Y = np.array(Y)
    T = np.linspace(t_0, t_f, n+1)

    return Y, T


'''
FOUR-WAVES INTEGRATION
'''

def four_waves_integration(Four_Waves, A_0, t_0, t_f, h):
    
    days = (10**5/14.54 * t_f)/24/60/60
    
    A_a, A_b, A_c, A_d = A_0
    
    Energy_0 = A_a*np.conj(A_a) + A_b * np.conj(A_b)
    Energy_0 += A_c * np.conj(A_c) + A_d * np.conj(A_d)  # quadratic

    Energy_0 += 2*np.real(A_c*A_b*np.conj(A_a))*Four_Waves.S_abc
    Energy_0 += 2*np.real(A_b*A_d*np.conj(A_a))*Four_Waves.S_abd

    Y, T = RK33(Four_Waves, t_0, t_f, h, A_0)

    Y_a = Y[:, 0]
    Y_b = Y[:, 1]
    Y_c = Y[:, 2]
    Y_d = Y[:, 3]
    
    # Energy

    E_3  = 2*np.real(Y_c*Y_b*np.conj(Y_a))*Four_Waves.S_abc
    E_3 += 2*np.real(Y_b*Y_d*np.conj(Y_a))*Four_Waves.S_abd
    E_3 /= Energy_0

    Y_a = Y_a * np.conj(Y_a)/Energy_0  # mode a
    Y_b = Y_b * np.conj(Y_b)/Energy_0  # mode b
    Y_c = Y_c * np.conj(Y_c)/Energy_0  # mode c
    Y_d = Y_d * np.conj(Y_d)/Energy_0  # mode d

    E_2 = Y_a + Y_b + Y_c + Y_d

    E_T = E_2 + E_3
    
    t = np.linspace(0, days, len(T))
   
    plt.plot(t, Y_a, label='mode a')
    plt.plot(t, Y_b, label='mode b')
    plt.plot(t, Y_c, label='mode c')
    plt.plot(t, Y_d, label='mode d')
    plt.plot(t, E_T, label='Total Energy')
    # plt.plot(T, E_2, label = r'$E^{(2)}$')
    # plt.plot(T, E_3, label = r'$E^{(3)}$')
    plt.xlabel('Time (days)')
    plt.ylabel(' Energy (nondimensional)')
    plt.legend()
    plt.title(fr'Maximum efficiency')
    plt.show()
    
'''
EXAMPLES
'''

# CONSTANTS

# eps = 100

g = 9.8
h_e = 10000

a = 6.38e+06
omega = 2*np.pi/24/60/60


eps = (4*a*a*omega*omega)/(g*h_e)
gamma = 1/np.sqrt(eps)

print()
print()
print('-------------')
print(fr'$\epsilon$ = {eps}')
print(fr'$h_e$ = {h_e}m')
print('-------------')
print()
#eps = 8.810572669756384  # equivalent heigh 10km

N = 10
deg = 300

t_0 = 0
t_f = 150  #25.2
h = 0.1


(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
 m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  5, 6, 1, 4, 6, 1, 1, 4, 1, 1, 3, 3 #EEER

(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
 m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  6, 9, 1, 5, 9, 1, 1, 2, 1, 1, 7, 3 #EEER

(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
 m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  7, 9, 1, 6, 9, 1, 1, 8, 1, 1, 7, 3 #EEER 
# EER quasi-resonant

#(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
# m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  5, 5, 1, 3, 3, 1, 2, 4, 1, 2, 3, 3 #EEER

#(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
# m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  4, 6, 1, 3, 7, 1, 1, 5, 1, 1, 2, 3 #EEER

#(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
# m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  8, 9, 1, 6,8, 1, 2, 5, 1, 2, 2, 3 #EEER


Four_Waves = FOUR_WAVES(gamma, m_a, n_a, alpha_a, m_b, n_b, alpha_b,
                        m_c, n_c, alpha_c, m_d, n_d, alpha_d)


'''
GRAPHS - PRECESSION MECHANISM
'''

u_a = norm_component(Four_Waves.uvh_a[0])*np.sqrt(g*h_e)
u_b = norm_component(Four_Waves.uvh_b[0])*np.sqrt(g*h_e)
u_c = norm_component(Four_Waves.uvh_c[0])*np.sqrt(g*h_e)

beta_1 = 1/np.real(u_b)
beta_100 = 100/np.real(u_b)

BETA = np.linspace(beta_1, beta_100, 20)

u_d = norm_component(Four_Waves.uvh_d[0])*np.sqrt(g*h_e)

alpha_1 = 1/np.real(u_a)
alpha_100 = 100/np.real(u_a)

ALPHA = np.linspace(alpha_1, alpha_100, 20)

A,B = np.meshgrid(ALPHA, BETA)

MEASURE = []

A, B = np.meshgrid(ALPHA, BETA)

MEASURE    = []
PRECESSION = []

max_value = 0
p = 0
b = 0

A_d = 27/np.real(u_d)

#ALPHA = np.array([ALPHA[0]])

for beta in BETA:

    A_b = beta
    A_c = 0.1*beta
    
    MEASURE_alpha    = []
    PRECESSION_alpha = []
    
    b = 0

    for alpha in ALPHA:
        
        A_a = alpha

        A_0 = np.array([A_a, A_b, A_c, A_d])

        Energy_0 = A_a*np.conj(A_a) + A_b * np.conj(A_b)
        Energy_0 += A_c * np.conj(A_c) + A_d * np.conj(A_d)  # quadratic

        Energy_0 += 2*np.real(A_c*A_b*np.conj(A_a))*Four_Waves.S_abc
        Energy_0 += 2*np.real(A_b*A_d*np.conj(A_a))*Four_Waves.S_abd

        Y, T = RK33(Four_Waves, t_0, t_f, h, A_0)
        
        Y_a = Y[:, 0]
        Y_b = Y[:, 1]
        Y_d = Y[:, 3]
        
        Y_d = Y_d * np.conj(Y_d)/Energy_0  # mode d
        
        measure = max(Y_d)-min(Y_d)
        MEASURE_alpha    += [ measure ]
        
        
        Phi_a = np.unwrap(np.angle(Y_a))
        Phi_b = np.unwrap(np.angle(Y_b))
        Phi_d = np.unwrap(np.angle(Y_d))
        
        PHI_abd = Phi_a - Phi_b - Phi_d
        precession = (PHI_abd[-1] - PHI_abd[0])/t_f
        #precession -= Four_Waves.mismatch_2
        PRECESSION_alpha += [precession]
        
        if measure > max_value:
            max_value = measure
            A_max = np.array([A_a, A_b, A_c, A_d])
        if abs(precession) > p:
            p = abs(precession)
            Phi_am = Phi_a
            Phi_bm = Phi_b
            Phi_dm = Phi_d         
        
        if b == 100:
            tt = np.arange(t_0,t_f,h )
          
            plt.plot(T, PHI_abd)
            b = 1
    
    
    MEASURE += [MEASURE_alpha]
    PRECESSION += [PRECESSION_alpha]
#plt.show()
MEASURE = np.array(MEASURE)
PRECESSION = np.array(PRECESSION)

'''
tt = T
plt.plot(tt, Phi_am, label = 'mode a')
plt.plot(tt, Phi_bm, label = 'mode b')
plt.plot(tt, Phi_dm, label = 'mode d')
plt.plot(tt, Phi_am - Phi_bm - Phi_dm, label = r'$\Phi$')
plt.legend()
plt.show()
'''


max_a = norm_component(Four_Waves.uvh_a[0])*A_max[0]*np.sqrt(g*h_e)
max_b = norm_component(Four_Waves.uvh_b[0])*A_max[1]*np.sqrt(g*h_e)
max_c = norm_component(Four_Waves.uvh_c[0])*A_max[2]*np.sqrt(g*h_e)
max_d = norm_component(Four_Waves.uvh_d[0])*A_max[3]*np.sqrt(g*h_e)

print('------------')
print(f' Rossby variation      = {100*np.real(max_value)}%')
print(f' Mode a zonal velocity = {np.real(max_a)}m/s')
print(f' Mode b zonal velocity = {np.real(max_b)}m/s')
print(f' Mode c zonal velocity = {np.real(max_c)}m/s')
print(f' Rossby zonal velocity = {np.real(max_d)}m/s')
print('------------')

levels = np.linspace(0,10,11)

fig, ax = plt.subplots()

cs = ax.contourf(A*u_a,B*u_b, 100*MEASURE, levels=levels, cmap = 'terrain')
plt.xlabel(r'Rossby zonal velocity (m/s)')
plt.ylabel(r'Pump zonal velocity (m/s)')
plt.title(r'Rossby Waves Modulation (%)')
plt.xlim(1,100)
#RdYlBu
#terrain
#gist_ncar
fig.colorbar(cs)
plt.show()   

t_f = 25.2
#four_waves_integration(Four_Waves, A_max, t_0, t_f, h=0.01)


# PRESSETION RESONANCE GRAPH
levels = np.linspace(-0.7,0.4,31)

fig, ax = plt.subplots()

cs = ax.contourf(A*u_a,B*u_b, (PRECESSION), levels=levels, cmap = 'terrain')
plt.xlabel(r'Rossby zonal velocity (m/s)')
plt.ylabel(r'Pump zonal velocity (m/s)')
plt.title(r'Rossby Waves Modulation (%)')
plt.xlim(1,100)
#RdYlBu
#terrain
#gist_ncar
fig.colorbar(cs)
plt.show()   

print()
print()
print('mode a = ', Four_Waves.mode_a)
print('freq a = ', Four_Waves.freq_a)
print('coef a_1 = ', Four_Waves.coef_ABC)
print('coef a_2 = ', Four_Waves.coef_ABD)
print()
print('mode b = ', Four_Waves.mode_b)
print('freq b = ', Four_Waves.freq_b)
print('coef b_1 = ', Four_Waves.coef_BAC)
print('coef b_2 = ', Four_Waves.coef_BAD)
print()
print('mode c = ', Four_Waves.mode_c)
print('freq c = ', Four_Waves.freq_c)
print('coef c = ', Four_Waves.coef_CAB)
print()
print('mode d = ', Four_Waves.mode_d)
print('freq d = ', Four_Waves.freq_d)
print('coef d = ', Four_Waves.coef_DAB)
print()
print('mismatch_1 = ',Four_Waves.mismatch_1)
print('mismatch_2 = ',Four_Waves.mismatch_2)
print()
print(Four_Waves.freq_a - Four_Waves.freq_b)

