import scipy
import numpy as np
import matplotlib.pyplot as plt
from rsw_sphere.dynamics.dynamic_triads import TRIAD, RK33
from rsw_sphere.hough_harmonics.normalization import norm_component

def Hamiltonian(A_0):
    
    A_a, A_b, A_c = A_0
    
    B_a = abs(A_a)
    B_b = abs(A_b)
    B_c = abs(A_c)
    Phi = np.angle(A_c) - np.angle(A_a) - np.angle(A_b)
    
    return B_a * B_b * B_c * np.sin(Phi)

def rho(Triad, A_0):
    
    A_a, A_b, A_c = A_0
    
    I_23  = abs(A_b)**2 #* Triad.coef_ABC * Triad.coef_CAB
    I_23 += abs(A_c)**2 #* Triad.coef_ABC * Triad.coef_BAC
    
    I_13  = abs(A_a)**2 #* Triad.coef_BAC * Triad.coef_CAB
    I_13 += abs(A_c)**2 #* Triad.coef_ABC * Triad.coef_BAC
    
    return I_23/I_13

def nu(Triad, A_0):
    
    A_a, A_b, A_c = A_0
    
    r = rho(Triad, A_0)
    H = Hamiltonian(A_0)
    
    I_13  = abs(A_a)**2 #* Triad.coef_BAC * Triad.coef_CAB
    I_13 += abs(A_c)**2 #* Triad.coef_ABC * Triad.coef_BAC
    
    I = I_13**3
    
    value  = (-2 +3*r + 3*r*r - 2*r*r*r)*I - 27*H*H
    value /= (2 * np.sqrt((1 - r + r*r)**3) * I)
    
    return np.arccos(value)

def mu(Triad, A_0):
    
    n = nu(Triad, A_0)
    
    value  = np.cos(n/3 + np.pi/6)
    value /= np.cos(n/3 - np.pi/6)
    
    return value

def PERIOD(Triad, A_0):
    
    A_a, A_b, A_c = A_0
    
    m = np.real(mu(Triad, A_0))
    print()
    print()
    print('mu = ',m)
    r = rho(Triad, A_0)
    print('rho = ', r)
    n = nu(Triad, A_0)
    print('nu = ', n)
    
    K   = scipy.special.ellipk(m)
    print('K = ', K)
    cos = np.cos(n/3 - np.pi/6)
    
    I_13  = abs(A_a)**2 #* Triad.coef_BAC * Triad.coef_CAB
    I_13 += abs(A_c)**2 #* Triad.coef_ABC * Triad.coef_BAC
    
    period  = 3**(1/4) * 2**(1/2) * K
    period /= ( (1 - r + r*r)**(1/4) * np.sqrt(cos) * np.sqrt(I_13))
    
    return period

'''
TRIADS
'''   

# CONSTANTS
'''
g = 9.8
h_e = 10000

a = 6.38e+06
omega = 2*np.pi/24/60/60
eps = (4*a*a*omega*omega)/(g*h_e)
#eps = 8.810572669756384
gamma = 1/np.sqrt(eps)

#eps = 1000
gamma = 1/np.sqrt(eps)
N = 10
deg = 300

t_0 = 0
t_f = 100#25.2 
h   = 0.001

m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,2,2, 2,3,2, 3,5,2  # WWW
m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,2,3, 1,11,3, 2,8,3 # RRR
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 5,11,3, 7,15,3, 12,13,3 # resonant
m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,3,3, 3,7,3, 4,5,3 # RRR 
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 2,2,3, 6,8,1, 8,9,1 # 

#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,7,3, 6,9,1, 7,9,1 # REE
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 3,11,3, 12,15,1, 15,15,1 # REE

Triad = TRIAD(gamma, m_a, n_a, alpha_a, m_b, n_b, 
             alpha_b, m_c,n_c, alpha_c, N, deg)   


u_a = norm_component(Triad.uvh_a[0])*np.sqrt(g*h_e)
u_b = norm_component(Triad.uvh_b[0])*np.sqrt(g*h_e)
u_c = norm_component(Triad.uvh_c[0])*np.sqrt(g*h_e)

A_a = 70/u_a
A_b = 70/u_b
A_c = 70/u_c
    
A_max = np.array([A_a, A_b, A_c])

A_0 = np.array([A_a*np.sqrt(Triad.coef_BAC*Triad.coef_CAB), 
                  A_b*np.sqrt(Triad.coef_ABC*Triad.coef_CAB), 
                  A_c*np.sqrt(Triad.coef_ABC*Triad.coef_BAC)])

#A_max = A_0
tt = PERIOD(Triad, 1j*A_0)

#tt = (10**5/14.54 * tt)/24/60/60
tt = tt/(4*np.pi)

print()
print('----------------')
print('T = ', tt)
print('----------------')
print()

Energy_0  = (A_a * np.conj(A_a) + A_b * np.conj(A_b) + A_c * np.conj(A_c) )
Energy_0 += (2*np.real(np.conj(A_a) * np.conj(A_b) * A_c) * Triad.Sabc)

Y, T = RK33(Triad, t_0, t_f, h, A_max)  

Y_a = Y[:,0]
Y_b = Y[:,1]
Y_c = Y[:,2]

Z_a = np.copy(Y[:,0])
Z_b = np.copy(Y[:,1])
Z_c = np.copy(Y[:,2])


# ENERGY

E_3 = (2*np.real(np.conj(Y_a) * np.conj(Y_b) * Y_c) * Triad.Sabc)/Energy_0 # cubic energy

Y_a = Y_a * np.conj(Y_a)/Energy_0  # mode a
Y_b = Y_b * np.conj(Y_b)/Energy_0  # mode b
Y_c = Y_c * np.conj(Y_c)/Energy_0  # mode c

E_2 = Y_a + Y_b + Y_c              # quadratic energy
#E_3 = (2*np.real(np.conj(Y_a) * np.conj(Y_b) * Y_c) * Triad.Sabc)/Energy_0 # cubic energy

E_T = E_2 + E_3  # total energy


days = (10**5/14.54 * t_f)/24/60/60
t = np.linspace(0,days, len(T))

plt.plot(t, Y_a, label = 'mode a')
plt.plot(t, Y_b, label = 'mode b')
plt.plot(t, Y_c, label = 'mode c')
plt.plot(t, E_T, label = 'Total Energy')
#plt.plot(T, E_2, label = r'$E^{(2)}$')
#plt.plot(T, E_3, label = r'$E^{(3)}$')
plt.axvline(x = float(tt), color='grey', lw='1', linestyle='--')
plt.xlabel('Time (days)')
plt.ylabel(' Energy (nondimensional)')
plt.legend()
#plt.title('Triad - temporal integration')
plt.show()

plt.plot(t, abs(Z_a), label='mode a')
plt.plot(t, abs(Z_b), label='mode b')
plt.plot(t, abs(Z_c), label='mode c')
plt.axvline(x = float(tt), color='grey', lw='1', linestyle='--')
plt.legend()
plt.show()


print('mode a = ', Triad.mode_a)
print('freq a = ', Triad.freq_a)
print('coef a = ', Triad.coef_ABC)
print()
print('mode b = ', Triad.mode_b)
print('freq b = ', Triad.freq_b)
print('coef b = ', Triad.coef_BAC)
print()
print('mode c = ', Triad.mode_c)
print('freq c = ', Triad.freq_c)
print('coef c = ', Triad.coef_CAB)
print()
print('mismatch = ',Triad.freq_a + Triad.freq_b - Triad.freq_c)
print('S_abc = ', Triad.Sabc)
print(Triad.freq_c - Triad.freq_b)
print()
'''