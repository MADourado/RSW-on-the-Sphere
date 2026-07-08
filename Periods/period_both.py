import scipy
import numpy as np
import matplotlib.pyplot as plt
from Dynamic_Triads import TRIAD, RK33
from Hough_Harmonics.Normalization import norm_component

from period import PERIOD as p_bus
from period_Harris import PERIOD as p_har

'''
TRIADS
'''   

# CONSTANTS

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
h   = 0.01

#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,2,2, 2,3,2, 3,5,2  # WWW
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,2,3, 1,11,3, 2,8,3 # RRR
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 5,11,3, 7,15,3, 12,13,3 # resonant
m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,3,3, 3,7,3, 4,5,3 # RRR 
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 2,2,3, 6,8,1, 8,9,1 # 

##m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,6,3, 2,4,3, 3,5,3 # REE
m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 3,11,3, 12,15,1, 15,15,1 # REE
m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 3,11,3, 12,15,1, 15,15,1 
m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 3,10,3, 1,2,3, 4,5,3 

Triad = TRIAD(gamma, m_a, n_a, alpha_a, m_b, n_b, 
              alpha_b, m_c,n_c, alpha_c, N, deg)   

Triad.mismatch *= -1
'''
Triad.freq_a *= -1
Triad.freq_b *= -1
Triad.freq_c *= -1
Triad.coef_ABC *= -1
Triad.coef_BAC *= -1
Triad.coef_CAB *= -1
'''
print(Triad.coef_ABC)
print(Triad.coef_BAC)
print(Triad.coef_CAB)



u_a = norm_component(Triad.uvh_a[0])*np.sqrt(g*h_e)
u_b = norm_component(Triad.uvh_b[0])*np.sqrt(g*h_e)
u_c = norm_component(Triad.uvh_c[0])*np.sqrt(g*h_e)

A_a = 25/u_a
A_b = 25/u_b
A_c = 25/u_c
    
A_max = np.array([A_a, A_b, A_c])
#A_max = np.array([0.00463481, 0.0464163,  0.61211021])
print()
print('A_max = ', A_max)
#A_max = np.array([0.00463481, 0.0464163,  0.61211021])
A_a, A_b, A_c = A_max

A_0 = np.array([1j*A_a*np.sqrt(Triad.coef_BAC*Triad.coef_CAB), 
                  1j*A_b*np.sqrt(Triad.coef_ABC*Triad.coef_CAB), 
                  1j*A_c*np.sqrt(Triad.coef_ABC*Triad.coef_BAC)])


#A_max = A_0
tt = p_bus(Triad, -A_0)
t2 = p_har(Triad, A_0)

t_f = 2*max(tt,t2)

#tt = (10**5/14.54 * tt)/24/60/60
tt = tt/(4*np.pi)
t2 = t2/(4*np.pi)

print()
print('----------------')
print('T_bus = ', tt)
print('T_har = ', t2)
print('Mismatch = ', Triad.mismatch)
print('----------------')
print()

Energy_0  = (A_a * np.conj(A_a) + A_b * np.conj(A_b) + A_c * np.conj(A_c) )
print(Energy_0)
Energy_0 += (2*np.real(np.conj(A_a) * np.conj(A_b) * A_c) * Triad.Sabc)

Y, T = RK33(Triad, t_0, t_f, h, A_max)  

Y_a = Y[:,0]
Y_b = Y[:,1]
Y_c = Y[:,2]
print()
print((abs(Y_a)**2)[:4])
print((abs(Y_b)**2)[:4])
print((abs(Y_c)**2)[:4])
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

plt.plot(t, Y_a, label = f'{Triad.label_a}')
plt.plot(t, Y_b, label = f'{Triad.label_b}')
plt.plot(t, Y_c, label = f'{Triad.label_c}')
plt.plot(t, E_T, label = 'Total Energy')
#plt.plot(T, E_2, label = r'$E^{(2)}$')
#plt.plot(T, E_3, label = r'$E^{(3)}$')
plt.axvline(x = float(tt), color='blue', lw='1', linestyle='--')
plt.text(float(tt) + 1, 0.9, "K-B", color='blue', fontsize=10)
plt.axvline(x = float(t2), color='red', lw='1', linestyle='--')
plt.text(float(t2) + 1, 0.8, "Harris", color='red', fontsize=10)
plt.xlabel('Time (days)')
plt.ylabel(' Energy (nondimensional)')
plt. legend (loc='upper right')
#plt.legend()
#plt.title('Triad - temporal integration')
plt.show()
