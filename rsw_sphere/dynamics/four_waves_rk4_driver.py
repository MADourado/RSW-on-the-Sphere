from rsw_sphere.dynamics.four_waves_pump import FOUR_WAVES
import numpy as np
import matplotlib.pyplot as plt
from rsw_sphere.hough_harmonics.normalization import norm_component


def FF(Four, A_0,t):
    
    A_a, A_b, A_c, A_d = A_0
    
    f1  = 1j*Four.coef_ABC * A_b*A_c*np.exp(-1j*t*Four.mismatch_1) 
    f1 += 1j*Four.coef_ABD * A_b*A_d*np.exp(-1j*t*Four.mismatch_2)
    
    f2 = 1j*Four.coef_BAC * A_a*np.conj(A_c)*np.exp(1j*t*Four.mismatch_1)
    f2 += 1j*Four.coef_BAD * A_a*np.conj(A_d)*np.exp(1j*t*Four.mismatch_2)
    
    f3 = 1j*Four.coef_CAB * A_a*np.conj(A_b)*np.exp(1j*t*Four.mismatch_1)

    f4 = 1j*Four.coef_DAB *A_a*np.conj(A_b)*np.exp(1j*t*Four.mismatch_2)
    
    return np.array([f1,f2,f3,f4])

def RK44(FF, Four, t_0, t_f, h, A_0):

    n = (t_f - t_0)/h
    n = int(n)

    y_0 = A_0

    Y = [y_0]
    t_1 = t_0
    for k in range(n):
        
        k1 = FF(Four, Y[-1], t_1)
        k2 = FF(Four, Y[-1] + h/2 * k1, t_1 + h/2)
        k3 = FF(Four, Y[-1] + h/2 * k2, t_1 + h/2)
        k4 = FF(Four, Y[-1] + h * k3, t_1 + h)

        Y += [Y[-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)]
        
        t_1 += h
    Y = np.array(Y)
    T = np.linspace(t_0, t_f, n+1)
    return Y, T


def four_integration(Four_Waves,FF, A_0, t_0, t_f, h, plot = True):
    
    days = (10**5/14.54 * t_f)/24/60/60
    
    A_a, A_b, A_c, A_d = A_0
    
    Energy_0  = A_a * np.conj(A_a) + A_b * np.conj(A_b)
    Energy_0 += A_c * np.conj(A_c) + A_d * np.conj(A_d)  # quadratic
   
    Energy_0 += 2*np.real(A_c*A_b*np.conj(A_a))*Four_Waves.S_abc
    Energy_0 += 2*np.real(A_b*A_d*np.conj(A_a))*Four_Waves.S_abd
    
    Y, T = RK44(FF,Four_Waves, t_0, t_f, h, A_0)
 
    Y_a = Y[:, 0]
    Y_b = Y[:, 1]
    Y_c = Y[:, 2]
    Y_d = Y[:, 3]

    E_3  = 2*np.real(Y_c*Y_b*np.conj(Y_a)*np.exp(-1j*T*Four_Waves.mismatch_1))*Four_Waves.S_abc
    E_3 += 2*np.real(Y_b*Y_d*np.conj(Y_a)*np.exp(-1j*T*Four_Waves.mismatch_2))*Four_Waves.S_abd
    E_3 /= Energy_0
        
    Y_a = Y_a * np.conj(Y_a)/Energy_0  # mode a
    Y_b = Y_b * np.conj(Y_b)/Energy_0  # mode b
    Y_c = Y_c * np.conj(Y_c)/Energy_0  # mode c
    Y_d = Y_d * np.conj(Y_d)/Energy_0  # mode d
    
    e = np.real(max(Y_d) - min(Y_d))
    
    #print('Efficiency = ', np.real(max(Y_d) - min(Y_d)))
    #print('--------------------------------------')
    E_2 = Y_a + Y_b + Y_c + Y_d
    print('E_2 = ', E_2[:10])
    E_T = E_2 + E_3
    
    t = np.linspace(0, days, len(T))
    
    if plot:
        
        #tt = PERIOD(Triad_b, Amp_change(Triad_b, A_g)) / (4*np.pi)
        plt.plot(t, Y_a, label= Four_Waves.label_a)
        plt.plot(t, Y_b, label= Four_Waves.label_b)
        plt.plot(t, Y_c, label= Four_Waves.label_c)
        plt.plot(t, Y_d, label= Four_Waves.label_d)
        plt.plot(t, E_T, label='Total Energy')
        #plt.plot(t, E_2, label = r'$E^{(2)}$')
        #plt.plot(t, E_3, label = r'$E^{(3)}$')
        #plt.axvline(x = float(tt), color='red', lw='1', linestyle='--')
        plt.xlabel('Time (days)')
        plt.ylabel(' Energy (nondimensional)')
        plt.legend()
        #plt.title(fr'Maximum efficiency')
        plt.show()
        print()
        print('Efficiency - mode a: ', 100*np.real(max(Y_a) - min(Y_a)))
        print('Efficiency - mode b: ', 100*np.real(max(Y_b) - min(Y_b)))
        print('Efficiency - mode c: ', 100*np.real(max(Y_c) - min(Y_c)))
        print('Efficiency - mode d: ', 100*np.real(max(Y_d) - min(Y_d)))
        print()
      
    

    

    
'''
EXAMPLES
'''

# CONSTANTS

# eps = 100

g = 9.8
h_e = 10000

a = 6.38e+06
omega = 2*np.pi/24/60/60


eps = 8.810572669756384#(4*a*a*omega*omega)/(g*h_e)
gamma = 1/np.sqrt(eps)

print()
print()
print('-------------')
print(fr'$\epsilon$ = {eps}')
print(fr'$h_e$ = {h_e}m')
print('-------------')
print()
#eps = 8.810572669756384  # equivalent heigh 10km

N = 30
deg = 300

t_0 = 0
t_f = 1000 #25.2
h = 0.1


(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
 m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  4, 5, 3, 1, 2, 3, 3, 10, 3, 3, 6, 3 #EEER
# exemplo legal de como a terceira onda, mesmo com pouca energia intefere na dinamica
# das outras tres, reduzindo a eficiencia da quarta - de 65 para 25%
#(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
# m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  4, 4, 1, 2, 4, 1, 2, 3, 3, 2, 7, 3 #EEER
#(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
# m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  4, 4, 1, 2, 4, 1, 2, 5, 3, 2, 3, 3 #EEER


(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
 m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  4, 5, 3, 3, 4, 3, 1, 1, 1, 1, 2 ,3 #EEER
#(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
# m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  5, 5, 1, 3, 3, 1, 2, 2, 1, 2, 3 ,3 #EEER

Four_Waves = FOUR_WAVES(gamma, m_a, n_a, alpha_a, m_b, n_b, alpha_b,
                        m_c, n_c, alpha_c, m_d, n_d, alpha_d)

#Triad_a = TRIAD(gamma, m_c, n_c, alpha_c, m_b, n_b, alpha_b, m_a, n_a, alpha_a)
#Triad_b = TRIAD(gamma, m_d, n_d, alpha_d, m_b, n_b, alpha_b, m_a, n_a, alpha_a)

print('------------------')
print('Coef_A = ', Four_Waves.coef_ABC)
print('Coef_B = ', Four_Waves.coef_BAC)
print('Coef_C = ', Four_Waves.coef_CAB)
print()
print('Coef_A = ', Four_Waves.coef_ABD)
print('Coef_B = ', Four_Waves.coef_BAD)
print('Coef_D = ', Four_Waves.coef_DAB)
print()
#print('CoefA1 = ', Triad_a.coef_ABC)
#print('CoefA2 = ', Triad_b.coef_ABC)



u_a = norm_component(Four_Waves.uvh_a[0])*np.sqrt(g*h_e)
u_b = norm_component(Four_Waves.uvh_b[0])*np.sqrt(g*h_e)
u_c = norm_component(Four_Waves.uvh_c[0])*np.sqrt(g*h_e)
u_d = norm_component(Four_Waves.uvh_d[0])*np.sqrt(g*h_e)

A_a = 1/u_a
A_b = 1/u_b
A_c = 1.0/u_c
A_d = 1.0/u_d

d_int = 10
e_max = 0


k = 40
A_0 = np.array([A_a*30, A_b*30, A_c*20, A_d *10])

print()
print('----------------------')
period_a = 20#PERIOD(Triad_a , Amp_change(Triad_a,  np.array([A_0[2], A_0[1], A_0[0]])))
#Triad_dynamics(Triad_a, np.array([A_0[2], A_0[1], A_0[0]]), t_0, t_f = 2*period_a, h = 0.1,
               #p = 'True')
print('Period = ', 
      period_a/(4*np.pi))
print()
print('-----------------------')
period_b = 20# PERIOD(Triad_b , Amp_change(Triad_b, np.array([A_0[3], A_0[1], A_0[0]])))
#Triad_dynamics(Triad_b, np.array([A_0[3], A_0[1], A_0[0]]), t_0, t_f = 2*period_b, h = 0.1,
             #  p = 'True')
print('Period = ', 
      period_b/(4*np.pi))

Triad_a = 1
four_integration(Four_Waves,FF, A_0, t_0, t_f=4*max(period_a, period_b), h=0.001)





















