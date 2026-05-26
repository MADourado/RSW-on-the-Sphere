import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.colors import ListedColormap

from Hough_Harmonics.Normalization import norm_Hough
from Hough_Harmonics.Normalization import norm_component
from Hough_Harmonics.Eigenvalues_and_eigenvectors.Eigenvectors import symetry
from Hough_Harmonics.innerproduct import inner_product
from Hough_Harmonics.innerproduct import S_abc

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
    
'''
RUNGE KUTTA 33
'''
def RK33(Triad, t_0, t_f, h, A_0 ):
    
    n = (t_f - t_0)/h
    n = int(n)
    
    y_0 = A_0
    
    Y = [y_0]
    
    for k in range(n):
        
        k1 = Triad.f( Y[-1])
        k2 = Triad.f( Y[-1] + h/2 * k1)
        k3 = Triad.f( Y[-1] + h/2 * k2)
        k4 = Triad.f( Y[-1] + h * k3)
        
        Y += [ Y[-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)]
        
    Y = np.array(Y)
    T = np.linspace(t_0, t_f, n+1)
    
    return Y, T  


def Energy_0(Triad, A_0):
    
    A_a, A_b, A_c = A_0
    
    Energy_02  = (A_a * np.conj(A_a) + A_b * np.conj(A_b) + A_c * np.conj(A_c) )
    Energy_03 = (2*np.real(np.conj(A_a) * np.conj(A_b) * A_c) * Triad.Sabc)
    
    return  Energy_02, Energy_03    

def Triad_dynamics(Triad, A_0, t_0, t_f, h, path = None):
    
    E_02, E_03 = Energy_0(Triad, A_0)
    E_0 = E_02 + E_03
    
    Y, T = RK33(Triad, t_0, t_f, h, A_0)
    
    Y_a = Y[:,0] # mode a
    Y_b = Y[:,1] # mode b
    Y_c = Y[:,2] # mode c
    
    Z = np.array([Y_a, Y_b, Y_c])
    E_2, E_3 = Energy_0(Triad, Z)
    E_T =  (E_2 + E_3) / E_0
    
    t = np.linspace(0,t_f/(4*np.pi), len(T))

    E_a = Y_a * np.conj(Y_a)/E_0
    E_b = Y_b * np.conj(Y_b)/E_0
    E_c = Y_c * np.conj(Y_c)/E_0
    
    ea = np.real(max(E_a) - min(E_a))
    eb = np.real(max(E_b) - min(E_b))
    ec = np.real(max(E_c) - min(E_c))
    
    if path:
        #period_Fourier(Y_a, T, h)
        print()
        print('--------EFFICIENCY--------------')
        print(f'Efficiency {Triad.label_a}: {np.round(100*np.real(max(E_a) - min(E_a)),2)}%')
        print(f'Efficiency {Triad.label_b}: {np.round(100*np.real(max(E_b) - min(E_b)),2)}%')
        print(f'Efficiency {Triad.label_c}: {np.round(100*np.real(max(E_c) - min(E_c)),2)}%')
        print()
        print('---------ENERGIES----------------')
        print('Cubic (E_3): ', np.real(E_03))
        print('Quadratic (E_2): ', np.real(E_02))
        print(f'max E_T - min E_T: {np.real(max(E_T) - min(E_T))*100}')
        
        plt.plot(t, E_a, label = Triad.label_a)
        plt.plot(t, E_b, label = Triad.label_b)
        plt.plot(t, E_c, label = Triad.label_c)
        plt.plot(t, E_T, label = 'Total Energy')
        plt.xlabel('Time (days)')
        plt.ylabel(' Energy (nondimensional)')
        plt.ylim(-0.05, 1.05)
        plt.legend (loc='upper right')
        plt.savefig(path)
        plt.close()
    
    return ea, eb, ec, E_a, E_b, E_c

def Triad_Precession(Triad, t_0, t_f, h,g=9.8,h_e=10000, vel_a = 10, pri = True):
    
    u_a = norm_component(Triad.uvh_a[0])*np.sqrt(g*h_e)
    u_b = norm_component(Triad.uvh_b[0])*np.sqrt(g*h_e)
    u_c = norm_component(Triad.uvh_c[0])*np.sqrt(g*h_e)

    alpha_1 = 1/np.real(u_b)
    alpha_100 = 50/np.real(u_b)  # velocities 1m/s --- 100m/s  mode b

    ALPHA = np.linspace(alpha_1, alpha_100, 10)

    beta_1 = 1/np.real(u_a)
    beta_100 = 50/np.real(u_a)   # velocities 1m/s --- 100m/s  mode c

    BETA = np.linspace(beta_1, beta_100, 10)

    A, B = np.meshgrid(ALPHA, BETA)

    MEASURE = []

    max_value = 0

    A_c = vel_a/np.real(u_c)
    i = 0
    for beta in BETA:
        i+= 1
        print(f'linha {i}/20')
        
        A_a = beta
        
        MEASURE_alpha = []
        
        for alpha in ALPHA:
            
            A_b = alpha
            
            A_0 = np.array([A_a, A_b, A_c])
            E_02, E_03 = Energy_0(Triad, A_0)
            E_0 = E_02 + E_03
            
            Y, T = RK33(Triad, t_0, t_f, h, A_0)
                                       
            Y_a = Y[:,0] # mode a
            Y_b = Y[:,1] # mode b
            Y_c = Y[:,2] # mode c
            
            Z = np.array([Y_a, Y_b, Y_c])
            E_2, E_3 = Energy_0(Triad, Z)
            
            Y_c = Y_c * np.conj(Y_c)/E_0
            
            measure = max(Y_c) - min(Y_c)
            MEASURE_alpha += [max(Y_c) - min(Y_c)]
            
            if measure > max_value:
                max_value = measure
                A_max = np.array([A_a, A_b, A_c])

        MEASURE += [MEASURE_alpha]
       
    MEASURE = np.array(MEASURE)
    #print('E_0 = ', E_0)
    print('-----------------------')
    print()
    print(f'Efficiency = {np.real(max_value*100)}%')
    print(f'Mode a = {2*A_max[0]*np.real(u_a)} m/s')
    print(f'Mode b = {2*A_max[1]*np.real(u_b)} m/s')
    print(f'Mode c = {2*A_max[2]*np.real(u_c)} m/s')
    print()
    print('-----------------------')
    print(A_max)
    #levels = np.linspace(0,1,100)

    fig, ax = plt.subplots()

    cs = ax.contourf(2*A*u_b, 2*B*u_a, 100*MEASURE, levels=100, cmap = 'terrain')
    plt.xlabel(f'{Triad.label_b} - zonal velocity (m/s)')
    plt.ylabel(f'{Triad.label_a} - zonal velocity (m/s)')
    plt.title(r'Efficiency of Energy Transfer (%)')
    #plt.xlim(1,100)
    #RdYlBu
    #terrain
    #gist_ncar
    fig.colorbar(cs)
    plt.show()   
    
    Triad_dynamics(Triad, A_max, t_0, t_f = 100, h = 0.001)
    
    if pri:
        print()
        print('------------------------------------------------')
        print('mode a = ', Triad.mode_a)
        print('freq a = ', Triad.freq_a)
        print('coef a = ', Triad.coef_ABC)
        m,n,alpha = Triad.mode_a
        print('symmetric = ', symetry(m, n, alpha))
        print()
        print('mode b = ', Triad.mode_b)
        print('freq b = ', Triad.freq_b)
        print('coef b = ', Triad.coef_BAC)
        m,n,alpha = Triad.mode_b
        print('symmetric = ', symetry(m, n, alpha))
        print()
        print('mode c = ', Triad.mode_c)
        print('freq c = ', Triad.freq_c)
        print('coef c = ', Triad.coef_CAB)
        m,n,alpha = Triad.mode_c
        print('symmetric = ', symetry(m, n, alpha))
        print()
        print('mismatch = ',Triad.freq_a + Triad.freq_b - Triad.freq_c)
        print('S_abc = ', Triad.Sabc)
        print('------------------------------------------------')

def eff_tri(T, A_0, u_a):
    
    t_0, t_f, h = 0, 300, 0.1
    
    A_a, A_b, A_c = A_0
    
    a1 = 0/u_a
    a100 = 50/u_a
    
    Ea = []
    Eb = []
    Ec = []
    ALPHA = np.linspace(a1, a100, 51)
    for alpha in ALPHA:
        
        ea, eb, ec,_,_,_ = Triad_dynamics(T, np.array([alpha, A_b, A_c]), 
                                          t_0, t_f, h, p = False)
        
        Ea += [ea]
        Eb += [eb]
        Ec += [ec]
    Ea = np.array(Ea)
    Eb = np.array(Eb)
    Ec = np.array(Ec)
    
    plt.plot(ALPHA*u_a*2, Ea*100, label = f'{T.label_a}')
    plt.plot(ALPHA*u_a*2, Eb*100, label = f'{T.label_b}')
    plt.plot(ALPHA*u_a*2, Ec*100, label = f'{T.label_c}')
    plt.scatter(20, Ec[10]*100, color = 'red')
    plt.scatter(60, Eb[30]*100, color = 'blue')
    #plt.scatter(60, Ec[30]*100, color = 'b')
    #plt.scatter(60, Eb[30]*100, color = 'b')
    plt.annotate(f'Example 2', (20, Ec[10]*100), textcoords="offset points", 
                 xytext=(30, 10), ha='center')
    plt.annotate(f'Example 3', (60, Ec[30]*100), textcoords="offset points", 
                 xytext=(30, 10), ha='center')
    plt.xlabel(f'Meridional Velocity - {T.label_a}')
    plt.ylabel(f'Efficiency (%)')
    plt.legend()
    plt.show()

def period_Fourier(Amp, t, h):
    
    fft_A = np.fft.fft(Amp)
    freq  = np.fft.fftfreq(len(t), d=h)
    
    fft_A = np.abs(fft_A)
    
    dom = freq[np.argmax(fft_A)]
    print()
    print('Dominant period = ', 1/dom/(4*np.pi))
    
    plt.plot(1/freq/(4*np.pi), fft_A)
    plt.xlim(left=0)
    plt.show()