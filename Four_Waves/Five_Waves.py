import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from Hough_Harmonics.Normalization import norm_Hough
from Hough_Harmonics.Normalization import norm_component
from Hough_Harmonics.Eigenvalues_and_eigenvectors.Eigenvectors import symetry
from Hough_Harmonics.innerproduct import inner_product
from Hough_Harmonics.innerproduct import S_abc
from Dynamic_Triads import TRIAD
from Dynamic_Triads import Triad_dynamics

class FIVE_WAVES:

    def __init__(self, gamma, m_a, n_a, alpha_a, m_b, n_b,
                 alpha_b, m_c, n_c, alpha_c, m_d, n_d, alpha_d,m_e, n_e, alpha_e,
                 N=10, deg=300):

        self.mode_a = np.array([m_a, n_a, alpha_a])
        self.mode_b = np.array([m_b, n_b, alpha_b])
        self.mode_c = np.array([m_c, n_c, alpha_c])
        self.mode_d = np.array([m_d, n_d, alpha_d])
        self.mode_e = np.array([m_e, n_e, alpha_e])

        A = norm_Hough(m_a, n_a, alpha_a, gamma, N, deg)
        eigen_a = A[-1]
        A = A[:-3]

        self.uvh_a = A
        label_a = f'({m_a}, {n_a},'
        if alpha_a == 1:
            label_a += ' EG)'
        elif alpha_a == 2:
            label_a += ' WG)'
        else:
            label_a += ' RH)'
        self.label_a = label_a

        B = norm_Hough(m_b, n_b, alpha_b, gamma, N, deg)
        eigen_b = B[-1]
        B = B[:-3]

        self.uvh_b = B
        label_b = f'({m_b}, {n_b},'
        if alpha_b == 1:
            label_b += ' EG)'
        elif alpha_b == 2:
            label_b += ' WG)'
        else:
            label_b += ' RH)'
        self.label_b = label_b

        C = norm_Hough(m_c, n_c, alpha_c, gamma, N, deg)
        eigen_c = C[-1]
        C = C[:-3]
        
        self.uvh_c = C
        label_c = f'({m_c}, {n_c},'
        if alpha_c == 1:
            label_c += ' EG)'
        elif alpha_c == 2:
            label_c += ' WG)'
        else:
            label_c += ' RH)'
        self.label_c = label_c

        D = norm_Hough(m_d, n_d, alpha_d, gamma, N, deg)
        eigen_d = D[-1]
        D = D[:-3]

        self.uvh_d = D
        label_d = f'({m_d}, {n_d},'
        if alpha_d == 1:
            label_d += ' EG)'
        elif alpha_d == 2:
            label_d += ' WG)'
        else:
            label_d += ' RH)'
        self.label_d = label_d
        
        E = norm_Hough(m_e, n_e, alpha_e, gamma, N, deg)
        eigen_e = E[-1]
        E = E[:-3]

        self.uvh_e = E
        label_e = f'({m_e}, {n_e},'
        if alpha_e == 1:
            label_e += ' EG)'
        elif alpha_e == 2:
            label_e += ' WG)'
        else:
            label_e += ' RH)'
        self.label_e = label_e

        self.freq_a = eigen_a
        self.freq_b = eigen_b
        self.freq_c = eigen_c
        self.freq_d = eigen_d
        self.freq_e = eigen_e

        inner_ABC = inner_product(A, m_a, B, m_b, C, m_c, deg, False)
        inner_ABD = inner_product(A, m_a, B, m_b, D, m_d, deg, False)
        inner_ABE = inner_product(A, m_a, B, m_b, E, m_e, deg, True)

        inner_BAC = inner_product(B, m_b, C, m_c, A, m_a, deg, True)
        inner_BAD = inner_product(B, m_b, D, m_d, A, m_a, deg, True)
        inner_BAE = inner_product(B, m_b, A, m_a, E, m_e, deg,  True)

        inner_CAB = inner_product(C, m_c, B, m_b, A, m_a, deg, True)

        inner_DAB = inner_product(D, m_d, B, m_b, A, m_a, deg, True)
        
        inner_EAB = inner_product(E, m_e, A, m_a, B, m_b, deg, False)
     

        self.coef_ABC = gamma * inner_ABC
        self.coef_ABD = gamma * inner_ABD
        self.coef_ABE = -gamma * inner_ABE
        
        self.coef_BAC = gamma * inner_BAC
        self.coef_BAD = gamma * inner_BAD
        self.coef_BAE = -gamma * inner_BAE
        
        self.coef_CAB = gamma * inner_CAB
        
        self.coef_DAB = gamma * inner_DAB
        
        self.coef_EAB = -gamma * inner_EAB

        self.mismatch_1 = self.freq_b + self.freq_c - self.freq_a
        self.mismatch_2 = self.freq_b + self.freq_d - self.freq_a
        self.mismatch_3 = self.freq_a + self.freq_b - self.freq_e

        self.S_abc = -S_abc(C, m_c, B, m_b, A, m_a, deg)
        self.S_abd = -S_abc(B, m_b, D, m_d, A, m_a, deg)
        self.S_abe = S_abc(A, m_a, B, m_b, E, m_e, deg)
        
    def f(self, AMP):
        
        abc = self.coef_ABC
        abd = self.coef_ABD
        abe = self.coef_ABE
        
        bac = self.coef_BAC 
        bad = self.coef_BAD
        bae = self.coef_BAE 
        
        cab = self.coef_CAB 
        
        dab = self.coef_DAB 
        
        eab = self.coef_EAB 
        
        A_a = AMP[0]
        A_b = AMP[1]
        A_c = AMP[2]
        A_d = AMP[3]
        A_e = AMP[4]
        #A_a, A_b, A_c, A_d, A_e = AMP
        
        #mode a
        F1 = -1j * self.freq_a * A_a + 1j * abc * A_b * A_c
        F1 += 1j * abd * A_d * A_b + 1j * abe * np.conj(A_b) * A_e

        F2 = -1j * self.freq_b * A_b + 1j * bac * A_a * np.conj(A_c)
        F2 += 1j * bad * A_a * np.conj(A_d)
        F2 += 1j * bae * np.conj(A_a) * A_e

        F3 = -1j * self.freq_c * A_c + 1j * cab * A_a * np.conj(A_b)

        F4 = -1j * self.freq_d * A_d + 1j * dab * A_a * np.conj(A_b)
        
        F5 = -1j * self.freq_e * A_e + 1j * eab * A_a * (A_b)
  

        return np.array([F1, F2, F3, F4, F5])


'''
RUNGE KUTTA 33
'''


def RK33(Five_Waves, t_0, t_f, h, A_0):

    n = (t_f - t_0)/h
    n = int(n)

    y_0 = A_0

    Y = [y_0]

    for k in range(n):

        k1 = Five_Waves.f(Y[-1])
        k2 = Five_Waves.f(Y[-1] + h/2 * k1)
        k3 = Five_Waves.f(Y[-1] + h/2 * k2)
        k4 = Five_Waves.f(Y[-1] + h * k3)

        Y += [Y[-1] + h/6 * (k1 + 2*k2 + 2*k3 + k4)]

    Y = np.array(Y)
    T = np.linspace(t_0, t_f, n+1)

    return Y, T


'''
FOUR-WAVES INTEGRATION
'''

def five_waves_integration(Five_Waves, T1, A_0, t_0, t_f, h, plot = True):
    
    days = (10**5/14.54 * t_f)/24/60/60
    
    A_a, A_b, A_c, A_d, A_e = A_0
    

    Y, T = RK33(Five_Waves, t_0, t_f, h, A_0)
    
    Y_a = Y[:, 0]
    Y_b = Y[:, 1]
    Y_c = Y[:, 2]
    Y_d = Y[:, 3]
    Y_e = Y[:, 4]
    
    # Energy

    Y_a = Y_a * np.conj(Y_a)  # mode a
    Y_b = Y_b * np.conj(Y_b)  # mode b
    Y_c = Y_c * np.conj(Y_c)  # mode c
    Y_d = Y_d * np.conj(Y_d)  # mode d
    Y_e = Y_e * np.conj(Y_e)  # mode e
    
    mc, mb, ma,_,_,_   = Triad_dynamics(T1, np.array([A_c, A_b, A_a]), t_0, t_f, h,p=False)
    
    
    qa = np.real(max(Y_a) - min(Y_a))
    qb = np.real(max(Y_b) - min(Y_b))
    
    ea = np.real((max(Y_a) - min(Y_a) - ma)/ma)
    eb = np.real((max(Y_b) - min(Y_b) - mb)/mb)

   
    L_a = Y_a + (Five_Waves.coef_ABC/Five_Waves.coef_CAB)*Y_c
    L_a += (Five_Waves.coef_ABD/Five_Waves.coef_DAB)*Y_d
    L_a += (Five_Waves.coef_ABE/Five_Waves.coef_EAB)*Y_e
    
    t =  np.linspace(0, days, len(T))
    
    if plot:
        print()
        print('Condition - triads')
        a1 = (Five_Waves.coef_BAC + Five_Waves.coef_CAB - Five_Waves.coef_ABC)
        a1 += (Five_Waves.mismatch_1 * Five_Waves.S_abc)
        print('Triad 1 :', a1)
        a1 = (Five_Waves.coef_BAD + Five_Waves.coef_DAB - Five_Waves.coef_ABD)
        a1 += (Five_Waves.mismatch_2 * Five_Waves.S_abd)
        print('Triad 2 :', a1)
        a1 = (Five_Waves.coef_BAE + Five_Waves.coef_ABE - Five_Waves.coef_EAB)
        a1 += (Five_Waves.mismatch_3 * Five_Waves.S_abe)
        print('Triad 3 :', a1)
   
        plt.plot(t, Y_a, label= Five_Waves.label_a)
        plt.plot(t, Y_b, label= Five_Waves.label_b)
        plt.plot(t, Y_c, label=Five_Waves.label_c)
        plt.plot(t, Y_d, label=Five_Waves.label_d)
        plt.plot(t, Y_e, label=Five_Waves.label_e)
        #plt.plot(t, E_T, label='Total Energy')
        # plt.plot(T, E_2, label = r'$E^{(2)}$')
        # plt.plot(T, E_3, label = r'$E^{(3)}$')
        plt.xlabel('Time (days)')
        plt.ylabel(' Energy (nondimensional)')
        plt.legend(loc='upper right')
        plt.ylim(-0.001, 0.05)
        #plt.title(fr'Maximum efficiency')
        plt.show()
    
    return ea, eb, Y_a, Y_b, Y_c, Y_d, Y_e

def five_two(F,T1, A_0, u_d, u_e):
    
    t_0, t_f, h = 0, 2000, 0.1
    A_a, A_b, A_c, A_d, A_e = A_0
    
    alpha_1 = 0/np.real(u_d)
    alpha_100 = 50/np.real(u_d)  # velocities 1m/s --- 100m/s  mode a
    
    ALPHA = np.linspace(alpha_1, alpha_100, 11)
    
    beta_1 = 0/np.real(u_e)
    beta_100 = 50/np.real(u_e)
    
    BETA = np.linspace(beta_1, beta_100, 11)
    
    A, B = np.meshgrid(ALPHA, BETA)
    
    E_a = []
    E_b = []
    
    J_a = []
    J_b = []
    
    k = 0
    
    _,_,_,_,_,Y3a = Triad_dynamics(T1, np.array([A_c,A_b,A_a]), t_0, t_f, h)
    fft_A = np.fft.fft(Y3a)
    freq  = np.fft.fftfreq(int((t_f - t_0)/h), d=h)
    fft_A = np.abs(fft_A)
    dom3  = freq[np.argmax(fft_A[1:]) + 1]
    
    for beta in BETA:
        
        A_e = beta

        L_a = []
        L_b = []   
        M_a = []
        M_b = []
        k+= 1
        print(f'linha {k}/10')
        j = 0
        for alpha in ALPHA:
            j += 1
            A_d = alpha
            
            A0 = np.array([A_a, A_b, A_c, A_d, A_e])
            
            ea, eb,Y4a,_,_,_,_ = five_waves_integration(F,T1, A0, t_0, t_f, h, plot = False)
            
            L_a += [ea]
            L_b += [eb]
            
            #--------------------
            fft_A = np.fft.fft(Y4a)
            freq  = np.fft.fftfreq(int((t_f - t_0)/h), d=h)
            fft_A = np.abs(fft_A)
            
            max_fft = [1]
            for i in range(1,len(fft_A)-1):
                if fft_A[i] > fft_A[i+1] and fft_A[i] > fft_A[i-1]:
                    max_fft += [fft_A[i]]
                
            dom = freq[np.where(fft_A == max_fft[-1])][0]
            
            dom1 = freq[np.argmax(fft_A[1:]) + 1]  
            
            M_a += [(1/abs(float(dom)) - 1/abs(float(dom3)))/(4*np.pi) ]
            M_b += [(1/abs(float(dom1)) - 1/abs(float(dom3)))/(4*np.pi)]
        
        E_a += [L_a]
        E_b += [L_b]
        
        J_a += [M_a]
        J_b += [M_b]
    
    E_a = np.array(E_a)
    E_b = np.array(E_b)
    J_a = np.array(J_a)
    J_b = np.array(J_b)
    
    '''
    fig, ax = plt.subplots()
    
    va = min(np.min(E_a), -np.max(E_a)) * 100
    vb = min(np.min(E_b), -np.max(E_b)) * 100
    v = min(va, vb)
    
    for i in range(len(E_a)):
        E_a[i] = [j if j < 1 else 1 for j in E_a[i]]
        
    cs = ax.contourf(2*A*u_d, 2*B*u_e, 100*E_a, levels=100, cmap = 'terrain',
                     vmin = 0, vmax = 100)
    plt.xlabel(f'{F.label_d} - zonal velocity (m/s)')
    plt.ylabel(f'{F.label_e} - zonal velocity (m/s)')
    plt.title(fr'$P$ measure - {F.label_a}')

    fig.colorbar(cs)
    plt.show()   
    
    fig, ax = plt.subplots()
    
    vmin = min(np.min(E_b), -np.max(E_b)) * 100
    
    for i in range(len(E_a)):
        E_b[i] = [j if j < 1 else 1 for j in E_b[i]]
    cs = ax.contourf(2*A*u_d, 2*B*u_e, 100*E_b, levels=100, cmap = 'terrain',
                     vmin = 0, vmax = 100)
    plt.xlabel(f'{F.label_d} - zonal velocity (m/s)')
    plt.ylabel(f'{F.label_e} - zonal velocity (m/s)')
    plt.title(fr'$P$ measure - {F.label_b}')
    fig.colorbar(cs)
    plt.show() '''
    
    #---------
    vb = min(np.min(J_b), -np.max(J_b)) 
    fig, ax = plt.subplots()
    cs = ax.contourf(2*A*u_d, 2*B*u_e, J_b, levels=100, cmap = 'bwr',
                     vmin = vb, vmax = -vb)
    plt.xlabel(f'{Five_Waves.label_d} - zonal velocity (m/s)')
    plt.ylabel(f'{Five_Waves.label_e} - zonal velocity (m/s)')
    plt.title(f'Dominant periods difference')
    fig.colorbar(cs)
    plt.show()
    
    fig.colorbar(cs)
    plt.show() 
    
    va = min(np.min(J_a), -np.max(J_a))
    vb = min(np.min(J_b), -np.max(J_b)) * 100
    v = np.real(min(va, vb))
    
    fig, ax = plt.subplots()
    for i in range(len(E_a)):
        J_a[i] = [j if j < 50 else 50 for j in J_a[i]]
    levels = [0,1,2,3,4,5,6,7,8,9,10,20,30,40,50]
    colors = ["#333399", "#1b63c9", "#0393f9", "#00b8a0", "#25d36d", 
              "#6de27c", "#b5f08a", "#fefe98", "#dad085", "#b6a272", 
              "#92735e", "#93756e", "#b7a39f", "#dbd1cf", "#ffffff"]
    cs = ax.contourf(2*A*u_d, 2*B*u_e, J_a, levels=levels, colors = colors,
                     vmin = 0, vmax = 50)
    plt.xlabel(f'{Five_Waves.label_d} - zonal velocity (m/s)')
    plt.ylabel(f'{Five_Waves.label_e} - zonal velocity (m/s)')
    plt.title(f'Periods difference')
    fig.colorbar(cs)
    plt.show()
    
    return A,B,J_a, J_b
    
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
eps = 8.810572669756384
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
t_f = 150
h = 0.1


(m_a, n_a, alpha_a, m_b, n_b, alpha_b,m_c, n_c, alpha_c,
 m_d, n_d, alpha_d, m_e, n_e, alpha_e) = 4,5,3, 3,4,3, 1,2,3, 1,1,1, 7,9,1 #EEER


Five_Waves = FIVE_WAVES(gamma, m_a, n_a, alpha_a, m_b, n_b, alpha_b,
                        m_c, n_c, alpha_c, m_d, n_d, alpha_d, m_e, n_e, alpha_e)

print('Coefficients')
print('Triad 1')
print('A = ', Five_Waves.coef_ABC)
print('B = ', Five_Waves.coef_BAC)
print('C = ', Five_Waves.coef_CAB)
print()
print('Triad 2')
print('A = ', Five_Waves.coef_ABD)
print('B = ', Five_Waves.coef_BAD)
print('D = ', Five_Waves.coef_DAB)
print()
print('Triad 3')
print('A = ', Five_Waves.coef_ABE)
print('B = ', Five_Waves.coef_BAE)
print('C = ', Five_Waves.coef_EAB)
print()
print()

#Triad_1 = TRIAD(gamma, m_c, n_c, alpha_c, m_b, n_b, alpha_b, m_a, n_a, alpha_a)
#Triad_2 = TRIAD(gamma, m_d, n_d, alpha_d, m_b, n_b, alpha_b, m_a, n_a, alpha_a)
#Triad_3 = TRIAD(gamma, m_c, n_c, alpha_c, m_e, n_e, alpha_e, m_b, n_b, alpha_b)
#Triad_4 = TRIAD(gamma, m_d, n_d, alpha_d, m_e, n_e, alpha_e, m_b, n_b, alpha_b)
u_a = norm_component(Five_Waves.uvh_a[0])*np.sqrt(g*h_e)
u_b = norm_component(Five_Waves.uvh_b[0])*np.sqrt(g*h_e)
u_c = norm_component(Five_Waves.uvh_c[0])*np.sqrt(g*h_e)
u_d = norm_component(Five_Waves.uvh_d[0])*np.sqrt(g*h_e)
u_e = norm_component(Five_Waves.uvh_e[0])*np.sqrt(g*h_e)

A_a = 1/u_a
A_b = 1/u_b
A_c = 1/u_c
A_d = 1/u_d
A_e = 1/u_e

A_0 = np.array([A_a*15, A_b*15, A_c*18, A_d*30, A_e*15])
print(A_0)
Triad_1 = TRIAD(gamma, m_c, n_c, alpha_c, m_b, n_b, alpha_b, m_a, n_a, alpha_a)
print('Coef a = ', Triad_1.coef_ABC)
#five_waves_integration(Five_Waves,Triad_1, A_0, t_0, t_f=2000, h=0.01)
A,B,Ja,Jb = five_two(Five_Waves, Triad_1, A_0, u_d, u_e)

