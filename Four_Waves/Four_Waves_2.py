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
from period_Harris import PERIOD
from period_Harris import Amp_change
from Dynamic_Triads import TRIAD
from Dynamic_Triads import Triad_dynamics


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
        
        label_a = f'({m_a}, {n_a},'
        if alpha_a == 1:
            label_a += ' EG)'
        elif alpha_a == 2:
            label_a += ' WG)'
        else:
            label_a += ' RH)'
        self.label_a = label_a
        
        self.uvh_a = A

        B = norm_Hough(m_b, n_b, alpha_b, gamma, N, deg)
        eigen_b = B[-1]
        B = B[:-3]
        
        label_b = f'({m_b}, {n_b},'
        if alpha_b == 1:
            label_b += ' EG)'
        elif alpha_b == 2:
            label_b += ' WG)'
        else:
            label_b += ' RH)'
        self.label_b = label_b

        self.uvh_b = B

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

        self.coef_ABC = gamma * inner_ABC
        self.coef_ABD = gamma * inner_ABD
        self.coef_BAC = gamma * inner_BAC
        self.coef_BAD = gamma * inner_BAD
        self.coef_CAB = gamma * inner_CAB
        self.coef_DAB = gamma * inner_DAB

        self.mismatch_1 = -self.freq_a + self.freq_c + self.freq_b
        self.mismatch_2 = -self.freq_a + self.freq_b + self.freq_d

        self.S_abc =  -S_abc(C, m_c, B, m_b, A, m_a, deg)
        self.S_abd =  -S_abc(B, m_b, D, m_d, A, m_a, deg)

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

def four_waves_integration(Four_Waves, A_0, t_0, t_f, h, plot = True):
    t_f = 200
    days = (10**5/14.54 * t_f)/24/60/60
    
    A_a, A_b, A_c, A_d = A_0
    
    Energy_0  = A_a * np.conj(A_a) + A_b * np.conj(A_b)
    Energy_0 += A_c * np.conj(A_c) + A_d * np.conj(A_d)  # quadratic
    
    Energy_0 += 2*np.real(np.conj(A_c)*np.conj(A_b)*(A_a))*Four_Waves.S_abc
    Energy_0 += 2*np.real(A_b*A_d*np.conj(A_a))*Four_Waves.S_abd
    
    #print()
    al1 = Four_Waves.coef_BAC + Four_Waves.coef_CAB -Four_Waves.coef_ABC
    al2 = Four_Waves.coef_BAD + Four_Waves.coef_DAB - Four_Waves.coef_ABD
    #print('Sabc = ',al1 +  Four_Waves.mismatch_1 * Four_Waves.S_abc)
    #print('Sabd = ', al2 + Four_Waves.mismatch_2 * Four_Waves.S_abd)
    
    
    Y, T = RK33(Four_Waves, t_0, t_f, h, A_0)

    Y_a = Y[:, 0]
    Y_b = Y[:, 1]
    Y_c = Y[:, 2]
    Y_d = Y[:, 3]
    #print('cd = ', np.imag(Y_c*np.conj(Y_d)))
    
    #I = integral(Four_Waves, Y_a, Y_b, Y_c, Y_d)
    
    C = np.copy(Y_c)
    D = np.copy(Y_d)
    
    
    # Energy

    E_3  = 2*np.real(Y_c*Y_b*np.conj(Y_a))*Four_Waves.S_abc
    E_3 += 2*np.real(Y_b*Y_d*np.conj(Y_a))*Four_Waves.S_abd
    E_3 /= Energy_0
    #print()
    #print()
    #print('len = ', len(E_3))
    #print('E_3 = ', np.real(E_3[200:1000]))
    Y_a = Y_a * np.conj(Y_a)/Energy_0  # mode a
    Y_b = Y_b * np.conj(Y_b)/Energy_0  # mode b
    Y_c = Y_c * np.conj(Y_c)/Energy_0  # mode c
    Y_d = Y_d * np.conj(Y_d)/Energy_0  # mode d
    
    
    L_E = -Four_Waves.coef_ABD*Y_b*Four_Waves.S_abc + Four_Waves.coef_BAD*Y_a*Four_Waves.S_abc
    L_E += Four_Waves.coef_ABC*Y_b*Four_Waves.S_abd - Four_Waves.coef_BAC*Y_a*Four_Waves.S_abd
    
    #print('L_E = ', np.real(L_E))
    #print()
    L_1 = Y_a + Four_Waves.coef_ABC/Four_Waves.coef_CAB * Y_c 
    L_1 += Four_Waves.coef_ABD/Four_Waves.coef_DAB * Y_d
    
    L_2 = Y_b - Four_Waves.coef_BAC/Four_Waves.coef_CAB * Y_c
    L_2 -= Four_Waves.coef_BAD/Four_Waves.coef_DAB * Y_d
    
    ea = np.real(max(Y_a) - min(Y_a))
    eb = np.real(max(Y_b) - min(Y_b))
    ec = np.real(max(Y_c) - min(Y_c))
    ed = np.real(max(Y_d) - min(Y_d))
    
    #print('Efficiency = ', np.real(max(Y_d) - min(Y_d)))
    #print('--------------------------------------')
    E_2 = Y_a + Y_b + Y_c + Y_d
    #print('E_2 = ', np.real(E_2[20:40]))
    E_T = E_2 + E_3
    #print('E_T = ', np.real(E_T[900:1100]))
    #print('max = ', max(np.real(E_T)))
    t = np.linspace(0, days, len(T))
    
    
    '''
    plt.plot(t, L_1, label = 'L1')
    plt.plot(t, L_2, label = 'L2')
    plt.legend()
    plt.show()
    plt.plot(t,E_2 - E_2[0], label = 'e2')
    plt.plot(t,E_3 - E_3[0], label = 'e3')
    plt.plot(t, E_T - E_T[0], label = 'et')
    plt.legend()
    plt.show()
    '''
    
    if plot:
        
        VAR1 = 2*(Four_Waves.coef_ABD * Y_b * Energy_0 - Four_Waves.coef_BAD * Y_a * Energy_0)
        VAR1 *= Four_Waves.S_abc
        VAR2 = 2*(Four_Waves.coef_ABC * Y_b * Energy_0 - Four_Waves.coef_BAC * Y_a * Energy_0)
        VAR2 *= Four_Waves.S_abd
        
        VAR = (VAR1 - VAR2) * np.imag(C* np.conj(D))
        #print('C*D = ',(C * np.conj(D))[:10])
        #print(' 1 - 2', (VAR1 - VAR2)[:10])
        #print('VAR = ', np.real(VAR[:10]))
        ET1 = (E_T[1:] - E_T[:-1] )/(h) * Energy_0
        print('max = ', max(E_T))
        print('min = ', min(E_T))
        print(max(E_T) - min(E_T))
        #print(np.real(ET1[:10]))
        #print('dif = ', np.real(VAR[:-1] - ET1)[:10])
        #print('max dif = ', max(np.real(VAR[:-1] - ET1)))
        x = np.argmax(VAR[:-1] - ET1)
        #print(VAR[x+1])
        #print(ET1[x])
        
        #tt = PERIOD(Triad_b, Amp_change(Triad_b, A_g)) / (4*np.pi)
        #plt.plot(t, Y_a, label= Four_Waves.label_a)
        #plt.plot(t, Y_b, label= Four_Waves.label_b)
        #plt.plot(t, Y_c, label= Four_Waves.label_c)
        #plt.plot(t, Y_d, label= Four_Waves.label_d)
        plt.plot(t, E_T, label='Total Energy')
        plt.plot(t, E_2, label = r'$E^{(2)}$')
        plt.plot(t, E_3, label = r'$E^{(3)}$')
        #plt.axvline(x = float(tt), color='red', lw='1', linestyle='--')
        plt.xlabel('Time (days)')
        plt.ylabel(' Energy (nondimensional)')
        #plt.ylim(top= 1.03)
        
        #plt.legend()
        plt.legend(loc='upper right')
        #plt.title(fr'Maximum efficiency')
        plt.show()
        print()
        print(f'Efficiency - {Four_Waves.label_a}: ', 100*np.real(max(Y_a) - min(Y_a)))
        print(f'Efficiency - {Four_Waves.label_b}: ', 100*np.real(max(Y_b) - min(Y_b)))
        print(f'Efficiency - {Four_Waves.label_c}: ', 100*np.real(max(Y_c) - min(Y_c)))
        print(f'Efficiency - {Four_Waves.label_d}: ', 100*np.real(max(Y_d) - min(Y_d)))
        print()
        '''
        tt = [t[i] for i in range(0,len(t)-1, 30)]
        et = [ET1[i] for i in range(0, len(ET1), 30)]
        #plt.plot(t, E_T * Energy_0, label = 'Total')
        plt.scatter(tt, et, label = 'approx.')
        plt.plot(t, VAR, label = 'derivative', color = 'orange')
        plt.legend()
        plt.show()'''
        
    
    return ea, eb, ec, ed, Y_a, Y_b, Y_c, Y_d

def integral(F, Aa, Bb, Cc, Dd):
    
    point, weight = np.polynomial.legendre.leggauss(300)
    Ang = np.pi * (point + 1)
    
    ma = F.mode_a[0]
    mb = F.mode_b[0]
    mc = F.mode_c[0]
    md = F.mode_d[0]
    
    Int_time = []
    
    
    plt.plot(Ang*180/np.pi, -F.uvh_d[0], label = 'u' )
    plt.plot(Ang*180/np.pi, F.uvh_d[1], label = 'v' )
    plt.plot(Ang*180/np.pi, -F.uvh_d[2], label = 'h' )
    plt.ylim(-1.5,1.5)
    plt.xlim(left = 0)
    plt.grid()
    plt.legend()
    plt.show()
    print('-----')
    print(len(Aa))
    for i in range(len(Aa)):
        
      if i % 100 == 0:
          print(i)
      
      A = Aa[i]
      B = Bb[i]
      C = Cc[i]
      D = Dd[i]
      
      L = []
      
      point, weight = np.polynomial.legendre.leggauss(len(F.uvh_a[0]))
      angle = np.pi/2 * point
      cos = np.cos(angle)
        
      for lam in Ang:
        
        u  = -F.uvh_a[0] * A * np.exp(1j * lam * ma) - F.uvh_b[0] * B * np.exp(1j * lam * mb)
        u += -F.uvh_c[0] * C * np.exp(1j * lam * mc) - F.uvh_d[0] * D * np.exp(1j * lam * md)
        u += np.conj(u)
        
        v  = F.uvh_a[1] * A * np.exp(1j * lam * ma) + F.uvh_b[1] * B * np.exp(1j * lam * mb)
        v += F.uvh_c[1] * C * np.exp(1j * lam * mc) + F.uvh_d[1] * D * np.exp(1j * lam * md)
        v *= 1j
        v += np.conj(v)
        
        h  = -F.uvh_a[2] * A * np.exp(1j * lam * ma) - F.uvh_b[2] * B * np.exp(1j * lam * mb)
        h += -F.uvh_c[2] * C * np.exp(1j * lam * mc) - F.uvh_d[2] * D * np.exp(1j * lam * md)
        h += np.conj(h)
        
        
        
        Energy  = (u*np.conj(u) + v*np.conj(v) + h*np.conj(h))
        Energy += h*(u*np.conj(u) + v*np.conj(v))
        
        int_phi = np.pi/2* sum(weight * (Energy)*cos)
        
        L += [int_phi]
    
      L = np.array(L)
      point, weight = np.polynomial.legendre.leggauss(len(Ang))
    
      integral = np.pi* sum(weight * (L))
      
      Int_time += [integral/(4*np.pi)]
    
    Int_time = np.array(Int_time)
    return Int_time
    
    
        
        
        
        

def eff_graph(Four, A_0, u_d):
    
    A_a, A_b, A_c, A_d = A_0
    
    alpha_1 = 1/np.real(u_d)
    alpha_100 = 50/np.real(u_d)  # velocities 1m/s --- 100m/s  mode a
    
    ALPHA = np.linspace(alpha_1, alpha_100, 100)
    
    E_a = []
    E_b = []
    E_c = []
    E_d = []

    for alpha in ALPHA:
        
        A_d = alpha
        A0  = np.array([A_a, A_b, A_c, A_d])
        ea, eb, ec, ed = four_waves_integration(Four, A0, t_0=0, t_f=300, 
                                                h=0.1, plot = False)
    
        E_a += [ea]
        E_b += [eb]
        E_c += [ec]
        E_d += [ed]
      
    E_a = np.array(E_a)
    E_b = np.array(E_b)
    E_c = np.array(E_c)
    E_d = np.array(E_d)
    
    plt.plot(ALPHA*u_d*2, E_a, label = f'{Four.label_a}')
    plt.plot(ALPHA*u_d*2, E_b, label = f'{Four.label_b}')
    plt.plot(ALPHA*u_d*2, E_c, label = f'{Four.label_c}')
    plt.plot(ALPHA*u_d*2, E_d, label = f'{Four.label_d}')
    plt.xlabel(f'Zonal Velocity - {Four.label_d}  (m/s)')
    plt.ylabel('Efficiency (%)')
    plt.legend()
    plt.show() 
    
def period_graph(Four, T1, T2, A_0, u_a):
    
    
    t_0 = 0
    t_f = 300
    h = 0.1
    A_a, A_b, A_c, A_d = A_0
    
    alpha_1 = 0/np.real(u_c)
    alpha_100 = 100/np.real(u_c)  # velocities 1m/s --- 100m/s  mode a
    
    ALPHA = np.linspace(alpha_1, alpha_100, 51)
    
    P1 = []
    P2 = []
    Ec = []
    Ed = []
    E01 = []
    E02 = []
    for alpha in ALPHA:
        
        A_c = alpha
        A0  = np.array([A_a, A_b, A_c, A_d])
        
        p1 = PERIOD(T1, Amp_change(T1, np.array([A_c, A_b, A_a]) ))
        p2 = PERIOD(T2, Amp_change(T2, np.array([A_d, A_b, A_a]) ))
        
        _, _, ec, ed, = four_waves_integration(Four, A0, t_0, t_f, 
                                                h, plot = False) 
        ec1, eb1, ea1 = Triad_dynamics(T1, np.array([A_c, A_b, A_a]), t_0, t_f
                                       , h, p = False)
        ed2, eb2, ea2 = Triad_dynamics(T2, np.array([A_d, A_b, A_a]), t_0, t_f
                                       , h, p = False)
        P1 += [p1]
        P2 += [p2]
        Ec += [ec-ec1]
        Ed += [ed-ed2]
        E01 += [A_a*np.conj(A_a) + A_b*np.conj(A_b) + A_c*np.conj(A_c)]
        E02 += [A_a*np.conj(A_a) + A_b*np.conj(A_b) + A_d*np.conj(A_d)]
        
    P1 = np.array(P1)/(4*np.pi)
    P2 = np.array(P2)/(4*np.pi)
    
    E01 = np.array(E01)
    E02 = np.array(E02)
    
    plt.plot(ALPHA*u_a*2, P1, label = 'Triad 1')
    plt.plot(ALPHA*u_a*2, P2, label = 'Triad 2')
    plt.legend()
    plt.show()
    
    plt.plot(ALPHA*u_a*2, E01/P1 , label = 'Triad 1')
    plt.plot(ALPHA*u_a*2, E02/P2 , label = 'Triad 2')
    plt.legend()
    plt.show()
    
    plt.plot(ALPHA*u_a*2, Ec, label = f'{Four.label_c}')
    plt.plot(ALPHA*u_a*2, Ed, label = f'{Four.label_d}')
    
    plt.legend()
    plt.show()
        
def eff_graph_double(Four,T1, A_0, u_c, u_d):
    
    t_0, t_f, h = 0, 2000, 0.1
    A_a, A_b, A_c, A_d = A_0
    
    alpha_1 = 0/np.real(u_d)
    alpha_100 = 50/np.real(u_d)  # velocities 1m/s --- 100m/s  mode a
    
    ALPHA = np.linspace(alpha_1, alpha_100, 11)
    
    beta_1 = 0/np.real(u_c)
    beta_100 = 50/np.real(u_c)
    
    BETA = np.linspace(beta_1, beta_100, 11)
    
    A, B = np.meshgrid(ALPHA, BETA)
    
    E_a = []
    E_b = []
    
    J_a = []
    J_b = []
    
    k = 0
    for beta in BETA:
        
        A_c = beta

        L_a = []
        L_b = []    
        
        M_a = []
        M_b = []
        k+= 1
        print(f'linha {k}/10')
        for alpha in ALPHA:
            
            A_d = alpha
            
            A0 = np.array([A_a, A_b, A_c, A_d])
            
            ea, eb,_,_, Y4a, Y4b, _,_ = four_waves_integration(Four, A0, t_0, t_f, h, plot = False)
            
            _, ebb, eaa, _, Y3b, Y3a = Triad_dynamics(T1, np.array([A_c, A_b, A_a]), t_0, 
                                         t_f, h, p = False)
            
            L_a += [ea - eaa]
            L_b += [eb - ebb]
            
            en_4  = np.abs(A_a)**2 + np.abs(A_b)**2 + np.abs(A_c)**2 + np.abs(A_d)**2            
            en_4 += 2*np.real(A_b*A_c*np.conj(A_a))*Four_Waves.S_abc
            en_4 += 2*np.real(A_b*A_d*np.conj(A_a))*Four_Waves.S_abd
            en_4 = np.real(en_4)
            
            en_3  = np.abs(A_a)**2 + np.abs(A_b)**2 + np.abs(A_c)**2
            en_3 += 2*np.real(A_b*A_c*np.conj(A_a))*Four_Waves.S_abc
            en_3 = np.real(en_3)
            
            #---------------------------
            fft_A = np.fft.fft(Y4a)
            freq  = np.fft.fftfreq(int((t_f - t_0)/h), d=h)
            fft_A = np.abs(fft_A)
            
            max_fft = [1]
            for i in range(1,len(fft_A)-1):
                if fft_A[i] > fft_A[i+1] and fft_A[i] > fft_A[i-1]:
                    max_fft += [fft_A[i]]
                
            dom = freq[np.where(fft_A == max_fft[-1])]
            dom = freq[np.argmax(fft_A[1:]) + 1]
            
            fft_A = np.fft.fft(Y3a)
            freq  = np.fft.fftfreq(int((t_f - t_0)/h), d=h)
            fft_A = np.abs(fft_A)
            dom3  = freq[np.argmax(fft_A[1:]) + 1]
            
            #------------------------------------
            M_a += [(1/abs(float(dom)) - 1/abs(float(dom3)))/(4*np.pi) ]
            #M_a += [ea*en_4/(eaa*en_3) - 1]
            M_b += [eb*en_4/(ebb*en_3) - 1]
                    
        E_a += [L_a]
        E_b += [L_b]
        
        J_a += [M_a]
        J_b += [M_b]
      
    
    E_a = np.array(E_a)
    E_b = np.array(E_b)
    J_a = np.array(J_a)
    J_b = np.array(J_b)

    fig, ax = plt.subplots()
    
    va = min(np.min(E_a), -np.max(E_a)) * 100
    vb = min(np.min(E_b), -np.max(E_b)) * 100
    v = min(va, vb)
    
    cs = ax.contourf(2*A*u_d, 2*B*u_c, 100*E_a, levels=100, cmap = 'bwr',
                     vmin = v, vmax = -v)
    plt.xlabel(f'{Four.label_d} - zonal velocity (m/s)')
    plt.ylabel(f'{Four.label_c} - zonal velocity (m/s)')
    plt.title(f'Efficiency difference - {Four.label_a}')
    plt.axhline(y=30, linestyle='--', color = 'black')
    plt.scatter(0,30, c = 'black')
    plt.scatter(20,30, c = 'black')
    plt.scatter(80,30, c='black')
    plt.annotate(f'Ex 1', (0, 30), textcoords="offset points", 
                 xytext=(10, 10), ha='center')
    plt.annotate(f'Ex 2', (80, 30), textcoords="offset points", 
                 xytext=(10, 10), ha='center')
    plt.annotate(f'Ex 3', (20, 30), textcoords="offset points", 
                 xytext=(10, 10), ha='center')
 
    fig.colorbar(cs)
    plt.show()   
    
    fig, ax = plt.subplots()
    
    vmin = min(np.min(E_b), -np.max(E_b)) * 100
    cs = ax.contourf(2*A*u_d, 2*B*u_c, 100*E_b, levels=100, cmap = 'bwr',
                     vmin = v, vmax = -v)
    plt.xlabel(f'{Four.label_d} - zonal velocity (m/s)')
    plt.ylabel(f'{Four.label_c} - zonal velocity (m/s)')
    plt.title(f'Efficiency difference - {Four.label_b}')
    plt.axhline(y=30, linestyle='--', color = 'black')
    plt.scatter(0,30, c = 'black')
    plt.scatter(20,30, c = 'black')
    plt.scatter(80,30, c='black')
    plt.annotate(f'Ex 1', (0, 30), textcoords="offset points", 
                 xytext=(10, 10), ha='center')
    plt.annotate(f'Ex 2', (80, 30), textcoords="offset points", 
                 xytext=(10, 10), ha='center')
    plt.annotate(f'Ex 3', (20, 30), textcoords="offset points", 
                 xytext=(10, 10), ha='center')
    
    fig, ax = plt.subplots()
    cs = ax.contourf(2*A*u_d, 2*B*u_c, 100*J_b, levels=100, cmap = 'bwr',
                     vmin = -100, vmax = 100)
    plt.xlabel(f'{Four.label_d} - zonal velocity (m/s)')
    plt.ylabel(f'{Four.label_c} - zonal velocity (m/s)')
    plt.title(f'Efficiency difference - {Four.label_b}')
    plt.axhline(y=30, linestyle='--', color = 'black')
    plt.scatter(0,30, c = 'black')
    plt.scatter(20,30, c = 'black')
    plt.scatter(80,30, c='black')
    plt.annotate(f'Ex 1', (0, 30), textcoords="offset points", 
                 xytext=(10, 10), ha='center')
    plt.annotate(f'Ex 2', (80, 30), textcoords="offset points", 
                 xytext=(10, 10), ha='center')
    plt.annotate(f'Ex 3', (20, 30), textcoords="offset points", 
                 xytext=(10, 10), ha='center')
    
    fig.colorbar(cs)
    plt.show()
    
    '''fig.colorbar(cs)
    plt.show() 
    
    va = min(np.min(J_a), -np.max(J_a))
    vb = min(np.min(J_b), -np.max(J_b)) * 100
    v = np.real(min(va, vb))
    
    fig, ax = plt.subplots()
    
    levels = [0,1,2,3,4,5,6,7,8,9,10,20,30,40,50]
    colors = ["#333399", "#1b63c9", "#0393f9", "#00b8a0", "#25d36d", 
              "#6de27c", "#b5f08a", "#fefe98", "#dad085", "#b6a272", 
              "#92735e", "#93756e", "#b7a39f", "#dbd1cf", "#ffffff"]
    cs = ax.contourf(2*A*u_d, 2*B*u_c, J_a, levels=levels, colors = colors,
                     vmin = 0, vmax = 50)
    plt.xlabel(f'{Four.label_d} - zonal velocity (m/s)')
    plt.ylabel(f'{Four.label_c} - zonal velocity (m/s)')
    plt.title(f'Efficiency difference - {Four.label_b}')
    fig.colorbar(cs)
    plt.show()'''
    
    return A, B, J_a
            
def eff_dif_graph(Four, T1, A_0, u_d):
    
    t_0, t_f, h = 0, 600, 0.1
    
    A_a, A_b, A_c, A_d = A_0
    
    alpha_1 = 0/np.real(u_d)
    alpha_100 = 100/np.real(u_d)  # velocities 1m/s --- 100m/s  mode a
    
    ALPHA = np.linspace(alpha_1, alpha_100, 51)
    
    E_a = []
    E_b = []
    E_d = []
    a = []
    aa = []
    i = 0
    
    ecc, ebb, eaa = Triad_dynamics(T1, np.array([A_c, A_b, A_a]), t_0, 
                                t_f=100, h=0.1, p = True)
    for alpha in ALPHA:
     
        A_d = alpha
        A0  = np.array([A_a, A_b, A_c, A_d])
        ea, eb, ec, ed = four_waves_integration(Four, A0, t_0=0, t_f=300, 
                                                h=0.1, plot = False)
        
        
        E_a += [ea-eaa]
        E_b += [eb-ebb]
        E_d += [ec-ecc]
        a += [eaa]
        aa += [ea]
   
    E_a = np.array(E_a)
    E_b = np.array(E_b)
    E_d = np.array(E_d)
    
    plt.plot(ALPHA*u_d*2, 100*E_a, label = f'{Four.label_a}')
    plt.plot(ALPHA*u_d*2, 100*E_b, label = f'{Four.label_b}')
    plt.plot(ALPHA*u_d*2, 100*E_d, label = f'{Four.label_d}')
    plt.xlabel(f'Zonal Velocity - {Four.label_c}  (m/s)')
    plt.ylabel('Efficiency (%)')
    plt.axhline(y=0, linestyle='--')
    plt.legend()
    plt.show()            

def period_Fourier(Q, T1, T2, Amp, t_0, t_f, h):
    
    a,b,c,d = Amp
    
    t = np.arange(t_0, t_f+h, h)
    
    _,_,_,_,a4, b4, c4, d4 = four_waves_integration(Q, Amp, t_0, t_f, h)
    _,_,_,c3,b3,a3   = Triad_dynamics(T1, np.array([c,b,a]), t_0, t_f, h, p=False)
    _,_,_,d3,b33,a33 = Triad_dynamics(T2, np.array([d,b,a]), t_0, t_f, h, p=False)
    
    Amp_a = np.array([b4])
    Amp_b = np.array([b4,b3,b33])
    
    legend = ['Quartet', f'Modes {T1.label_c};{T1.label_b};{T1.label_a}', 
              f'Modes {T2.label_c};{T2.label_b};{T2.label_a}']
    i = 0
    for A in Amp_a:
        fft_A = np.fft.fft(A)
        freq  = np.fft.fftfreq(len(t), d=h)
        fft_A = np.abs(fft_A)
        dom = freq[np.argmax(fft_A[1:])+1]
        print(1/dom/(4*np.pi))
    
        plt.plot(1/freq/(4*np.pi), fft_A, label = legend[i])
        i += 1

    plt.xlim(left=-100, right = 100)
    plt.xlabel('Period (days)')
    plt.ylabel('Power spectrum')
    plt.yscale('log')
    plt.title(f'Power Spectrum - mode {Q.label_a}')
    plt.ylim(90,120)
    plt.legend()
    plt.show()
    
    fig, axes = plt.subplots(3,1,figsize = (6,6))
    j = 0
    for A in Amp_b:
        fft_A = np.fft.fft(A)
        print(len(fft_A))
        freq  = np.fft.fftfreq(len(t), d=h)
        print(len(freq))
        fft_A = np.abs(fft_A)
        
        ax = axes[j]
        ax.plot(1/freq[:]/(4*np.pi), fft_A, label = legend[j], color = 'b')
        ax.set_title(legend[j], loc = 'right')
        ax.set_xlim(left = 0, right = 400)
        ax.set_yscale('log')
        if j == 2:
            
            ax.set_xlabel('Period (days)')
            
        max_fft = [1]
        for i in range(1,len(fft_A)-1):
            if fft_A[i] > fft_A[i+1] and fft_A[i] > fft_A[i-1]:
                max_fft += [fft_A[i]]
            
        dom = freq[np.where(fft_A == max_fft[-1])]
        print(f'{1/dom/(4*np.pi)} and {max_fft[-1]}')
        j += 1
    
    plt.tight_layout()
    plt.show()
    '''            
    plt.xlim(left=0, right = 20)
    plt.xlabel('Period (days)')
    plt.ylabel('Power spectrum')
    plt.yscale('log')
    plt.title(f'Power Spectrum - mode {Q.label_b}')
    plt.legend()
    plt.show()'''
    
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
t_f = 400 #25.2
h = 0.01


(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
 m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  4, 5, 3, 1, 2, 3, 3, 4, 3, 3, 6, 3 #EEER
# exemplo legal de como a terceira onda, mesmo com pouca energia intefere na dinamica
# das outras tres, reduzindo a eficiencia da quarta - de 65 para 25%
#(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
# m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  4, 4, 1, 2, 4, 1, 2, 3, 3, 2, 7, 3 #EEER
#(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
# m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  4, 4, 1, 2, 4, 1, 2, 5, 3, 2, 3, 3 #EEER


(m_a, n_a, alpha_a, m_b, n_b, alpha_b,
 m_c, n_c, alpha_c, m_d, n_d, alpha_d) =  4, 5, 3, 3, 4, 3, 1, 2, 3, 1, 1 ,1 #EEER


Four_Waves = FOUR_WAVES(gamma, m_a, n_a, alpha_a, m_b, n_b, alpha_b,
                        m_c, n_c, alpha_c, m_d, n_d, alpha_d)

Triad_a = TRIAD(gamma, m_c, n_c, alpha_c, m_b, n_b, alpha_b, m_a, n_a, alpha_a)
Triad_b = TRIAD(gamma, m_d, n_d, alpha_d, m_b, n_b, alpha_b, m_a, n_a, alpha_a)

print('------------------')
print('FOUR WAVES')
print('------------------')
print('Coef_A = ', Four_Waves.coef_ABC)
print('Coef_B = ', Four_Waves.coef_BAC)
print('Coef_C = ', Four_Waves.coef_CAB)
print()
print('Coef_A = ', Four_Waves.coef_ABD)
print('Coef_B = ', Four_Waves.coef_BAD)
print('Coef_D = ', Four_Waves.coef_DAB)
print()
print('Coef_A Tri = ', Triad_a.coef_ABC)
print('Coef_A = ', Triad_b.coef_ABC)
#print('CoefA1 = ', Triad_a.coef_ABC)
#print('CoefA2 = ', Triad_b.coef_ABC)
print('Triad1 = ', -Four_Waves.coef_ABC + Four_Waves.coef_BAC + Four_Waves.coef_CAB +
    Four_Waves.mismatch_1 * Four_Waves.S_abc      )
print('Triad2 = ', -Four_Waves.coef_ABD + Four_Waves.coef_BAD + Four_Waves.coef_DAB +
    Four_Waves.mismatch_2 * Four_Waves.S_abd      )
print()
print('A1:', Four_Waves.coef_BAC/Four_Waves.S_abc)
print('A2:', Four_Waves.coef_BAD/Four_Waves.S_abd)
print('S1:', Four_Waves.S_abc)
print('S2:', Four_Waves.S_abd)

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

A_0 = np.array([A_a*15, A_b*15, A_c*15, A_d*40])
#A_0 = np.array([1,1,1,1])
print(A_0)
#period_graph(Four_Waves, Triad_a, Triad_b, A_0, u_d)
four_waves_integration(Four_Waves, A_0, t_0, t_f, h)
#Triad_dynamics(Triad_a, np.array([A_0[2], A_0[1], A_0[0]]), t_0, t_f, h)
#Triad_dynamics(Triad_b, np.array([A_0[3], A_0[1], A_0[0]]), t_0, t_f, h)
#A,B, J_a = eff_graph_double(Four_Waves, Triad_a, A_0, u_c, u_d)
#eff_dif_graph(Four_Waves, Triad_a, A_0, u_d)
#eff_graph(Four_Waves, A_0, u_d)
#period_Fourier(Four_Waves, Triad_a, Triad_b, A_0, t_0, t_f, h)
'''
print()
print('----------------------')
period_a = PERIOD(Triad_a , Amp_change(Triad_a,  np.array([A_0[2], A_0[1], A_0[0]])))
Triad_dynamics(Triad_a, np.array([A_0[2], A_0[1], A_0[0]]), t_0, t_f = 2*period_a, h = 0.1,
               p = 'True')
print('Period = ', 
      period_a/(4*np.pi))
print()
print('-----------------------')
period_b =  PERIOD(Triad_b , Amp_change(Triad_b, np.array([A_0[3], A_0[1], A_0[0]])))
Triad_dynamics(Triad_b, np.array([A_0[3], A_0[1], A_0[0]]), t_0, t_f = 2*period_b, h = 0.1,
               p = 'True')
print('Period = ', 
      period_b/(4*np.pi))
'''

Triad_a = 1
#four_waves_integration(Four_Waves, A_0, t_0, t_f=max(period_a, period_b), h=0.01)
#four_waves_integration(Four_Waves, A_0, t_0, t_f=300, h=0.1)
P1 = []
P2 = []
E = []
E1 = []
E2 = []

t1 = []
t2 = []
'''
V = np.linspace(1,50,300)
I = 0
for v in V:
    
    if I%10 == 0:
        print(I)
    
    #A_d = v/u_d

    A_0 = np.array([A_a, A_b, A_c, A_d*10/v])*v
    
    a,b,c,d = A_0
    
    T1 = PERIOD(Triad_a, Amp_change(Triad_a, np.array([c,b,a]) ))
    T2 = PERIOD(Triad_b, Amp_change(Triad_b, np.array([d,b,a]) ))
    t1 += [T1]
    t2 += [T2]

    e, i,p1,p2 = four_waves_integration(Four_Waves,Triad_a, A_0, t_0, t_f, h, plot = False)
    
    P1 += [p1]
    P2 += [p2]
    E  += [e]
    
    e1 = Triad_dynamics(Triad_a, np.array([c,b,a]), t_0, t_f = 1.1*T1, h=0.1,p=False)
    e2 = Triad_dynamics(Triad_b, np.array([d,b,a]), t_0, t_f = 1.1*T2, h=0.1,p=False)
    E1 += [e1]
    E2 += [e2]
    I += 1

h = np.array(P1) * t1 / (2*np.pi)

plt.plot(V, P1, label = 'P_abd')
plt.plot(V, P2, label  ='P_abc')
plt.plot(V, E, label = 'eff')
plt.legend()
plt.show()

plt.plot(V, np.array(t1)/(4*np.pi) , label = 'period_1')
plt.plot(V, np.array(t2)/(4*np.pi) , label = 'period_2')
#plt.ylim([0, 30])
#plt.xlim([20,50])
plt.legend()
plt.show()

plt.plot(V, E1, label = 'E_1')
plt.plot(V, E2, label  ='E_2')
plt.plot(V, E, label = 'eff')
plt.legend()
plt.show()
#plt.plot(V, ii, label = 'ratio')
'''


#t_f = 1000
#four_waves_integration(Four_Waves,Triad_a, AA, t_0, t_f, h, plot = True)
#four_waves_integration(Four_Waves,Triad_a, AB, t_0, t_f, h, plot = True)








'''PRECESSION'''
def four_waves_precession(Four_Waves, t_0, t_f, h):
    
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
    PRECESSION_2 = []

    max_value = 0
    p = 10

    A_d = 10/np.real(u_d)
    A_c = 10/np.real(u_c)

    for beta in BETA:
        
        A_b = beta
        #A_c = 0.5*beta
        
        MEASURE_alpha    = []
        PRECESSION_alpha = []
        P_2 =[]
        power = []

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
            Y_c = Y[:, 2]
            Y_d = Y[:, 3]
            
            Phi_a = np.unwrap(np.angle(Y_a))
            Phi_b = np.unwrap(np.angle(Y_b))
            Phi_c = np.unwrap(np.angle(Y_c))
            Phi_d = np.unwrap(np.angle(Y_d))
            
            Y_d = Y_d * np.conj(Y_d)/Energy_0  # mode d
            
            measure = max(Y_d) - min(Y_d)
            #measure = np.var(Y_d)
            MEASURE_alpha += [ measure ]
            
            PHI_abd = Phi_a - Phi_b - Phi_d
            PHI_abc = Phi_a - Phi_b - Phi_c
            precession = (PHI_abd[-1] - PHI_abd[0])/t_f
            precession2 = (PHI_abc[-1] - PHI_abc[0])/t_f

            PRECESSION_alpha += [precession]
            P_2 += [precession2]
            
            if measure > max_value:
                max_value = measure
                p = abs(precession)
                Phi_am = Phi_a
                Phi_bm = Phi_b
                Phi_cm = Phi_c
                Phi_dm = Phi_d         
                A_max = np.array([A_a, A_b, A_c, A_d])
                Y_aa = Y_a
                Y_bb = Y_b
         
        MEASURE += [MEASURE_alpha]
        PRECESSION += [PRECESSION_alpha]
        PRECESSION_2 += [P_2]

    MEASURE = np.array(MEASURE)
    PRECESSION = np.array(PRECESSION)
    PRECESSION_2 = np.array(PRECESSION_2)
    
    max_a = norm_component(Four_Waves.uvh_a[0])*A_max[0]*np.sqrt(g*h_e)
    max_b = norm_component(Four_Waves.uvh_b[0])*A_max[1]*np.sqrt(g*h_e)
    max_c = norm_component(Four_Waves.uvh_c[0])*A_max[2]*np.sqrt(g*h_e)
    max_d = norm_component(Four_Waves.uvh_d[0])*A_max[3]*np.sqrt(g*h_e)
    
    print('------------')
    print(f' Rossby variation      = {100*np.real(max_value)}%')
    print(f' Mode a zonal velocity = {np.real(max_c)}m/s')
    print(f' Mode b zonal velocity = {np.real(max_b)}m/s')
    print(f' Mode c zonal velocity = {np.real(max_a)}m/s')
    print(f' Rossby zonal velocity = {np.real(max_d)}m/s')
    print('------------')

    levels = np.linspace(0,10,11)

    fig, ax = plt.subplots()

    cs = ax.contourf(A*u_a,B*u_b, 100*MEASURE, levels=levels, cmap = 'terrain')
    plt.xlabel(r'Mode c zonal velocity (m/s)')
    plt.ylabel(r'Mode b zonal velocity (m/s)')
    plt.title(r'Efficiency of Energy Transfer (%)')
    plt.xlim(1,100)
    #RdYlBu
    #terrain
    #gist_ncar
    fig.colorbar(cs)
    plt.show()   

    t_f = 150
    four_waves_integration(Four_Waves, A_max, t_0, t_f, h=0.01)


    # PRESSETION RESONANCE GRAPH
    levels = np.linspace(-0.9,0.0,31)

    fig, ax = plt.subplots()

    cs = ax.contourf(A*u_a,B*u_b, (PRECESSION), levels=levels, cmap = 'terrain')
    plt.xlabel(r'Mode c zonal velocity (m/s)')
    plt.ylabel(r'Mode b zonal velocity (m/s)')
    plt.title(r'Precession Reonance Frequency - Triad 2')
    plt.xlim(1,100)
    #RdYlBu
    #terrain
    #gist_ncar
    fig.colorbar(cs)
    plt.show()   

    levels = np.linspace(0.5,1.02,31)

    fig, ax = plt.subplots()

    cs = ax.contourf(A*u_a,B*u_b, (PRECESSION_2), levels=levels, cmap = 'terrain')
    plt.xlabel(r'Mode c zonal velocity (m/s)')
    plt.ylabel(r'Mode b zonal velocity (m/s)')
    plt.title(r'Precession Reonance Frequency - Triad 1')
    plt.xlim(1,100)
    #RdYlBu
    #terrain
    #gist_ncar
    fig.colorbar(cs)
    plt.show()  
    
    return