import scipy
import numpy as np
import matplotlib.pyplot as plt
from rsw_sphere.dynamics.dynamic_triads import TRIAD, RK33
from rsw_sphere.hough_harmonics.normalization import norm_component
from rsw_sphere.dynamics.dynamic_triads import Energy_0

def UU(A_0):
    
    return (abs(A_0[0])**2 + abs(A_0[1])**2)/2

def J(A_0):
    
    return np.sqrt( UU(A_0)*UU(A_0) - (abs(A_0[0])*abs(A_0[1]))**2 )

def E(A_0):
  
    return abs(A_0[2])**2 + UU(A_0)

def Hamiltonian(A_0):
    
    A_a, A_b, A_c = A_0
    
    B_a = abs(A_a)
    B_b = abs(A_b)
    B_c = abs(A_c)
    Phi = np.angle(A_a) + np.angle(A_b) - np.angle(A_c)
    
    return np.imag(A_a * A_b * np.conj(A_c))
    
    return B_a * B_b * B_c * np.sin(Phi)

def P(Triad, A_0):
    
    d = Triad.mismatch/2
    e = E(A_0)
    j = J(A_0)
    g = Hamiltonian(A_0)
    u = UU(A_0)
    
    value = -1/3 * ((d*d - e)**2 + 3*(2*d*d*u -2*d*g + j*j))
    
    return value

def Q(Triad, A_0):
    
    d = Triad.mismatch/2
    e = E(A_0)
    j = J(A_0)
    g = Hamiltonian(A_0)
    u = UU(A_0)
    
    value  = (d*d - e)/27 * (2*(d*d - e)**2 +9* (2*d*d*u - 2*d*g + j*j))
    value += e*j*j
    value += (g - d*u)**2
    
    return value

def D_theta(Triad, A_0):
    
    p = P(Triad, A_0)
    q = Q(Triad, A_0)
    
    value = -1/2 * q * (3/abs(p))**(3/2)
    return value, np.arccos(value)

def M(Triad, A_0):
    
    d, ang = D_theta(Triad, A_0)
    
    value  = np.cos(ang/3 + np.pi/6)
    value /= np.cos(ang/3 - np.pi/6)
    
    return value

def PERIOD(Triad, A_0):
    
    m = float(M(Triad, A_0))
    p = P(Triad, A_0)
    q = Q(Triad, A_0)
    d, ang = D_theta(Triad, A_0)
    cos = np.cos(ang/3 - np.pi/6)
    
    
    r = np.roots([1,0,p,q])
    #print('roots = ', r)
    if abs(d) > 1:
        
        print()
        print()
        print('----------------------------------------------------')
        print('The discriminant has absolute value higher than 1!')
        print('There are not three real roots!')
        print('D = ', d)
        print('----------------------------------------------------')
        
        return 0
    
    elif p > 0:
        
        print()
        print()
        print('----------------------------------------------------')
        print('p is not non positive!')
        print('There are not three real roots!')
        print('p = ', p)
        print('----------------------------------------------------')
        
        #return 0
        
    
    value  = np.sqrt(2)*scipy.special.ellipk(m)
    value /=  ( (abs(p))**(1/4)*np.sqrt(cos) )
    
    return value

def Amp_change(Triad, A_0):
    
    A_a, A_b, A_c = A_0
    
    a = np.real(Triad.coef_ABC)
    b = np.real(Triad.coef_BAC)
    c = np.real(Triad.coef_CAB)
    
    aa = A_a*np.sqrt(Triad.coef_BAC*Triad.coef_CAB)
    bb = A_b*np.sqrt(Triad.coef_ABC*Triad.coef_CAB) 
    cc = A_c*np.sqrt(Triad.coef_ABC*Triad.coef_BAC)
    
    if (a>0 and b>0 and c>0) or (a<0 and b<0 and c<0): #case I
        
        A = np.array([aa,bb,cc])
        
        return -1j*A
    
    elif (a<0 and b>0 and c<0) or (a>0 and b<0 and c>0): #case II
        
        A = -1j*np.array([bb,np.conj(cc),-np.conj(aa)])
        
        return A
    
    elif (a<0 and b>0 and c>0) or (a>0 and b<0 and c<0): #case III
        
        A = -1j*np.array([aa,np.conj(cc),-np.conj(bb)])
        
        return A
    
    elif (a>0 and b>0 and c<0) or (a<0 and b<0 and c>0): # case IV
    
        print('Case IV')
        return 
    
    

def Triad_dynamics_T(Triad, A_0, t_0, t_f, h):
    
    tt = PERIOD(Triad, Amp_change(Triad, A_0))/ (4*np.pi)

    #t_f = float(2*tt*(4*np.pi))

    
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
    
    
    plt.plot(t, E_a, label = 'mode a')
    plt.plot(t, E_b, label = 'mode b')
    plt.plot(t, E_c, label = 'mode c')
    #plt.plot(t, E_2/E_0, label = r'$E^{(2)}$')
    #plt.plot(t, E_3/E_0, label = r'$E^{(3)}$')
    plt.plot(t, E_T, label = 'Total Energy')
    plt.xlabel('Time (days)')
    plt.ylabel(' Energy (nondimensional)')
    #plt.xlim([0, min(2*tt,t_f/(4*np.pi))])
    plt.axvline(x = float(tt), color='red', lw='1', linestyle='--')
    #plt.legend(bbox_to_anchor=(1.05, 1),
                         #loc='upper left', borderaxespad=0.)
    plt. legend (loc='upper right')
    plt.show()
    
    print('T = ', tt)
    
def period_graph(Triad, A_0, u_a):
    
    A_a, A_b, A_c = A_0
    
    alpha_1 = 1/np.real(u_a)
    alpha_100 = 50/np.real(u_a)  # velocities 1m/s --- 100m/s  mode a
    
    ALPHA = np.linspace(alpha_1, alpha_100, 50)
    
    P = []
    i = 0
    for alpha in ALPHA:
        
        A_a = alpha
        A0  = np.array([alpha, A_b, A_c])
        p = PERIOD(Triad, Amp_change(Triad, A0))
        
        P += [p]
        
        if i % 13 == 0:
            print(f'{i} : {p/(4*np.pi)}')
            Triad_dynamics_T(Triad, A0, t_0=0, t_f=300, h=0.01)
            
        i += 1
    P = np.array(P)
    P = P/(4*np.pi)
    
    plt.plot(ALPHA*u_a*2, P, color = 'blue')
    plt.xlabel(f'Meridional Velocity - {Triad.label_a}  (m/s)')
    plt.ylabel('Period (days)')
    plt.scatter(20, P[9], color = 'red')
    plt.scatter(60, P[30], color = 'b')
    plt.annotate(f'Example 2', (20, P[10]), textcoords="offset points", 
                 xytext=(30, 10), ha='center')
    plt.annotate(f'Example 3', (60, P[30]), textcoords="offset points", 
                 xytext=(30, 10), ha='center')
    plt.legend()
    plt.show()
    
    print()
    print('indice :', np.argmax(P))
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
t_f = 100
h   = 0.01

m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,2,2, 2,3,2, 3,5,2  # WWW
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,2,3, 1,11,3, 2,8,3 # RRR
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 5,11,3, 7,15,3, 12,13,3 # resonant
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,3,3, 3,7,3, 4,5,3 # RRR 
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 2,2,3, 6,8,1, 8,9,1 # 

#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,7,3, 6,9,1, 7,9,1 # REE
#m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 3,11,3, 12,15,1, 15,15,1 # REE
m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c =  1,4,3, 3,4,3, 4,5,3

m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,1,1, 3,4,3, 4,5,3
m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 2,3,3, 2,4,1, 4,4,1
m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c = 1,7,3, 6,9,1, 7,9,1

Triad = TRIAD(gamma, m_a, n_a, alpha_a, m_b, n_b, 
             alpha_b, m_c,n_c, alpha_c, N, deg)   

u_a = norm_component(Triad.uvh_a[0])*np.sqrt(g*h_e)
u_b = norm_component(Triad.uvh_b[0])*np.sqrt(g*h_e)
u_c = norm_component(Triad.uvh_c[0])*np.sqrt(g*h_e)

A_a = 10/u_a
A_b = 10/u_b
A_c = 0/u_c
    
A_0 = np.array([A_a, A_b, A_c])

#period_graph(Triad, A_0, u_a)
#Triad_dynamics_T(Triad, A_max, t_0, t_f, h)

print('A:', Triad.coef_ABC)
print('B:', Triad.coef_BAC)
print('C:', Triad.coef_CAB)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
