import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from Eigenvalues_and_eigenvectors.Matrix_system import matriz_A
from Eigenvalues_and_eigenvectors.Matrix_system import matriz_B

def Hough_coef_A(m,n,alpha, gamma, N):
    
    AA = matriz_A(m, gamma, N)
    
    #value_A, vector_A = np.linalg.eig(AA)
    value_A, vector_A = scipy.linalg.eig(AA)
    
    value_order = np.sort(value_A)
    
    l = n-m
    
    if alpha == 1: # EASTWARD GRAVITY MODES 
        
        index = 2*N + l//2
    
    elif alpha == 2: # WESTWARD GRAVITY MODES
        
        index = N-1 - l//2
    
    else: # ROTATIONAL MODES
        
        index = N + l//2
        
    eigen = value_order[index]
    j = np.where(value_A == eigen)
    
    index = j[0][0]
        
    X = vector_A[:,index]
    
    #if X[0] < 0:
    #   X = -X
    
    A = X[1::3]
    B = X[2::3]
    C = X[::3]
    
    return A,B,C, eigen

def Hough_coef_B(m,n,alpha, gamma, N):
    
    BB = matriz_B(m, gamma, N)
    
    #value_B, vector_B = np.linalg.eig(BB)
    value_B, vector_B = scipy.linalg.eig(BB)
    
    value_order = np.sort(value_B)
    
    l = n-m
    
    if alpha == 1: # EASTWARD GRAVITY MODES 
        
        index = 2*N + l//2
    
    elif alpha == 2: # WESTWARD GRAVITY MODES
        
        index = N-1 - l//2
    
    else: # ROTATIONAL MODES
        
        index = N + l//2
        
    eigen = value_order[index]
    j = np.where(value_B == eigen)
    
    index = j[0][0]
        
    X = vector_B[:,index]
    
    A = X[2::3]
    B = X[::3]
    C = X[1::3]
    
    return A,B,C, eigen

#-----------------------------------------------------------------------
# We now start the construction of the spherical vector harmonics

def norm_Pmn(m, i ,f): # norm of the associated Legendre polynomials
    
    zero = np.array([])
    
    j = m-i
    
    if j > 0:
        
        zero = np.zeros(j)
    
    n = np.arange(max(m,i), f+1)
    
    num = (2*n+1)*scipy.special.factorial(n-m)
    den = 2*scipy.special.factorial(n+m)
    
    norm = np.sqrt(num/den)
    
    norm = np.concatenate((zero, norm))
    
    return norm

def important_factor(m,i, N):
    
    zero = np.array([])
    
    j = m-i
    
    if j > 0:
        
        zero = np.zeros(j)
    
    n = np.arange(max(i,i+j), i + 2*(N-1) + 2)
    
    fat1 = np.concatenate((zero, np.sqrt( (n-m)*(n+m+1) )))
    fat6 = np.concatenate((zero, np.sqrt( (n-m)*(n-m-1) )))
    fat2 = np.concatenate((zero, np.sqrt( (n+m)*(n-m+1) )))
    
    n = np.arange(i,i+2*(N-1) + 2 )
    
    fat3 =  1/np.sqrt( n*(n+1) )
    fat4 = np.sqrt( (2*n + 1)/(2*n-1) )
    fat5 = np.sqrt( (n+m)*(n+m-1) )
    
    return fat1, fat2, fat3, fat4, fat5, fat6
    
def Pmn_and_derivative(m,N, phi):
     
    # For a fixed zonal wavenumber m, the derivative of the normalized
    # associated Legendre polynomials depend on P_m+1,n and P_m-1,n
    # Therefore, we considered the associated Legendre polynomials to order m+1
    
    P_mn, dP_mn = scipy.special.lpmn(m+2,m + 2*(N-1) +1, np.sin(phi))
    
    Pmn  = np.copy(P_mn[m][m-1:])      # P_mn from n = m-1 to n = m + 2*(N-1) + 1
    Pm1n = np.copy(P_mn[m+1][m-1:])    # P_m+1,n 
    P1mn = np.copy(P_mn[m-1][m-1:])    # P_m-1,n
    Pm2n = np.copy(P_mn[m+2][m-1:])    # P_m+2,n 
    
    dPmn  = np.copy(dP_mn[m][m-1:])*np.cos(phi)
    dPm1n = np.copy(dP_mn[m+1][m-1:])*np.cos(phi)
    dP1mn = np.copy(dP_mn[m-1][m-1:])*np.cos(phi)
    
    # Normalizing the Legendre polynomials
    
    norm_0 = norm_Pmn(m  ,m-1,m + 2*(N-1) +1)
    norm_1 = norm_Pmn(m+1,m-1,m + 2*(N-1) +1)
    norm_2 = norm_Pmn(m-1,m-1,m + 2*(N-1) +1)
    norm_3 = norm_Pmn(m+2,m-1,m + 2*(N-1) +1)
    
    Pmn  *= norm_0
    Pm1n *= norm_1
    P1mn *= norm_2
    Pm2n *= norm_3
    
    dPmn  *= norm_0
    dPm1n *= norm_1
    dP1mn *= norm_2
    
    if m == 1:
        
        P2mn = - np.copy(Pmn)
    
    else:
        
        P2mn = np.copy(P_mn[m-2][m-1:])    # P_m-2,n
        norm_4 = norm_Pmn(m-2,m-1,m + 2*(N-1) +1)
        P2mn *= norm_4
        
    
    # DERIVATIVE
    # Initialy we calculate the square root factors
    # We also use the loop below to calculate the factor sqrt( n(n+1) ), which
    # will be used in the definition of y_1, y_2, and y_3
    
    fat1, fat2, fat3, fat4, fat5, fat6 = important_factor(m,m,N)

    # The derivative is then 1/2 ( fat1 * P_m+1,n - fat2 * P_m-1,n)
    DP_mn = 1/2*( fat1 * np.copy(Pm1n[1:])- fat2 * np.copy(P1mn[1:])) 
    
    # The term involving cos(phi) is calculated below
    cosP_mn = 1j/2 *  fat4 * (fat5 * P1mn[:-1] + fat6 * Pm1n[:-1])
    
    # SECOND DERIVATIVE 
    # We will need DP_m+1,n and DP_m-1 to apply the recurence formula
    
    fat11, fat22, fat33, fat44, fat55, fat66 = important_factor(m+1,m-1,N+0.5)
    
    DP_m1n = 1/2*( fat11 * np.copy(Pm2n)- fat22 * np.copy(Pmn)) 
    DP_m1n[0] = 0
    DP_m1n[1] = 0
    
    fat11, fat22, fat33, fat44, fat55, fat66 = important_factor(m-1,m-1,N+0.5)
    DP_1mn = 1/2*( fat11 * np.copy(Pmn)- fat22 * np.copy(P2mn)) 
    
    # D^2 P_mn
    D2P_mn = 1/2 * (fat1 * DP_m1n[1:] - fat2 * DP_1mn[1:])
    
    #(D_cos)
    D_cos = 1j/2 * fat4 * (fat5 * DP_1mn[:-1] + fat6 * DP_m1n[:-1])
 
    #return Pmn[1:],-DP_mn, -cosP_mn, D2P_mn, D_cos, fat3
    return Pmn[1:],DP_mn, cosP_mn, D2P_mn, D_cos, fat3
    
def Spherical_vector_harmonics(m,N, phi):
    
    # With the previous calculus, we are able to define the sphercial vector
    # harmonics
    
    Pmn,DP_mn, cosP_mn, D2P_mn, D_cos, fat3 = Pmn_and_derivative(m, N, phi)
    
    y_1 = [np.copy(cosP_mn) , np.copy(DP_mn)]
    y_1 = np.array(y_1)
    y_1 = fat3 * y_1

    y_2 = [-np.copy(DP_mn) , np.copy(cosP_mn)]
    y_2 = np.array(y_2)
    y_2 = fat3 * y_2
    
    y_3 = np.copy(Pmn)
    
    # Derivative
    
    dy_1 = [np.copy(D_cos) , np.copy(D2P_mn)]
    dy_1 = np.array(dy_1)
    dy_1 = fat3 * dy_1

    dy_2 = [-np.copy(D2P_mn) , np.copy(D_cos)]
    dy_2 = np.array(dy_2)
    dy_2 = fat3 * dy_2
    
    dy_3 = np.copy(DP_mn)
    
    return y_1, y_2, y_3, dy_1, dy_2, dy_3

def symetry(m,n,alpha):
    
    if alpha == 3:
        
        if (m-n)%2 == 1:
        
            return True
        
        return False
    
    else:
        
        if (m-n)%2 == 0:
            
            return True
    
        return False


def Hough_harmonic(m,n,alpha, gamma,phi,N):
    
    sym = symetry(m,n,alpha)

    if sym:
        
        AA,BB,CC, eigen = Hough_coef_A(m,n,alpha, gamma, N)
        y_1,y_2,y_3,dy_1, dy_2, dy_3  = Spherical_vector_harmonics(m,N, phi)
        
        A = 1j*np.copy(AA)*y_1[:,::2]
        B = np.copy(BB)*y_2[:,1::2]
        C = -np.copy(CC)*y_3[::2]
        
        DA = 1j*AA*dy_1[:,::2]
        DB = BB*dy_2[:,1::2]
        DC = -CC*dy_3[::2] 
        
        U = sum(A[0]) + sum(B[0])
        V = -1j * (sum(A[1]) + sum(B[1]))
        Z = sum(C)
        
        DU = sum(DA[0]) + sum(DB[0])
        DV = -1j * (sum(DA[1]) + sum(DB[1]))
        DZ = sum(DC)
        
        return U,V,Z, DU, DV, DZ, eigen
    
    else:
        
        AA,BB,CC, eigen = Hough_coef_B(m,n,alpha, gamma, N)
        y_1,y_2,y_3,dy_1, dy_2, dy_3  = Spherical_vector_harmonics(m,N, phi)
        
        A = 1j*AA*y_1[:,1::2]
        B = BB*y_2[:,::2]
        C = -CC*y_3[1::2]
        
        DA = 1j*AA*dy_1[:,1::2]
        DB = BB*dy_2[:,::2]
        DC = -CC*dy_3[1::2]

        U = sum(A[0]) + sum(B[0]) 
        V = -1j * (sum(A[1]) + sum(B[1]))
        Z = sum(C)
        
        DU = sum(DA[0]) + sum(DB[0])
        DV = -1j * (sum(DA[1]) + sum(DB[1]))
        DZ = sum(DC)
        
        return U,V,Z, DU, DV, DZ, eigen

    
''' EXAMPLES'''
'''
PHI = np.linspace(0,np.pi/2,100)
ANG = np.linspace(0,90, 100)

U_1 = []
V_1 = []
Z_1 = []

for phi in PHI:

    g = 1/np.sqrt(10) # gamma = epsilon^-1/2
    
    m = 1
    n = 2
    alpha = 1 # mode = 1 Eastward gravity, = 2 Westward gravity, = 3 rotational
    
    u_1,v_1,z_1 = Hough_harmonic(m,n,alpha,  g, phi, 30)
    
    U_1 += [-3*u_1]    # WHAT IS THE CONSTANT ?!?
    V_1 += [3*v_1]
    Z_1 += [4*z_1]        

plt.plot(U_1, ANG,label = r'$U$')  # change here to plot U, V or Z
plt.plot(V_1, ANG,label = r'$V$')
plt.plot(Z_1, ANG,label = r'$Z$')

plt.axvline(x = -1, color='grey', lw='1', linestyle='--')
plt.axvline(x = 1, color='grey', lw='1', linestyle='--')  
plt.axvline(x = 0, color='grey', lw='1', linestyle='--') 

plt.ylim([0,90])
plt.xlim([-2,2])
plt.legend()
plt.title('Height (Z)')
plt.show()
   '''     