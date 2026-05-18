import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from Eigenvalues_and_eigenvectors.Matrix_m0 import matriz_C
from Eigenvalues_and_eigenvectors.Matrix_m0 import matriz_D
from Eigenvalues_and_eigenvectors.Matrix_m0 import p
from Eigenvalues_and_eigenvectors.Matrix_m0 import r

def Hough_coef_C(n,alpha, gamma, N):
    
    # A_0, A_2, ... , A_2(N-1)
    # C_0, C_2, ... , C_2(N-1)
    # B_1, B_3, ..., B_2(N-1)-1
    
    CC = matriz_C( gamma, N)
    
    value_C, vector_C = np.linalg.eig(CC)
    
    value_C = np.sqrt(value_C)
    
    value_order = np.sort(value_C)
        
    eigen = value_order[2*N + n//2]  # it is just necessary to consider 
                                     # eastward waves
  
    j = np.where(value_C == eigen)

    index = j[0][0]
        
    A = vector_C[:,index]
    
    #------
    
    k = np.arange(6*N-1)
    
    p_n = p(k)
    r_n = r(k,gamma)
    
    #-------
    
    if eigen != 0:
        
        C = 1/eigen * r_n[::2] * A
        B = 1/eigen * (p_n[1:-1:2] * A[:-1] + p_n[2::2]*A[1:])
        
    
    return A,B,C

def Hough_coef_D(n,alpha, gamma, N):
    
    # A_1, A_3, A_5, ... , A_2(N-1) + 1
    # C_1, C_3, C_5, ... , C_2(N-1) + 1
    # B_2, B_4, B_6, ... , B_2(N-1)
    
    DD = matriz_C( gamma, N)
    
    value_D, vector_D = np.linalg.eig(DD)
    
    value_D = np.sqrt(value_D)
    
    value_order = np.sort(value_D)
    
    eigen = value_order[2*N + n//2]
    
    j = np.where(value_D == eigen)
    
    index = j[0][0]
        
    A = vector_D[:,index]
    
    #------
    
    k = np.arange(6*N)
    
    p_n = p(k)
    r_n = r(k,gamma)
    
    #-------
    
    if eigen != 0:
        
        C = 1/eigen * r_n[1::2] * A
        B = 1/eigen * (p_n[2::2] * A[:-1] + p_n[3::2]*A[1:])
        
    return A,B,C

def norm_Pmn(m, i ,f):
    
    norm = []
    
    for n in range (i,f+1):
        
        if n-m < 0:
            norm += [0]
        
        else:
            num = (2*n+1)*np.math.factorial(n-m)
            den = 2*np.math.factorial(n+m)
        
            norm += [np.sqrt(num/den)]
    
    norm = np.array(norm)
    
    return norm
    
def Spherical_vector_harmonics(N, phi):
    
    m = 0
    
    # For a fixed zonal wavenumber m, the derivative of the normalized
    # associated Legendre polynomials depend on P_m+1,n and P_m-1,n
    # Therefore, we considered the associated Legendre polynomials to order m+1
    
    P_mn, dP_mn = scipy.special.lpmn(m+1,m + 2*(N-1) +1, np.sin(phi))
    
    Pmn  = np.copy(P_mn[m][m:])      # P_mn from n = m to n = m + 2*(N-1) + 1
    Pm1n = np.copy(P_mn[m+1][m:])    # P_m+1,n 
    
    dPmn  = np.copy(dP_mn[m][m:])*np.cos(phi)
    dPm1n = np.copy(dP_mn[m+1][m-1:])*np.cos(phi)
    dP1mn = np.copy(dP_mn[m-1][m-1:])*np.cos(phi)
    
    
    #----------- extra parameters (an alternative for the im/cos phi P_mn)
    Q1 = np.copy(P_mn[m+1][m:-1])
    Q1 = np.concatenate((np.zeros(1), Q1))

    
    normQ1 = norm_Pmn(m+1,m-1,m+2*(N-1))

    Q1 *= normQ1
 

    
    # Normalizing the Legendre polynomials
    
    norm_0 = norm_Pmn(m,m,m + 2*(N-1) +1)
    norm_1 = norm_Pmn(m+1,m,m + 2*(N-1) +1)

    Pmn  *= norm_0
    Pm1n *= norm_1
    
    dPmn *= norm_0


    
    
    # DERIVATIVE
    # Initialy we calculate the square root factors
    # We also use the loop below to calculate the factor sqrt( n(n+1) ), which
    # will be used in the definition of y_1, y_2, and y_3
    
    fat1 = []
    fat2 = []
    fat3 = []
    
    fat4 = [] # extra factors (an alternative)
    fat5 = []
    fat6 = []
    
    fat7 = []

    
    for n in range(m, m + 2*(N-1) + 2):
        
        fat1 += [ np.sqrt( (n-m)*(n+m+1) )]
        fat2 += [ np.sqrt( (n+m)*(n-m+1) )]
        fat3 += [ 1/np.sqrt( n*(n+1) )]
        
        fat4 += [np.sqrt( (2*n + 1)/(2*n-1) )]
        fat5 += [np.sqrt( (n+m)*(n+m-1) )]
        fat6 += [np.sqrt( (n-m)*(n-m-1) )]
        
        fat7 += [-1/((n+1)*(n+2))]
    
    fat1 = np.array(fat1)
    fat2 = np.array(fat2)
    fat3[0]=0
    fat3 = np.array(fat3)
    fat4 = np.array(fat4)
    fat5 = np.array(fat5)
    fat6 = np.array(fat6)
    
    
    fat7 = np.array(fat7)    
    fat7 = np.concatenate((np.zeros(1), fat7[:-1]))
    
    P1mn = fat7 * Pm1n
    Q2 = np.concatenate((np.zeros(1), fat7[:-1])) * Q1
    # The derivative is then 1/2 ( fat1 * P_m+1,n - fat2 * P_m-1,n)
    
    #DP_mn = 1/2*( fat1 * np.copy(Pm1n)- fat2 * np.copy(P1mn)) 
    #DP_mn[0] = 0
    #DP_mn = 1/2*(1 + fat7)*Pm1n
    #DP_mn = Pm1n
    #DP_mn = dP_mn[m][m:] * norm_0
    
    
    DP_mn = dPmn
    # The term involving cos(phi) is calculated below
    
    #cosP_mn = 1j * (m/np.cos(phi)) * np.copy(Pmn)
    
    cosP_mn = 1j/2 *  fat4 * (fat5 * Q2 + fat6 * Q1)
    cosP_mn[0] = 0
    
    # With the previous calculus, we are able to define the sphercial vector
    # harmonics
    
    y_1 = np.copy(DP_mn)
    #y_1 = np.array(y_1)
    if phi == 0:
        print('y_1 = ', y_1)
        print()
    y_1 = fat3 * y_1
    #print('soma = ',sum(y_1[0]))

    #y_2 = [-np.copy(DP_mn) , np.copy(cosP_mn)]
    #y_2 = np.array(y_2)
    y_2 = -np.copy(DP_mn)
    y_2 = fat3 * y_2
    
    y_3 = np.copy(Pmn)
    
    return y_1, y_2, y_3

def symetry(n,alpha):
       
        if (n)%2 == 0:
            
            return True
    
        return False


def Hough_harmonic(n,alpha, gamma,phi,N):
    
    sym = symetry(n,alpha)

    if sym:
        
        A,B,C = Hough_coef_C(n,alpha, gamma, N)
        y_1,y_2,y_3 = Spherical_vector_harmonics(3*N, phi)
        
    
        A = 1j*A*y_1[::2]
        B = B*y_2[1:-2:2]
        C = -C*y_3[::2]
        
        U =  sum(B)
        V = -1j * (sum(A))
        Z = sum(C)
        
        return U,V,Z
    
    else:
        
        A,B,C = Hough_coef_D(n,alpha, gamma, N)
        y_1,y_2,y_3 = Spherical_vector_harmonics(3*N, phi)
        
        A = 1j*A*y_2[1::2]
        B = B*y_1[2::2]
        C = -C*y_3[1::2]
        
        U =  sum(B)
        V = -1j * (sum(A))
        Z = sum(C)
        
        return U,V,Z