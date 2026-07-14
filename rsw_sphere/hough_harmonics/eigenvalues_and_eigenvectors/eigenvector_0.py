import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.matrix_m0 import matriz_C
from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.matrix_m0 import matriz_D
from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.matrix_m0 import p
from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.matrix_m0 import r

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
    
    DD = matriz_D(gamma, N)
    
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

def norm_Pmn(i ,f):
    
    n = np.arange(i, f+1)
    
    num = (2*n+1)
    den = 2
    
    norm = np.sqrt(num/den)
    
    return norm
    
    return norm
    
def Spherical_vector_harmonics(N, phi):
    
    m = 0
    
    # For a fixed zonal wavenumber m, the derivative of the normalized
    # associated Legendre polynomials depend on P_m+1,n and P_m-1,n
    # Therefore, we considered the associated Legendre polynomials to order m+1
    
    P_mn, dP_mn = scipy.special.lpmn(m+1,m + 2*(N-1) +1, np.sin(phi))
    
    Pmn   = np.copy(P_mn[m][m:])      # P_mn from n = m to n = m + 2*(N-1) + 1 
    dPmn  = np.copy(dP_mn[m][m:])*np.cos(phi)
    dPm1n = np.copy(dP_mn[m+1][m:])*np.cos(phi)
    
    # Normalizing the Legendre polynomials
    
    k = np.arange(2*(N-1) + 2)
    norm_0 = np.sqrt((2*k+1)/2)
    norm_1 = np.sqrt((2*k+1)/(2*k*(k+1)))
    
    #norm_1 = np.sqrt((2*k+1)/2 * (k*(k+1)))
    norm_1[0] = 0


    Pmn   *= norm_0
    dPmn  *= norm_0
    dPm1n *= norm_1
    #print(dPm1n)
    

    # DERIVATIVE
    
    k = np.arange(2*(N-1) +2)
    
    fat3 = np.sqrt(1/(k*(k+1)))
    fat3[0] = 0

    
    y_1 = np.copy(dPmn)
    y_1 = fat3 * y_1

    y_2 = -np.copy(dPmn)
    y_2 = fat3 * y_2
    
    y_3 = np.copy(Pmn)
    
    dy_1 = np.copy(dPm1n)
    dy_2 = -np.copy(dPm1n)
    dy_3 = np.copy(dPmn)
    
    return y_1, y_2, y_3, dy_1,dy_2,dy_3

def symetry(n,alpha):
       
        if (n)%2 == 0:
            
            return True
    
        return False


def Hough_harmonic(n,alpha, gamma,phi,N):
    
    sym = symetry(n,alpha)

    if sym:
        
        AA,BB,CC = Hough_coef_C(n,alpha, gamma, N)
        y_1, y_2, y_3, dy_1,dy_2,dy_3 = Spherical_vector_harmonics(3*N, phi)
        
    
        A = 1j*AA*y_2[::2]
        B = BB*y_1[1:-2:2]
        C = -CC*y_3[::2]
        
        U =  sum(B)
        V = -1j * (sum(A))
        Z = sum(C)
        
        DA = 1j*AA*dy_2[::2]
        DB = BB*dy_1[1:-2:2]
        DC = -CC*dy_3[::2]
        
        DU =  sum(DB)
        DV = -1j * (sum(DA))
        DZ = sum(DC)
        
        return U,V,Z, DU, DV, DZ
    
    else:
        
        AA,BB,CC = Hough_coef_D(n,alpha, gamma, N)
        y_1, y_2, y_3, dy_1,dy_2,dy_3 = Spherical_vector_harmonics(3*N, phi)
        
        A = 1j*AA*y_2[1::2]
        B = BB*y_1[2::2]
        C = -CC*y_3[1::2]
        
        U =  sum(B)
        V = -1j * (sum(A))
        Z = sum(C)
        
        DA = 1j*AA*dy_2[1::2]
        DB = BB*dy_1[2::2]
        DC = -CC*dy_3[1::2]
        
        DU =  sum(DB)
        DV = -1j * (sum(DA))
        DZ = sum(DC)
        
        return U,V,Z, DU, DV,DZ