import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from Eigenvalues_and_eigenvectors.Matrix_m0 import matriz_C
from Eigenvalues_and_eigenvectors.Matrix_m0 import matriz_D
from Eigenvalues_and_eigenvectors.Eigenvector_0 import Hough_harmonic


def norm_Hough(n,alpha,gamma, N,deg):
    
    point, weight = np.polynomial.legendre.leggauss(deg)
    
    ang = np.pi/2 * point
    
    U   = []
    V   = []
    Z   = []
    DU  = []
    DV  = []
    DZ  = []
    
    for phi in ang:
        
        u,v,z,du,dv,dz = Hough_harmonic( n, alpha, gamma, phi, N)
        
        U   += [u]
        V   += [v]
        Z   += [z]
        
        DU  += [du]
        DV  += [dv]
        DZ  += [dz]
        
    U   = np.array(U)
    V   = np.array(V)
    Z   = np.array(Z)
    DU  = np.array(DU)
    DV  = np.array(DV)
    DZ  = np.array(DZ) 
    cos = np.cos(ang)
 
    norm = np.pi/2* sum(weight * (U*U + V*V + Z*Z)*cos)
    
    U = 1/np.sqrt(norm) * U
    V = 1/np.sqrt(norm) * V
    Z = 1/np.sqrt(norm)*Z
    
    DU = 1/np.sqrt(norm)*DU
    DV = 1/np.sqrt(norm)*DV
    DZ = 1/np.sqrt(norm)*DZ
    
    return U,V,Z,DU, DV, DZ, point, norm

m = 0
n = 3
alpha = 1
gamma = 1/np.sqrt(100)

U,V,Z,DU, DV, DZ, ANG,norm = norm_Hough(n, alpha, gamma, 30, 300)

print(norm)
aa = np.arcsin(ANG)
#print('aa = ',aa)

U_1 = []
V_1 = []
Z_1 = []
DU_1 = []
DV_1 = []
DZ_1 = []

for phi in aa:
    
    u_1,v_1,z_1,du1, dv1, dz1 = Hough_harmonic(n,alpha,  gamma, phi, 30)
    
    U_1 += [u_1]    
    V_1 += [v_1]
    Z_1 += [z_1]  
    
    DU_1 += [du1]
    DV_1 += [dv1]
    DZ_1 += [dz1]
    
U_1 = 1/np.sqrt(norm) * np.array(U_1)
V_1 = 1/np.sqrt(norm) * np.array(V_1)
Z_1 = 1/np.sqrt(norm) * np.array(Z_1)

DU_1 = 1/np.sqrt(norm) * np.array(DU_1)
DV_1 = 1/np.sqrt(norm) * np.array(DV_1)
DZ_1 = 1/np.sqrt(norm) * np.array(DZ_1)


print('Aprox. Primeira Ordem')
x = (U_1[1:] - U_1[:-1])/(aa[1:] - aa[:-1])
ang = np.arctan(x)*180/np.pi
print(x)
print()

print('U_1 = ', np.arctan(DU_1)*180/np.pi)
    

#print('U = ',U[29:])
#print('V = ', V[40:55])
#print('Z = ', Z[40:55])


ANG = 90*ANG
print()
#print(ANG)

plt.plot( ANG,-U,label = r'$U$')  # change here to plot U, V or Z
plt.plot( ANG,-V,label = r'$V$')
plt.plot( ANG,Z,label = r'$Z$')

plt.axvline(x = -1, color='grey', lw='1', linestyle='--')
plt.axvline(x = 1, color='grey', lw='1', linestyle='--')  
plt.axvline(x = -0.5, color='grey', lw='1', linestyle='--')
plt.axvline(x = 0.5, color='grey', lw='1', linestyle='--') 
plt.axvline(x = 0, color='grey', lw='1', linestyle='--') 
plt.axhline(y = 30, color='grey', lw='1', linestyle='--') 
plt.axhline(y = 60, color='grey', lw='1', linestyle='--') 

plt.xlim([0,90])
plt.ylim([-1.5,1.5])
plt.legend()
plt.title('Height (Z)')
plt.show()