import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from .eigenvalues_and_eigenvectors.matrix_system import matriz_A
from .eigenvalues_and_eigenvectors.matrix_system import matriz_B
from .eigenvalues_and_eigenvectors.eigenvectors import Hough_harmonic
from .eigenvalues_and_eigenvectors.eigenvectors import symetry


def norm_Hough(m,n,alpha,gamma, N,deg):
    
    # Considering the inner product, here the norm of a Hough harmonic
    # is calculated, in order to define an orthonormal basis
    
    # Gaussian quadrature is used for the integral
    point, weight = np.polynomial.legendre.leggauss(deg)
    
    s = symetry(m,n,alpha)    
    
    # Interval [-1,1] is changed to [-pi/2, pi/2]
    ang = np.pi/2 * point
    
    U   = []
    V   = []
    Z   = []
    DU  = []
    DV  = []
    DZ  = []
    
    for phi in ang:
        
        # Calculating the fields and their derivatives in each latitude
        # for the quadrature
        u,v,z,du,dv,dz, eigen = Hough_harmonic(m, n, alpha, gamma, phi, N)
        
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
    
    # This is the integral, recalling that once the change of variables is
    # applyed, a factor of pi/2 must be considered
    norm = np.pi/2* sum(weight * (U*U + V*V + Z*Z)*cos)
    
    # Normalizing the fields
    U = 1/np.sqrt(norm)*U
    V = 1/np.sqrt(norm)*V
    Z = 1/np.sqrt(norm)*Z
    
    DU = 1/np.sqrt(norm)*DU
    DV = 1/np.sqrt(norm)*DV
    DZ = 1/np.sqrt(norm)*DZ
    
    return U,V,-Z,-DU,-DV,DZ,point, norm, eigen
    
    '''
    if s:
        
        return U ,V , -Z, -DU, -DV, DZ, point, norm, eigen
    
    else:
        
        return -U ,-V , Z, DU, DV, -DZ, point, norm, eigen
    '''
    
    if s:
        
        return -U ,-V , Z, DU, DV, -DZ, point, norm, eigen
    
    else:
        
        return U ,V , -Z, -DU, -DV, DZ, point, norm, eigen
    
def norm_component(u, deg = 300):
    
    # Norm of the zonal velocity component (u)
    # This norm is used to define the necessary amplitude to obtain a
    # specific zonal velocity.
    
    point, weight = np.polynomial.legendre.leggauss(deg)
    
    ang = np.pi/2 * point
    
    cos = np.cos(ang)

    norm = np.pi/2* sum(weight * (u*u)*cos)
    
    return np.sqrt(norm)
    
    
#------------------
# PLOTS
#------------------

def label(m,n,alpha,height):

    l = ''
    if alpha == 1:
        l += 'EIG'
    elif alpha == 2:
        l += 'WIG'
    else:
        l += 'RH'
    
    l += f'({m},{n}) at {height}m'
    return l

def hough_and_derivatives(m,n,alpha, h_e:int = 10000):

    l = label(m,n,alpha, h_e)
    g = 9.8
    a = 6.38e+06
    omega = 2*np.pi/24/60/60
    eps = (4*a*a*omega*omega)/(g*h_e)
    gamma = 1/np.sqrt(eps)

    U,V,Z,DU, DV, DZ, ANG,norm, eigen = norm_Hough(m, n, alpha, gamma, 10, 60)

    angle = np.pi/2*ANG

    U_1 = []
    V_1 = []
    Z_1 = []
    DU_1 = []
    DV_1 = []
    DZ_1 = []

    for phi in angle: 
        
        u_1,v_1,z_1,du1, dv1, dz1, eigen = Hough_harmonic(m,n,alpha,  gamma, phi, 30)
        
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

    
    ANG = 90*ANG
    
    plt.plot(ANG,U,label = r'$u$')  
    plt.plot(ANG,V,label = r'$v$')
    plt.plot(ANG,Z,label = r'$h$')

    plt.ylim([-1.5,1.5])
    plt.xlim([0,90])
    x_ticks = np.linspace(0,90,7) 
    plt.xticks(x_ticks)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend()
    plt.title(l)
    plt.xlabel(r'Latitude ($\phi$) - deg')
    plt.show()

    plt.plot(ANG, DU, label = 'DU')
    plt.plot(ANG, DV, label = 'DV')
    plt.plot(ANG, DZ, label = 'DZ')

    plt.ylim([-1.5,1.5])
    plt.xlim([0,90])
    x_ticks = np.linspace(0,90,7) 
    plt.xticks(x_ticks)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend()
    plt.title('Derivative - ' + l)
    plt.xlabel(r'Latitude ($\phi$) - deg')
    plt.show()

if __name__ == "__main__":
    hough_and_derivatives(1,2,3,10000)

    


    
        
        
    
    