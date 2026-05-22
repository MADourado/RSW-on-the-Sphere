import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from Hough_Harmonics.Normalization import norm_Hough, norm_component
from Hough_Harmonics.Eigenvalues_and_eigenvectors.Eigenvectors import Hough_harmonic



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