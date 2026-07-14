import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from rsw_sphere.hough_harmonics.normalization import norm_Hough, norm_component
from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.eigenvectors import Hough_harmonic

def alpha_label(alpha):

    if alpha == 1:
        return 'EIG'
    elif alpha == 2:
        return 'WIG'
    else:
        return 'RH'

def label(m,n,alpha,height):

    return f'{alpha_label(alpha)}({m},{n}) at {height}m'

def mode_tag(m,n,alpha):

    return f'{alpha_label(alpha)}-{m}-{n}'

def hough_and_derivatives(m,n,alpha, h_e:int = 10000, path:str = None):

    l = label(m,n,alpha, h_e)
    tag = mode_tag(m,n,alpha)
    g = 9.8
    a = 6.38e+06
    omega = 2*np.pi/24/60/60
    eps = (4*a*a*omega*omega)/(g*h_e)
    gamma = 1/np.sqrt(eps)

    U,V,Z,DU, DV, DZ, ANG,norm, eigen = norm_Hough(m, n, alpha, gamma, 10, 60)

    # U,V,Z,DU,DV,DZ are physically real; norm_Hough returns them with a
    # complex dtype (imaginary part is numerical noise), which trips up
    # matplotlib's isfinite() check and prints a ComplexWarning otherwise.
    U, V, Z = np.real(U), np.real(V), np.real(Z)
    DU, DV, DZ = np.real(DU), np.real(DV), np.real(DZ)

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
    plt.savefig(f'{path}/Hough_harmonic_{tag}.png')
    plt.close()

    plt.plot(ANG, DU, label = 'DU')
    plt.plot(ANG, DV, label = 'DV')
    plt.plot(ANG, DZ, label = 'DZ')

    #plt.ylim([-1.5,1.5])
    plt.xlim([0,90])
    x_ticks = np.linspace(0,90,7) 
    plt.xticks(x_ticks)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.legend()
    plt.title('Derivative - ' + l)
    plt.xlabel(r'Latitude ($\phi$) - deg')
    plt.savefig(f'{path}/derivatives_{tag}.png')
    plt.close()

if __name__ == "__main__":
    hough_and_derivatives(1,2,3,10000)