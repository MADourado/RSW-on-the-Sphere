import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from Matrix_m0 import matriz_C, matriz_D

def p(n,m):
    
    num = (n-1)*(n+1)*(n-m)*(n+m)
    den = n*n*(2*n-1)*(2*n+1)
    
    return np.sqrt(num/den)

def q(n,m):
    
    return m/(n*(n+1))

def r(n,gamma):
    
    return gamma*np.sqrt(n*(n+1))

def matriz_A(m,gamma, N):
    
    A = np.zeros((3*N,3*N))
    
    for i in range(N):
        
        # DIAGONAL
        A[3*i+1][3*i+1] = -q(m + 2*i,m)
        A[3*i+2][3*i+2] = -q(m + 2*i + 1,m)
        
        # BANDA 1
        A[3*i][3*i+1]   = r(m+2*i,gamma)
        A[3*i+1][3*i+2] = p(m+2*i+1,m)
        
        A[3*i+1][3*i]   = r(m+2*i,gamma)
        A[3*i+2][3*i+1] = p(m+2*i+1,m)
        
        if i < N-1:
            
            # BANDA 2
            A[3*i+2][3*i+4]  = p(m+2+2*i,m)
            A[3*i+4][3*i+2]  = p(m+2+2*i,m)
    
    return A

def matriz_B(m,gamma, N):
    
    B = np.zeros((3*N,3*N))
    
    for i in range(N):
        
        # DIAGONAL
        B[3*i][3*i]     = -q(m + 2*i,m)
        B[3*i+2][3*i+2] = -q(m + 2*i + 1,m)
        
        # BANDA 1
        B[3*i+1][3*i+2] = r(m+2*i+1,gamma)
        B[3*i+2][3*i+1] = r(m+2*i+1,gamma)
        
        if i < N-1:
            B[3*i+2][3*i+3] = p(m+2*i+2,m)
            B[3*i+3][3*i+2] = p(m+2*i+2,m)
            
        # BANDA 2
        B[3*i][3*i+2]  = p(m+1+2*i,m)
        B[3*i+2][3*i]  = p(m+1+2*i,m)
    
    return B

#---------------------------------
# DISPERSION RELATION
#---------------------------------

def dispersion_relation(h_e: int = 10000):
    g = 9.8                           # gravity
    a = 6.38e+06                      # Earth's radius
    omega = 2*np.pi/24/60/60          # Earth's rotation rate

    eps = (4*a*a*omega*omega)/(g*h_e) # Lamb's number
    gamma = 1/np.sqrt(eps)

    N = 3               # there will be 6N eigenvalues (2N for each wave type)
    M = np.arange(1,11) # zonal wavenumber (the m=0 case is treated alone)

    s = 2*omega*1e+4    # scale (the graph unit will be 10-4)

    eigen_M = []
    for m in M:
        
        AA = matriz_A(m, gamma, N) # symmetric modes
        BB = matriz_B(m, gamma, N) # anti-symmetric modes
        
        eigen_a, value_a = np.linalg.eig(AA)
        eigen_b, value_b = np.linalg.eig(BB)
        
        eigen = np.concatenate((eigen_a, eigen_b))
        eigen = np.sort(eigen)
        
        eigen_M += [ eigen*s]

    eigen_M = np.array(eigen_M)

    # Case m = 0
    CC = matriz_C(gamma, N)
    DD = matriz_D(gamma, N)
    eigen_c, value_c = np.linalg.eig(CC)
    eigen_d, value_d = np.linalg.eig(DD)

    eigen0 = np.concatenate((np.sqrt(eigen_c), np.sqrt(eigen_d)))
    eigen0 = np.sort(eigen0)[1:7]*s
    print(eigen0)
    M = np.concatenate((np.array([0]), M))

    #-----------------------------------------------------
    # PLOTS - RH and WIG modes have negative frequencies,
    #         therefore it is taken -w for both of them
    #-----------------------------------------------------

    C = ['#010ae7', '#01d9e7', '#01e741', '#f77001', '#ff0505', '#000000']
    l = ['n-m = 0', 'n-m = 1', 'n-m = 2', 'n-m = 3', 'n-m = 4', 'n-m = 5']

    fig, ax = plt.subplots()

    #-------------------------------------
    # EIG - Eastward Inertia-Gravity Waves
    #-------------------------------------

    # Kelvin Wave
    n=12
    K = np.concatenate((np.array([eigen0[n-12]]), eigen_M[:,n]))
    ax.plot(M[:],K,'-', linewidth=2, color = C[1])

    # Mixed Rossby-Gravity Wave
    n=13
    K = np.concatenate((np.array([eigen0[n-12]]), eigen_M[:,n]))
    ax.plot(M[:],K,'-', linewidth=2, color = C[2])

    # Pure EIG
    for n in range(14,18):
        K = np.concatenate((np.array([eigen0[n-12]]), eigen_M[:,n]))
        ax.plot(M[:],K,'-', linewidth=2, color = C[0])

    #-------------------------------------
    # WIG - Westward Inertia-Gravity Waves
    #-------------------------------------
        
    for n in range(2,6):
        K = np.concatenate((np.array([eigen0[7-n]]), -eigen_M[:,n]))
        ax.plot(-M[:], K,'-', linewidth=2, color = C[0]) 
    
    #-------------------------------------
    # RH - Rossby-Haurwitz Waves
    #-------------------------------------

    # Mixed Rossby-Gravity
    K = np.concatenate((np.array([eigen0[1]]), -eigen_M[:,6]))
    ax.plot(-M[:], K,'-', linewidth=2, color = C[2]) 

    for n in range(7,12):
        K = np.concatenate((np.array([0]), -eigen_M[:,n]))
        ax.plot(-M[:], K,'-', linewidth=2, color = C[3]) 
        

    ax.set_xticks(np.linspace(-10,10,5))

    ax.grid()

    ax.set_ylim([0,4])
    ax.set_xlabel('Zonal wavenumber (m)')
    ax.set_ylabel(
        r'Frequency $(10^{-4}\ \mathrm{rad}\ s^{-1})$'
    )

    # =========================================================
    # Conversão frequência -> período
    # =========================================================

    def freq_to_period(freq):
        """
        freq está em unidades de 10^-4 rad/s
        """
        
        omega = freq * 1e-4
        
        return 2*np.pi / (omega * 86400)

    # =========================================================
    # Eixo direito manual
    # =========================================================

    secax = ax.twinx()
    secax.set_ylim(ax.get_ylim())
    freq_ticks = np.array([0.5, 1,1.5, 2,2.5, 3,3.5, 4])
    secax.set_yticks(freq_ticks)
    period_labels = [
        f"{freq_to_period(f):.2f}"
        for f in freq_ticks
    ]
    secax.set_yticklabels(period_labels)
    secax.set_ylabel('Period (days)')
    plt.show()

if __name__ == "__main__":
    h_e = 10000 # equivalent height, by default = 10km
    dispersion_relation(h_e)
