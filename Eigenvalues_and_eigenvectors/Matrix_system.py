import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from Eigenvalues_and_eigenvectors.Matrix_m0 import matriz_C, matriz_D

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

EPS = 1
gamma = 1/np.sqrt(EPS)


g = 9.8
h_e = 10000

a = 6.38e+06
omega = 2*np.pi/24/60/60
eps = (4*a*a*omega*omega)/(g*h_e)
#eps = 8.810572669756384
gamma = 1/np.sqrt(eps)
print(eps)
N = 3

M = np.arange(0,11)+0.00001
s = 1 #7.272205*2/10
eigen_M = []

for m in M:
    
    AA = matriz_A(m, gamma, N)
    BB = matriz_B(m, gamma, N)
    
    eigen_a, value_a = np.linalg.eig(AA)
    eigen_b, value_b = np.linalg.eig(BB)
    
    eigen = np.concatenate((eigen_a, eigen_b))
    eigen = np.sort(eigen)
    
    eigen_M += [ eigen*s]
    



CC = matriz_C(gamma, N)
DD = matriz_D(gamma, N)
eigen_c, value_c = np.linalg.eig(CC)
eigen_d, value_d = np.linalg.eig(DD)

print()
eigen0 = np.concatenate((np.sqrt(eigen_c), np.sqrt(eigen_d)))
eigen0 = np.sort(eigen0)[1:7]*s

eigen_M = np.array(eigen_M)



C = ['#010ae7', '#01d9e7', '#01e741', '#f77001', '#ff0505', '#000000']
#C.reverse()
l = ['n-m = 0', 'n-m = 1', 'n-m = 2', 'n-m = 3', 'n-m = 4', 'n-m = 5']
#l.reverse()
print()
print('M = ', M)
fig, ax = plt.subplots()

n=12
K = np.concatenate((np.array([eigen0[n-12]]), eigen_M[1:,n]))
ax.plot(M[:],K,'-', linewidth=2, color = C[1])#, label = l[n%6])
n=13
K = np.concatenate((np.array([eigen0[n-12]]), eigen_M[1:,n]))
ax.plot(M[:],K,'-', linewidth=2, color = C[2])#, label = l[n%6])
for n in range(14,18):
    #print(f'n={n}, eigen = {eigen_M[:,n]}')
    K = np.concatenate((np.array([eigen0[n-12]]), eigen_M[1:,n]))
    ax.plot(M[:],K,'-', linewidth=2, color = C[0])
    
for n in range(2,6):
    #print(f'n={n}, eigen = {-eigen_M[:,n]}')
    if n >= 2 :
        K = np.concatenate((np.array([eigen0[7-n]]), -eigen_M[1:,n]))
        ax.plot(0, eigen0[n],'-', linewidth=2, color = C[n%6])#, label = l[n%6])
        ax.plot(-M[:], K,'-', linewidth=2, color = C[0]) #, label = l[n%6])
    else:
        K = -eigen_M[1:,n]
        ax.plot(0, eigen0[n],'-', linewidth=2, color = C[n%6], label = l[n%6])
        ax.plot(-M[1:], K,'-', linewidth=2, color = C[(5-n)%6]) #, label = l[n%6])
    
    
for n in range(6,12):
    print(f'n={n}, eigen = {-2*np.pi/(eigen_M[:,n]*1e-4)/86400}')

    K = np.concatenate((np.array([0]), -eigen_M[1:,n]))
    if n == 6:
        K = np.concatenate((np.array([eigen0[1]]), -eigen_M[1:,n]))
        ax.plot(-M[:], K,'-', linewidth=2, color = C[2]) #, label = l[n%6])
    else:
        ax.plot(-M[:], K,'-', linewidth=2, color = C[3]) #, label = l[n%6])
    

# plt.xticks(np.linspace(-10,10,5))
# plt.grid()
# #plt.xlim([0,20])
# plt.ylim([0,5])
# #plt.title(r'Dispersion Relation - Eastward Inertia-Gravity Waves ')
# plt.xlabel('Zonal wavenumber (m)')
# plt.ylabel(r'Frequency $(10^{-4} \text{ rad } s^{-1})$')
# plt.legend(loc='upper left')
# plt.show()

ax.set_xticks(np.linspace(-10,10,5))

ax.grid()

# IMPORTANTE:
# não usar zero para evitar infinito no período
ax.set_ylim([0.01,2.5])

ax.set_xlabel('Zonal wavenumber')

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

# mesmos limites visuais do eixo esquerdo
secax.set_ylim(ax.get_ylim())

# ticks escolhidos manualmente
freq_ticks = np.array([0.5, 1,1.5, 2,2.5, 3,3.5, 4])

# posiciona ticks exatamente nas mesmas alturas
secax.set_yticks(freq_ticks)

# converte para período
period_labels = [
    f"{freq_to_period(f):.2f}"
    for f in freq_ticks
]

secax.set_yticklabels(period_labels)

secax.set_ylabel('Period (days)')

plt.show()

    


















