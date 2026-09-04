import numpy as np

def p(n):
    
    num = (n-1)*(n+1)
    den = (2*n-1)*(2*n+1)
    
    return np.sqrt(num/den)

def r(n,gamma):
    
    return gamma*np.sqrt(n*(n+1))

def matriz_C(gamma, N):
    
    C = np.zeros((3*N,3*N))
    
    C[0][0] = r(0,gamma)**2 + p(1)**2
    
    for i in range(1,3*N):
        
        # DIAGONAL
        C[i][i] = r(2*i,gamma)**2 + p(2*i)**2 + p(2*i+1)**2
        
        # BANDA 1
        C[i][i-1] = p(2*i-1)*p(2*i)
        C[i-1][i] = p(2*i-1)*p(2*i)

    return C

def matriz_D(gamma, N):
    
    D = np.zeros((3*N,3*N))
    
    D[0][0] = r(1,gamma)**2 + p(1)**2 + p(2)**2
    for i in range(1,3*N-1):
        
        # DIAGONAL
        D[i][i] = r(2*i+1,gamma)**2 + p(2*i+1)**2 + p(2*i+2)**2
        
        # BANDA 1
        D[i][i-1] = p(2*i)*p(2*i+1)
        D[i-1][i] = p(2*i)*p(2*i+1)
    
    return D
    