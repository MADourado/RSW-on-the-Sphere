import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from .Matrix_m0 import matriz_C, matriz_D

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
