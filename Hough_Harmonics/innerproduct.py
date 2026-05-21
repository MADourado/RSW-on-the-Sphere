import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

from Hough_Harmonics.Normalization import norm_Hough
from Hough_Harmonics.Eigenvalues_and_eigenvectors.Eigenvectors import symetry


def inner_product(A,m_a,B,m_b,C,m_c,deg, conj = True): 
    
    # Inner product defining the coupling coefficient for mode a
    
    u_a,v_a,h_a,du_a,dv_a,dh_a = np.copy(A)
    u_b,v_b,h_b,du_b,dv_b,dh_b = np.copy(B)
    u_c,v_c,h_c,du_c,dv_c,dh_c = np.copy(C)
    
    # Observing the definition of the coupling coefficient, one notes
    # that if m_c = m_a + m_b, the projection on mode a depends on the 
    # conjugate of mode b, therefore conj = True for this mode. On the other
    # hand, the projection on mode c does not depend on conjugates, so conj = False
    # for this case
    
    if conj:
        
        m_b  *= -1
        v_b  *= -1
        dv_b *= -1        
  
    
    point, weight = np.polynomial.legendre.leggauss(deg)
    
    ang = np.pi/2 * point
    
    cos_phi = np.cos(ang)
    tan_phi = np.tan(ang)
    
    # Factor B for the inner product 
    B1  = m_c/cos_phi*u_b*u_c + v_b*du_c - u_b*v_c*tan_phi
    B1 += m_b/cos_phi*u_c*u_b + v_c*du_b - u_c*v_b*tan_phi
    
    B2  = 1j* (m_c/cos_phi*u_b*v_c + v_b*dv_c - u_b*u_c*tan_phi)
    B2 += 1j* (m_b/cos_phi*u_c*v_b + v_c*dv_b - u_c*u_b*tan_phi)
    
    B3  = m_c/cos_phi*u_b*h_c + v_b*dh_c 
    B3 += m_b/cos_phi*u_b*h_c  + h_c*dv_b - tan_phi*v_b*h_c
    
    B3 += m_b/cos_phi*u_c*h_b + v_c*dh_b 
    B3 += m_c/cos_phi*u_c*h_b  + h_b*dv_c - tan_phi*v_c*h_b
    
    inner = np.pi/2* sum(weight * (B1*u_a - 1j*B2*v_a + B3*h_a)*cos_phi)
    
    return inner

def S_abc(A,m_a,B,m_b,C,m_c,deg):
    
    # Integral for the cubic energy
    
    u_a,v_a,h_a,du_a,dv_a,dh_a = np.copy(A)
    u_b,v_b,h_b,du_b,dv_b,dh_b = np.copy(B)
    u_c,v_c,h_c,du_c,dv_c,dh_c = np.copy(C)
    
    point, weight = np.polynomial.legendre.leggauss(deg)
    
    ang = np.pi/2 * point
    
    cos_phi = np.cos(ang)
    
    B1 = h_a*(u_b*u_c + v_b*v_c)
    B2 = h_b*(u_a*u_c + v_a*v_c)
    B3 = h_c*(u_a*u_b - v_a*v_b)
    
    inner = np.pi/2* sum(weight * (B1+B2+B3)*cos_phi)
    
    return inner
    

