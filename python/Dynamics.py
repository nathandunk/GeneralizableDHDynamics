import numpy as np
from sympy import *
# from math import pi

def dh2tf(a, alpha, d, theta):
    if not all(len(lst) == len(a) for lst in [alpha, d, theta]):
        print("Incorrect length of a, alpha, d, theta. Returning 0")
        return 0
    Ti  = []
    T0i = []
    T0N = np.eye(4)

    for i in range(0,len(a)):
        sinTheta = sin(theta[i]);
        cosTheta = cos(theta[i]);
        sinAlpha = sin(alpha[i]);
        cosAlpha = cos(alpha[i]);
        T = np.array([[         cosTheta,         -sinTheta,       0.0,           a[i]],
                      [sinTheta*cosAlpha, cosTheta*cosAlpha, -sinAlpha, -sinAlpha*d[i]],
                      [sinTheta*sinAlpha, cosTheta*sinAlpha,  cosAlpha,  cosAlpha*d[i]],
                      [              0.0,               0.0,       0.0,            1.0]])
        Ti.append(T)
        T0N = T0N.dot(T)
        T0i.append(T0N)

    return T0N,Ti,T0i

def dynamics_newtonian(m,Pc,Ic,Ti,Qd,Qdd,g0):
    print("here")

def dynamics(a, alpha, d, theta):
    T_array , _, _ = dh2tf(a,alpha,d,theta)
    T_array = np.array([[simplify(T_array[i][j]) for j in range(0,len(T_array[i]))] for i in range(0,len(T_array))])
    
    Q = symbols('q1 q2 q3')
    
    Qd = symbols('q1d q2d q3d')
    
    Qdd = symbols('q1dd q2dd q3dd')
    
    m = symbols('m1 m2 m3')
    
    Pc1 = symbols('Pc1x Pc1y Pc1z')
    Pc2 = symbols('Pc2x Pc2y Pc2z')
    Pc3 = symbols('Pc3x Pc3y Pc3z')
    Pc = (Pc1, Pc2, Pc3)

    Ic1 = symbols('Ic1xx Ic1xy Ic1xz Ic1yy Ic1yz Ic1zz')
    Ic2 = symbols('Ic2xx Ic2xy Ic2xz Ic2yy Ic2yz Ic2zz')
    Ic3 = symbols('Ic3xx Ic3xy Ic3xz Ic3yy Ic3yz Ic3zz')
    Ic = (Ic1, Ic2, Ic3)

    g = symbols('g')
    g0 = np.array([[0,0,-g]])

    dynamics_newtonian(m, Pc, Ic, T_array, Qd, Qdd, g0)


if __name__ == "__main__":
    l1, l2, q1, q2, q3 = symbols('l1 l2 q1 q2 q3');
    a = (0, 0, l1, l2)
    alpha = (0, pi/2, 0, 0)
    d = (0, 0, 0, 0)
    theta = (q1, q2+pi/2, q3, 0)
    dynamics(a, alpha, d, theta)