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
    num = len(m)

    w = []
    wd = []
    vd = []
    vcd = []
    F = []
    N = []

    # Pad vectors for notation consistency
    m = list(m)
    m.insert(0,0)
    Pc = list(Pc)
    Pc.insert(0,0)
    Ic = list(Ic)
    Ic.insert(0,0)
    Qd = list(Qd)
    Qd.insert(0,0) # COLUMN
    Qdd = list(Qdd)
    Qdd.insert(0,0) # COLUMN

    Z = np.array([[0, 0, 1]]).T # COLUMN

    w.append(np.array([[0, 0, 0]]).T)
    wd.append(np.array([[0, 0, 0]]).T)
    vd.append(np.array([[0, 0, 0]]).T)
    vcd.append(np.array([[0, 0, 0]]).T)
    F.append(np.array([[0, 0, 0]]).T)
    N.append(np.array([[0, 0, 0]]).T)

    G = -g0
    vd.append(G)

    for i in range(0,num):
        R        = Ti[i,0:3,0:3].T                                                                         # ^i+1_i R
        P        = Ti[i,0:3,3]                                                                             # ^i P_i+1
        w.append(R.dot(w[i]) + Z * Qd[i+1])                                                                # 6.45
        wd.append(R.dot(wd[i]) + np.cross(R.dot(w[i]),Z * Qd[i+1],axis=0) + Z * Qdd[i+1])                  # 6.46
        vd.append(R.dot(np.cross(wd[i],P,axis=0) + np.cross(w[i],np.cross(w[i],P,axis=0),axis=0) + vd[i])) # 6.47
        vcd.append(np.cross(wd[i+1],np.array(Pc[i+1]).T,axis=0) + np.cross(w[i+1],np.cross(w[i+1],np.array(Pc[i+1]).T,axis=0),axis=0) + vd[i+1])    # 6.48
        F.append(m[i+1]*vcd[i+1])                                                                          # 6.49
        N.append(Ic[i+1]*wd[i+1] + np.cross(w[i+1],Ic[i+1]*w[i+1],axis=0))                                 # 6.50

    # for i in range(num,1,-1):
    #     if i == num
    #         f[i] = F[i]                                           # 6.51
    #         n[i] = N[i] + np.cross(np.array(Pc[i]).T,F[i],axis=0) # 6.52
    #     else
    #         R = Ti{i}(1:3,1:3);
    #         P = Ti{i}(1:3,4);
    #         f{i} = R*f{i+1} + F{i}; % 6.51
    #         n{i} = N{i} + R*n{i+1} + cross(Pc{i},F{i}) + cross(P,R*f{i+1}); % 6.52
    #     Tau(i,1) = simplify( n{i}.'*Z ); % 6.53

def dynamics(a, alpha, d, theta):
    _ , _, T_array = dh2tf(a,alpha,d,theta)
    # T_array = np.array([[simplify(T_array[i][j]) for j in range(0,len(T_array[i]))] for i in range(0,len(T_array))])
    # print(T_array[0][0][0])
    for i in range(0,len(T_array)):
        for j in range(0,len(T_array[i])):
            for k in range(0,len(T_array[i][j])):
                T_array[i][j][k]= simplify(T_array[i][j][k])
    T_array = np.array(T_array)

    Q = symbols('q1 q2 q3')
    
    Qd = symbols('q1d q2d q3d')
    
    Qdd = symbols('q1dd q2dd q3dd')
    
    m = symbols('m1 m2 m3')
    
    Pc1 = symbols('Pc1x Pc1y Pc1z')
    Pc2 = symbols('Pc2x Pc2y Pc2z')
    Pc3 = symbols('Pc3x Pc3y Pc3z')
    Pc = ([list(Pc1)], [list(Pc2)], [list(Pc3)])

    Ic1vars = symbols('Ic1xx Ic1xy Ic1xz Ic1yy Ic1yz Ic1zz')
    Ic2vars = symbols('Ic2xx Ic2xy Ic2xz Ic2yy Ic2yz Ic2zz')
    Ic3vars = symbols('Ic3xx Ic3xy Ic3xz Ic3yy Ic3yz Ic3zz')
    Ic1 = np.array([[ Ic1vars[0], -Ic1vars[1], -Ic1vars[2]],
                    [-Ic1vars[1],  Ic1vars[3], -Ic1vars[4]],
                    [-Ic1vars[2], -Ic1vars[4],  Ic1vars[5]]])
    Ic2 = np.array([[ Ic2vars[0], -Ic2vars[1], -Ic2vars[2]],
                    [-Ic2vars[1],  Ic2vars[3], -Ic2vars[4]],
                    [-Ic2vars[2], -Ic2vars[4],  Ic2vars[5]]])
    Ic3 = np.array([[ Ic3vars[0], -Ic3vars[1], -Ic3vars[2]],
                    [-Ic3vars[1],  Ic3vars[3], -Ic3vars[4]],
                    [-Ic3vars[2], -Ic3vars[4],  Ic3vars[5]]])                    
    
    Ic = (list(Ic1), list(Ic2), list(Ic3))

    g = symbols('g')
    g0 = np.array([[0,0,-g]]).T

    dynamics_newtonian(m, Pc, Ic, T_array, Qd, Qdd, g0)


if __name__ == "__main__":
    l1, l2, q1, q2, q3 = symbols('l1 l2 q1 q2 q3');
    a = (0, 0, l1, l2)
    alpha = (0, pi/2, 0, 0)
    d = (0, 0, 0, 0)
    theta = (q1, q2+pi/2, q3, 0)
    dynamics(a, alpha, d, theta)