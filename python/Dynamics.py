import numpy as np
from sympy import *
# from sympy.codegen.ast import Assignment
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
    Qd.insert(0,0)
    Qdd = list(Qdd)
    Qdd.insert(0,0)

    Z = np.array([[0, 0, 1.0]]).T

    w = [np.array([[0, 0, 0]]).T] * (num+1)
    wd = [np.array([[0, 0, 0]]).T] * (num+1)
    vd = [np.array([[0, 0, 0]]).T] * (num+1)
    vcd = [np.array([[0, 0, 0]]).T] * (num+1)
    F = [np.array([[0, 0, 0]]).T] * (num+1)
    N = [np.array([[0, 0, 0]]).T] * (num+1)

    f = [np.array([[0, 0, 0]]).T] * (num+1)
    n = [np.array([[0, 0, 0]]).T] * (num+1)
    
    Tau = [0] * (num+1)

    G = -g0
    vd[0] = G

    for i in range(0,num):
        R        = Ti[i,0:3,0:3].T                                                                        # ^i+1_i R
        P        = Ti[i,0:3,3]                                                                            # ^i P_i+1
        w[i+1]   = R.dot(w[i]) + Z * Qd[i+1]                                                                # 6.45
        wd[i+1]  = R.dot(wd[i]) + np.cross(R.dot(w[i]),Z * Qd[i+1],axis=0) + Z * Qdd[i+1]                  # 6.46
        vd[i+1]  = R.dot(np.cross(wd[i],P,axis=0) + np.cross(w[i],np.cross(w[i],P,axis=0),axis=0) + vd[i]) # 6.47
        vcd[i+1] = np.cross(wd[i+1],np.array(Pc[i+1]).T,axis=0) + np.cross(w[i+1],np.cross(w[i+1],np.array(Pc[i+1]).T,axis=0),axis=0) + vd[i+1]          # 6.48
        F[i+1]   = m[i+1]*vcd[i+1]                                                                          # 6.49
        N[i+1]   = Ic[i+1].dot(wd[i+1]) + np.cross(w[i+1],Ic[i+1].dot(w[i+1]),axis=0)                       # 6.50

    for i in range(num,0,-1):
        if i == num:
            f[i] = F[i]                                           # 6.51
            n[i] = N[i] + np.cross(np.array(Pc[i]).T,F[i],axis=0) # 6.52
        else:
            R = Ti[i,0:3,0:3]
            P = Ti[i,0:3,3]    
            f[i] = R.dot(f[i+1]) + F[i]  # 6.51
            n[i] = N[i] + R.dot(n[i+1]) + np.cross(np.array(Pc[i]).T,F[i],axis=0) + np.cross(P,R.dot(f[i+1]),axis=0) # 6.52
        Tau[i] = (n[i].T.dot(Z))[0,0] # 6.53
    
    return Tau[1:]

def dynamics(a, alpha, d, theta):
    _ , T_array, _ = dh2tf(a,alpha,d,theta)

    # for i in range(0,len(T_array)):
        # for j in range(0,len(T_array[i])):
            # for k in range(0,len(T_array[i][j])):
                # T_array[i][j][k]= simplify(T_array[i][j][k])

    T_array = np.array(T_array)

    ints = range(1,len(a)+1)

    Q = symbols([('q' + str(i) + ' ') for i in ints][:-1])

    
    Qd = symbols([('q' + str(i) + 'd ') for i in ints][:-1])
    
    Qdd = symbols([('q' + str(i) + 'dd ') for i in ints][:-1])
    
    m = symbols([('m' + str(i) + ' ') for i in ints][:-1])
    
    Pc = []
    Ic = []
    for i in range(1,len(a)+1):
        Pci = symbols('Pc' + str(i) + 'x Pc' + str(i) + 'y Pc' + str(i) + 'z')
        Pc.append([list(Pci)])

        Icivars = symbols('Ic' + str(i) + 'xx Ic' + str(i) + 'xy Ic' + str(i) + 'xz Ic' + str(i) + 'yy Ic' + str(i) + 'yz Ic' + str(i) + 'zz')
        Ic.append(np.array([[ Icivars[0], -Icivars[1], -Icivars[2]],
                            [-Icivars[1],  Icivars[3], -Icivars[4]],
                            [-Icivars[2], -Icivars[4],  Icivars[5]]]))
    # Ic2vars = symbols('Ic2xx Ic2xy Ic2xz Ic2yy Ic2yz Ic2zz')
    # Ic3vars = symbols('Ic3xx Ic3xy Ic3xz Ic3yy Ic3yz Ic3zz')
        # Ic1 = np.array([[ Ic1vars[0], -Ic1vars[1], -Ic1vars[2]],
    #                 [-Ic1vars[1],  Ic1vars[3], -Ic1vars[4]],
    #                 [-Ic1vars[2], -Ic1vars[4],  Ic1vars[5]]])
    # Ic2 = np.array([[ Ic2vars[0], -Ic2vars[1], -Ic2vars[2]],
    #                 [-Ic2vars[1],  Ic2vars[3], -Ic2vars[4]],
    #                 [-Ic2vars[2], -Ic2vars[4],  Ic2vars[5]]])
    # Ic3 = np.array([[ Ic3vars[0], -Ic3vars[1], -Ic3vars[2]],
    #                 [-Ic3vars[1],  Ic3vars[3], -Ic3vars[4]],
    #                 [-Ic3vars[2], -Ic3vars[4],  Ic3vars[5]]])                    
    
    # Ic = (Ic1, Ic2, Ic3)

    g = symbols('g')
    g0 = np.array([[0,0,-g]]).T

    Tau = dynamics_newtonian(m, Pc, Ic, T_array, Qd, Qdd, g0)

    print(Tau[2])

    return separate_mvg(Tau, Qdd, g)

def separate_mvg(Tau, Qdd, g):
    n = len(Tau)
    M = [[0]*n]*n
    G = [0]*n
    V = [0]*n
    for i in range(0,n):
        for j in range(0,n):
            M[i][j] = diff(Tau[i],Qdd[j])
        G[i] = diff(Tau[i],g) * g
        V[i] = Tau[i].subs([(qdd, 0) for qdd in Qdd] + [(g,0)])
    return (M, V, G)

if __name__ == "__main__":
    l1, l2, q1, q2, q3 = symbols('l1 l2 q1 q2 q3');
    a = (0, 0, l1, l2)
    alpha = (0, pi/2, 0, 0)
    d = (0, 0, 0, 0)
    theta = (q1, q2+pi/2, q3, 0)
    M, V, G = dynamics(a, alpha, d, theta)