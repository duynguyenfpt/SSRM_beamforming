import numpy as np
from numpy import linalg as LA

import matlab as ml

import scipy as sp
from scipy import linalg as sp_LA
from scipy.sparse.linalg import eigs

import math


def isHermitian(a, tol=1e-8):
    return np.allclose(a, a.getH(), atol=tol)


def genComplexMatrix(num_row, num_col):
    # calculate normal distribution
    mu, sigma = 0, 1.0
    norm_dist_real = np.random.normal(mu, sigma, num_row * num_col)
    norm_dist_imag = np.random.normal(mu, sigma, num_row * num_col)
    # generate complex random array based on normal distribution
    complex_norm = (1 / np.sqrt(2)) * (norm_dist_real + 1j * norm_dist_imag)

    # reshape the complex array to a num_row-row and num_col-col array
    complex_norm.shape = (num_row, num_col)

    # get the matrix form of complex_norm array
    return np.matrix(complex_norm)

def powerConsume(Xs):
    tmp = Xs.getH() * (np.kron(P1_hat.getT(),P2_hat)) * Xs
    tmp1 = Xs.getH() * (np.kron(Q1_hat.getT(), Q2_hat)) * Xs

    return (tmp/tmp1)

#test number 10
data = np.load('data/testdata10.npz')

N = np.int(data['N'])
f1 = np.matrix(data['f1'])
f2 = np.matrix(data['f2'])
nR = np.int(data['nR'])
n1 = np.int(data['n1'])
n2 = np.int(data['n2'])
L = np.matrix(data['L'])
n1E = np.int(data['n1E'])
n2E = np.int(data['n2E'])
F1 = np.matrix(data['F1'])
F2 = np.matrix(data['F2'])
g1 = np.complex128(data['g1'])
g2 = np.complex128(data['g2'])
D1 = np.matrix(data['D1'])
D2 = np.matrix(data['D2'])

#Z = L[f1,f2]
Z = L*np.column_stack((f1,f2))

U,S,V =LA.svd(Z.getH())

#G is column orthogonal matrix corresponding to zero sigular of Z^H
# => row[i] of G is column[i+2] of V

def createG():
    G = np.random.rand(N, 0)
    for i in range(2, N):
        tmp = V[i, :]
        G = np.c_[ G, tmp.getT()]
    return G

#create G
G = createG()

#print(G.getH().shape)

#Power consume at two Sources
PowerT = 4000
Power1 = 1000
Power2 = 1000

#variance
variance2_R = 1
variance2_k = 1

#(28a) , (28b)
R1 = F1*f2*f2.getH()*F2 #(13)
R2 = R1 #(13)

P1 = G.H * (variance2_R*D1 + Power2*R1)* G
P2 = G.H * (variance2_R*D2 + Power1*R2)* G

#A0

A0 = G.getH() * (Power1*D1 + Power2*D2 + variance2_R*np.identity(N)) * G # (12)

Q1 = variance2_R*G.getH()*D1*G
Q2 = variance2_R*G.getH()*D2*G

try:
    Ax = LA.inv(A0)
except np.linalg.LinAlgError:
    print("Input Matrix Is Not Invertible")
    pass

Ax = LA.inv(np.matrix(sp_LA.sqrtm(A0)))

#(29)
P1_nga = Ax.getH() * P1 * Ax
P2_nga = Ax.getH() * P2 * Ax
##
Q1_nga = Ax.getH() * Q1 * Ax
Q2_nga = Ax.getH() * Q2 * Ax

#(31) & (32)

tmp = variance2_k / (PowerT-Power1-Power2)
P1_hat = P1_nga + tmp * np.identity(N-2)
P2_hat = P2_nga + tmp * np.identity(N-2)
Q1_hat = Q1_nga + tmp * np.identity(N-2)
Q2_hat = Q2_nga + tmp * np.identity(N-2)


Aopt  = LA.inv(np.kron(Q1_hat.getT(), Q2_hat)) * (np.kron(P1_hat.getT(),P2_hat))

maxveigenvalues , xOpt = eigs(Aopt,1,which='LM')

xOpt =np.matrix(xOpt.reshape(18, 18))


if (isHermitian(xOpt)==True):
    eigenvalues, c_hat = eigs(xOpt,1,which='LM')
    w = math.sqrt(PowerT-Power1-Power2) * G * Ax * c_hat
    print(w)
else:
    result =-100
    w = np.random.rand(3).reshape(3,1)
    for i in range(61):
        phi_l = genComplexMatrix(N-2,N-2)
        X_l = xOpt * phi_l
        ##find the eigenvectors corresponding to largest eigenvalue
        ##subAlgorithm B
        values, xl_nga = eigs(X_l,1,which='LM')
        xl_nga= np.matrix(xl_nga)
        Xl_nga = xl_nga*xl_nga.getH()
        xl_hat = Xl_nga.reshape((N-2)* (N-2), 1)
        cur_value = powerConsume(xl_hat)
        if (cur_value > result):
            result = cur_value
            w = xl_nga
    print(w)



