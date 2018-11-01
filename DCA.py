# import libs
import numpy as np
import cvxpy as cvx

def genComplexMatrix(num_row, num_col):
    # calculate normal distribution
    np.random.seed(100)
    mu, sigma = 0, 1.0
    dist_real = np.random(mu, sigma, num_row * num_col)
    dist_imag = np.random(mu, sigma, num_row * num_col)
    return np.matrix(dist_real+dist_imag).shape(num_row,num_col)

# ============================= Start reading data from files ===================================
data = np.load('data/Testdata_Exp1_4_34.npz')
f1 = np.matrix(data['f1'])
f2 = np.matrix(data['f2'])
sigmaR = np.int(data['sigmaR'])
sigma1 = np.int(data['sigma1'])
sigma2 = np.int(data['sigma2'])
L = np.matrix(data['L'])
sigma1E = np.int(data['sigma1E'])
sigma2E = np.int(data['sigma2E'])
F1 = np.matrix(data['F1'])
F2 = np.matrix(data['F2'])
g1 = np.complex128(data['g1'])
g2 = np.complex128(data['g2'])
D1 = np.matrix(data['D1'])
D2 = np.matrix(data['D2'])
Z = np.matrix(data['Z'])

e = 0.00001



# G(w) = (w.H)Dw +ln(C)

R1 = F1 * f2 * f2.H * F1.H
R2 = F2 * f1 * f1.H * F2.H

def G(w,D,C):
    return cvx.quad_form(w, D) + np.log(C)
    # return w.H * D * w + np.log10(C)


def G_result(w_k,D,C):
    w_k = np.matrix(w_k)
    return w_k.H * D * w_k + np.log(C)


def H(w_k,D,A1,B1,A2,B2):
    w_k = np.matrix(w_k)
    return w_k.H * D * w_k \
           + np.log(np.real(pow(sigma1, 2) + w_k.H * A1 * w_k)) + np.log(np.real(pow(sigma2, 2) + w_k.H * A2 * w_k)) \
           - np.log(np.real(pow(sigma1, 2) + w_k.H * B1 * w_k)) - np.log(np.real(pow(sigma2, 2) + w_k.H * B2 * w_k))


#
def Hbar(w_k,D,A1,B1,A2,B2):
    w_k = np.matrix(w_k)
    return cvx.real(H(w_k)) + 2 * cvx.real((((w - w_k).H) * D * w_k)) \
           + 2 * cvx.real((((w - w_k).H) * A1 * w_k)) / (pow(sigma1, 2) + (w_k.H) * A1 * w_k) \
           + 2 * cvx.real((((w - w_k).H) * A2 * w_k)) / (pow(sigma2, 2) + (w_k.H) * A2 * w_k) \
           - 2 * cvx.real((((w - w_k).H) * B1 * w_k)) / (pow(sigma1, 2) + (w_k.H) * B1 * w_k) \
           - 2 * cvx.real((((w - w_k).H) * B2 * w_k)) / (pow(sigma2, 2) + (w_k.H) * B2 * w_k)


# F(w)
def F(w_k,D,C):
    return 1 / (2 * np.log(2)) * (G_result(w_k,D,C) - H(w_k,D))


def solvePro(N,Pt):
    N = N
    Pt = Pt * 10
    P0 = P1 =P2 = Pt/4
    # ============================= Finish reading data from files ===================================
    I = np.identity(N)  # I is unit matrix
    A1 = P2 * R1 + sigmaR * sigmaR * D1
    A2 = P1 * R2 + sigmaR * sigmaR * D2

    B1 = sigmaR * sigmaR * D1
    B2 = sigmaR * sigmaR * D2

    D = A1 / (4 * sigma1 * sigma1) + A2 / (4 * sigma2 * sigma2) + B1 / (sigma1 * sigma1) + B2 / (sigma2 * sigma2)
    C = 1 + (P1 * pow(np.abs(g1), 2) + P2 * pow(np.abs(g2), 2)) / pow(sigma1E, 2)

    # variables - vector w
    w = cvx.Variable(shape=(N, 1), complex=True)

    w0 = genComplexMatrix(N, 1)
    constraints = [cvx.real(
        P1 * cvx.quad_form(w, D1) + P2 * cvx.quad_form(w, D2) + pow(sigmaR, 2) * cvx.quad_form(w, I)) <= Pt - P1 - P2,
                   cvx.real(w.H * Z) == 0,
                   cvx.imag(w.H * Z) == 0]
    previous_w = w0
    i = 0
    while (1):
        obj = cvx.Maximize(-cvx.real(G() - Hbar(previous_w,D,A1,B1,A2,B2)))
        prob = cvx.Problem(obj, constraints)
        prob.solve()

        if (np.linalg.norm(w.value - previous_w) / (1 + pow(np.linalg.norm(previous_w), 2)) < e):
            print("smaller1")
        if (abs((np.real(F(w.value,D,C)) - np.real(F(previous_w,D,C)))) / (1 + abs(np.real(F(previous_w,D,C)))) < e):
            print("smaller2")

        if (np.linalg.norm(w.value - previous_w) / (1 + pow(np.linalg.norm(previous_w), 2)) < e or abs(
                (np.real(F(w.value,D,C)) - np.real(F(previous_w,D,C)))) / (1 + abs(np.real(F(previous_w,D,C)))) < e):
            print("result = ", obj.value / (2 * np.log(2)))
            break

        previous_w = w.value
        i = i + 1

for N in range(4, 9, 2):
    for Pt in range(30, 41, 1):
        solvePro(N, Pt)



