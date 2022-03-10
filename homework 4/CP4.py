import numpy as np
import scipy.linalg as la
np.set_printoptions(precision = 9, linewidth = 800)
import math


### Problem 1

### Jacobi and Gauss Seidel Iteration functions
### Create your functions here
### Both functions will need two outputs and three inputs
### The code within the function will be very similar to
### Week 4 coding lecture 2

def jacobi(A: np.ndarray, b: np.ndarray, tolerance: float):
    n = A.shape[0]

    # counter
    t = 0

    # diagonal matrix
    P = np.diag(np.diag(A))

    # splitting original matrix
    T = A - P

    # initialize vectors to store and compare
    X = np.zeros([n, 1])

    # set up initial error value
    e = tolerance + 1

    # check if method is able to converge and perform method
    M = -la.solve(P, T)
    w, V = np.linalg.eig(M)
    if np.max(np.abs(w)) < 1:
        while e >= tolerance:
            # X[:, t:(t+1)] returns a column vector
            # apply Jacobi method, but multiply two sides with the diagonal matrix
            X = np.hstack((X, la.solve_triangular(P,-T @ X[:, t:(t+1)] + b)))

            # X[:, t+1] returns a row vector
            e = np.max(np.abs(A @ X[:,t+1:]-b))
            t += 1

    return e, t


def gauss_seidel(A: np.ndarray,b: np.ndarray, tolerance: float):
    n = A.shape[0]

    # P is the lower triangle matrix of A
    P = np.tril(A)

    T = A - P

    # counter
    t = 0

    # initialize vectors to store and compare
    X = np.zeros([n, 1])

    # set up initial error value
    e = tolerance + 1

    # check convergence
    M = -la.solve(P, T)
    w, V = np.linalg.eig(M)
    if np.max(np.abs(w)) < 1:
        while e >= tolerance:
            X = np.hstack((X, la.solve_triangular(P, -T @ X[:, t:(t + 1)] + b, lower=True)))
            e = np.max(np.abs(A@X[:,t+1:]-b))
            t += 1

    return e, t


### Once you have created your functions initialize your matrix A and RHS b

A = np.array([[1.1, 0.2, -0.2, 0.5],
              [0.2, 0.9, 0.5, 0.3],
              [0.1, 0, 1, 0.4],
              [0.1, 0.1, 0.1, 1.2]])
b = np.array([[1],[0],[1],[0]])

### Use your Jacobi and Gauss-Seidel functions to find A1 through A4.

Ej_2, Tj_2 = jacobi(A, b, 1e-2)
Egs_2, Tgs_2 = gauss_seidel(A, b, 1e-2)

Ej_4, Tj_4 = jacobi(A, b, 1e-4)
Egs_4, Tgs_4 = gauss_seidel(A, b, 1e-4)

Ej_6, Tj_6 = jacobi(A, b, 1e-6)
Egs_6, Tgs_6 = gauss_seidel(A, b, 1e-6)

Ej_8, Tj_8 = jacobi(A, b, 1e-8)
Egs_8, Tgs_8 = gauss_seidel(A, b, 1e-8)

A1 = np.array([Tj_2, Tj_4, Tj_6, Tj_8]).reshape(1,4)
A2 = np.array([Ej_2, Ej_4, Ej_6, Ej_8]).reshape(1,4)
A3 = np.array([Tgs_2, Tgs_4, Tgs_6, Tgs_8]).reshape(1,4)
A4 = np.array([Egs_2, Egs_4, Egs_6, Egs_8]).reshape(1,4)

### Problem 2
###  Initialize your Day 0 vector x

x0 = np.array([[0.9],[0.09],[0.01]])

###  Part 1: without a vaccine
###  Make sure to have p = 0
###  Initialize the SIR matrix M, and save it as A5

# Call k(n) and k(n+1) the two consecutive days, we have:
#
# S_k(n+1)  = already s +   r->s    -    s->r (vaxx)   -    s->i
#           = S_k(n)    +   1/10000*R_k(n) -  p*S_k(n) -    1/200 S_k(n)
#           = (1-p-1/200)*S_k(n) + 0*I_k(n) + 1/10000 * R_k(n)
#
# I_k(n+1)  = already i + s->i          - i->r
#           = I_k(n)    + 1/200*S_k(n)  - 1/1000* I_k(n)
#           = 1/200S_k(n) + (1-1/1000)*I_k(n) + 0*R_k(n)
#
# R_k(n+1)  = already_r     + i->r          + s->r(vaxx)    - r->s (variant)
#           = R_k(n)        + 1/1000*I_k(n) + p*S_k(n)      - 1/10000*R_k(n)
#           = p*S_k(n)      + 1/1000*I_k(n) + (1-1/10000)*R_k(n)
p1 = 0
M1 = np.array([[1 - p1 - 1 / 200, 0, 1 / 10000],
                   [1 / 200, 1 - 1 / 1000, 0],
                   [p1, 1 / 1000, 1 - 1 / 10000]])
A5 = M1

###  Create a loop to find the day that the number of infected
###  individuals hits 50% and another loop for the steady state of the
###  infected population
###  There is away to put everything under one loop if you make clever use
###  of conditionals
def problem2_solver(M, x0):
    D = 0
    F = 0

    X = x0
    t = 1e-8
    err = t+ 1
    # initialize vectors to store and compare
    while err > t:
        X = np.hstack((X, M @ X[:, F:]))
        if X[1, F+1] < 0.5 and D-F == 0:
            D = D + 1

        err = abs(X[1, F] - X[1, F+1])
        F += 1
    return D, X[1,F]


### Save the days and steady state in a row vector A6
D0, F0 = problem2_solver(M1, x0)
A6 = np.array([D0+1, F0]).reshape(1,2)

###  Reinitialize your Day 0 vector x


###  Part 2: with a vaccine
###  Make sure to have p = 2/1000
###  Initialize the SIR matrix M, and save it as A7

p2 = 2/1000
M2= np.array([[1 - p2 - 1 / 200, 0, 1 / 10000],
                   [1 / 200, 1 - 1 / 1000, 0],
                   [p2, 1 / 1000, 1 - 1 / 10000]])

A7 = M2
### Save the days and steady state in a row vector A8
D0, F0 = problem2_solver(M2, x0)
A8 = np.array([D0+1, F0]).reshape(1,2)



### Problem 3

###  Initialize your 114x114 tridiagonal matrix A

A = np.diag(np.full(114, 2))+np.diag(np.full(113,-1),1)+np.diag(np.full(113,-1),-1)

A9 = A.copy()
###  Initialize your 114x1 RHS column vector rho

rho = np.zeros([114,1])

for i, j in enumerate(rho):
    rho[i] = 2*(1-math.cos(53*math.pi/115))*math.sin(53*math.pi*(i+1)/115)

A10 = rho.copy()
###  Create a column vector phi that contains the exact solution given in
###  the assignment file

phi = np.ones([114,1])
for i, j in enumerate(phi):
    phi[i] = math.sin(53*math.pi*(i+1)/115)


###  Implement Jacobi's method for this system.
###  Don't use the function you created before because that was designed for
###  a specific task, and will not work here.

def jacobi_problem3(A: np.ndarray, b: np.ndarray, tolerance: float):
    n = A.shape[0]

    # counter
    t = 0

    # diagonal matrix
    D = np.diag(np.diag(A))
    d = np.diag(A).reshape(n,1)
    # splitting original matrix
    T = A - D


    # initialize vectors to store and compare
    X = np.ones([n, 1])
    c = b/d
    M = -T/d

    # set up initial error value
    e = tolerance + 1

    # check if method is able to converge and perform method
    while e >= tolerance:
        X = np.hstack((X, M @ X[:, t:(t+1)]+c))
        e = np.max(np.abs(X[:,t+1] - X[:,t]))
        t +=1

    return X[:,t:(t+1)], t+1

A11, A12 = jacobi_problem3(A, rho, 1.0e-5)


### Matrix Iteration Function -- students don't have to use a function for this

###  Save the difference of the Jacobi solution and the exact solution as
###  A13.  Use the maximal entry in absolute value to calculate this error.

A13 = np.max(np.abs(A11-phi))


###  Implement Gauss-Seidel for this system.
###  Don't use the function you created before because that was designed for
###  a specific task, and will not work here.

def gauss_seidel_problem3(A: np.ndarray,b: np.ndarray, tolerance: float):
    n = A.shape[0]

    # P = A + D
    LpD = np.tril(A)

    U = A - LpD

    # counter
    t = 0

    # initialize vectors to store and compare
    X = np.ones([n, 1])

    # set up initial error value
    e = tolerance + 1

    M = -la.solve(LpD, U)
    c = la.solve(LpD, b)

    while e >= tolerance:
        X = np.hstack((X, M @ X[:, t:(t+1)]+c))
        e = np.max(np.abs(X[:,t+1] - X[:,t]))
        t +=1

    return X[:,t:(t+1)], t+1


###  Save the difference of the Gauss-Seidel solution and the exact solution as
###  A16.  Use the maximal entry in absolute value to calculate this error.
A14, A15 = gauss_seidel_problem3(A, rho, 1.0e-5)
A16 = np.max(np.abs(A14-phi))
