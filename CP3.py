import numpy as np
import math
import scipy.linalg as la

np.set_printoptions(precision = 9, linewidth = 500)
A = np.genfromtxt('bridge_matrix.csv', delimiter=',') ### Don't delete this line

################################################################################

### Problem 1
### A is already initialized above from a separate file (don't delete line 4).
### Initialize the data (right hand side) b.

b = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[3],[0],[math.exp(2)],[0],[np.pi]])
print(b)
### Solve for the force vector and save it as A1

A1 = la.solve(A,b)


### Compute the PA = LU factorization of A

P_, L, U = la.lu(A)
P = P_.T

### You may want to use some of these variables later on so don't forget to
### use .copy() wherever appropriate
### Save L AS A2, and c as A3.

A2 = L.copy()
c = la.solve_triangular(L, P@b, lower=True)
A3 = c.copy()

### Create a loop that breaks when one of the forces is greater than 20 tons
### Save A4 as the weight of the truck in position 8
### Save A5 as the entry of the force vector that exceeds 20 tons
b1 = b.copy()
x = np.zeros([13, 1])

while np.abs(x).max() < 20:
    b1[8] += 0.001
    y = la.solve_triangular(L, P @ b1, lower=True)
    x = la.solve_triangular(U, y)

A4 = b1[8]
A5 = np.abs(x).argmax()+1

### Problem 2
### Initialize, alpha, omega, and A, and compute the PA = LU factorization

alpha = -0.002
omega = 0.06
A = np.array([[1-alpha, -omega],[omega, 1-alpha]])

P2_, L2, U2 = la.lu(A)

P2 = P2_.T

### The initializations can get a little tricky so definitely ask for help
### if you're stuck.
### Initialize a matrix made up of the position vector at each time

B = np.zeros([1001, 2])


### Set the first x and y coordinates at time = 0 in your matrix
### to the values instructed in the assignment file.
B[0] = [1, -1]

### Create a loop that loops through each time given in the assignment file.
### Compute the new right hand side c using P, L, and/or U.
### You may need to recall that the inverse of P is P transpose
### Solve for the position by solving the Ux = c equation.

for i in range(0,1000,1):
    b = B[i].reshape(-1,1)
    c = la.solve_triangular(L2, P2@b, lower=True)
    B[i+1] = la.solve_triangular(U2, c).reshape(1, -1)


### Save all x coordinates as A6
### Save all y coordinates as A7
### Save the distance from the origin as A8

A6 = B[0:, 0:1].reshape(1, -1)
A7 = B[0:, 1:].reshape(1, -1)

A8 = np.zeros([1,1001])

for i in range(A8.size):
    A8[0, i] = math.sqrt(math.pow(A6[0, i], 2)+math.pow(A7[0, i], 2))


### Initialize a position vector
### Initialize a distance variable
### Initialize a time variable
### Create a loop that breaks when the distance from the origin is
### less than 0.06.
### In the loop compute the position using P, L, and/or U and
### compute the distance from the origin.
### Iterate time at each iteration of the loop.
### Save the time the loop breaks as A9.
### Save the distance from the origin as A10.

i = 0
d = A8[0, i]
while d >= 0.06:
    i += 1
    d = A8[0, i]

A9 = i
A10 = d

### Problem 3
### Create a function here for the rotation matrix that
### takes an input in radians and returns the matrix.

def rotate(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0,1,0],
                    [-np.sin(theta),0,np.cos(theta)]])


    
### Save A11 as R(pi/8)
### Rotate the vector given in the assignment file and save it as A12.

A11 = rotate(np.pi/8)

x = np.array([[np.pi/10],[2.1],[-math.e]])
A12 = rotate(np.pi/3)@x

### Find the vector x that was rotated to give you vector b.
### Save the vector x as A13

b = np.array([[1.4],[-np.pi/10],[2.8]])

A13 = la.solve(rotate(np.pi/6),b)

### Invert the R(3*pi/4) and save it as A14.
### Find the angle theta that would give you this inverse
### without having to do matrix operations, and save the angle
### as A15.

A14 = la.inv(rotate(3*np.pi/4))

A15 = -3*np.pi/4
