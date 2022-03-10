import numpy as np
import scipy.integrate
import math
######################
np.set_printoptions(precision = 9, linewidth=150)

# Problem 1
data = np.genfromtxt('population.csv', delimiter=',')
t = data[0, :]
N = data[1, :]
######################

### Determine your stepsize dt from the array t
dt = (np.abs(t[-1]-t[-2]))


### Use the appropriate second order differences from the Theory Lecture


### For dN/dt you will need to use a combination of the above differences,
### but the choice will be obvious based on which direction you can/cannot
### go in the horizontal axis.  Whenever possible use central difference;
### only use forward or backward when central is not possible.

# backward difference approximation for year 2020
A1 = (3*N[-1]-4*N[-2]+N[-3])/(2*dt)

# central difference approximation for year 1880
A2 = (N[t.tolist().index(1890-1900)]-N[t.tolist().index(1870-1900)])/(2*dt)

#forward difference approximation for year 1790
A3 = (-3*N[0]+4*N[1]-N[2])/(2*dt)

A4 = np.zeros([1,24])

for i in range(1,A4.size-1):
    A4[0][i] = (N[i+1]-N[i-1])/(2*dt)

A4[0][-1] = A1
A4[0][0] = A3

A5 = np.divide(A4,N.reshape(1,24))
A6 = np.mean(A5)

# Problem 2
data = np.genfromtxt('brake_pad.csv', delimiter=',')
r = data[0, :]
T = data[1, :]
######################

### Determine your stepsize dr from the array r

# rounded
dr = round(np.abs(r[1]-r[0]),5)

# # unrounded
# dr = np.abs(r[1]-r[0])
# print(dr)

theta = 0.7051

### Use the LHR formula from the coding lecture
A7 = 0
for k in range(r.size - 1):
    A7 = A7 + dr*T[k]*r[k]*theta


A_left = 0
for k in range(r.size-1):
    A_left = A_left + dr*r[k]*theta

A8 = A7/A_left

### Use the RHR formula from the coding lecture

A9 = 0
for k in range(1, r.size):
    A9 = A9 + dr*T[k]*r[k]*theta


A_right = 0
for k in range(1, r.size):
    A_right = A_right + dr*r[k]*theta

A10 = A9/A_right

### Use the Trapezoid rule formula or the trapz function from the coding lecture

A11 = (A7+A9)/2
A_trap = (A_left+A_right)/2

A12 = A11/A_trap


### Problem 3
### You'll have to use anonymous functions here.  You can see the syntax in
### the Numerical Integration coding lecture where the builtin function
### "integrate.quad" is used.

F = lambda x: math.pow(x,2)/2-math.pow(x,3)/3

f = lambda z, mu: mu/math.sqrt(F(mu)-F(mu*z))

A13, err13 = scipy.integrate.quad(f, 0, 1, args=(0.95,))
A14, err14 = scipy.integrate.quad(f, 0, 1, args=(0.5,))
A15, err15 = scipy.integrate.quad(f, 0, 1, args=(0.01,))