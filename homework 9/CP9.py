import numpy as np
import scipy.integrate
import scipy.linalg as la
import math
import matplotlib.pyplot as plot
######################

### Problem 1
### Use solve_ivp to solve the Fitzhugh-Nagumo IVP
### For the maxima use the plot to narrow down the times you use to search
### for the maximum.

def model1(t, x):
    I = 1/10*(5+np.sin(np.pi*t/10))
    dx1 = x[0] - 1/3*math.pow(x[0],3) - x[1] + I
    dx2 = (0.7 + x[0] - x[1])/12
    return np.array([dx1, dx2])

t = np.arange(0,100+0.5, 0.5)
tspan = (0,100)

v0 = 1
w0 = 0
x = np.array([v0, w0])

sol = scipy.integrate.solve_ivp(model1, tspan,x, t_eval=t)

v = sol.y[0,:]
A1 = v.reshape([201,1])
t_max = 0
v_max = -2
for i in range(1, 21):
    if v[i] > v_max:
        v_max = v[i]
        t_max = t[i]
    else:
        break
A2 = t_max

t_max = 0
v_max = -2
for i in range(80, 100):
    if v[i] > v_max:
        v_max = v[i]
        t_max = t[i]
    else:
        break
A3 = t_max

A4 = 1/(A3-A2)

# plot.plot(t,v)
# plot.show()

### Problem 2
### Use solve_ivp to solve the Chua equation
### You can tell something is chaotic if it is seemingly random
### If it looks like all solutions tend toward a point or a circle it is
### not chaotic.

def model2(t, x):
    dx1 = 16*(x[1]+1/6*x[0]-1/16*math.pow(x[0],3))
    dx2 = x[0]-x[1]+x[2]
    dx3 = -30*x[1]
    return np.array([dx1, dx2, dx3])

x0 = 0.1
y0 = 0.2
z0 = 0.3

x = np.array([x0,y0,z0])
tspan = (0,100)
t = np.arange(0,100+0.05,0.05)

sol1 = scipy.integrate.solve_ivp(model2, tspan,x, t_eval=t)


#
# plot.plot(t, sol1.y[0,:])
# plot.plot(t, sol2.y[0,:])
# plot.show()

A5 = 1
A6 = sol1.y.reshape([3,2001])

x2 = np.array([x0,y0+1e-5,z0])
sol2 = scipy.integrate.solve_ivp(model2, tspan,x2, t_eval=t)

A7 = np.max(np.abs(sol1.y-sol2.y))

# def model3(t, x):
#     dx1 = 16*(x[1]+1/6*x[0]-1/16*math.pow(x[0],3))
#     dx2 = x[0]-x[1]+x[2]
#     dx3 = -100*x[1]
#     return np.array([dx1, dx2, dx3])
#
# x0 = 0.1
# y0 = 0.2
# z0 = 0.3
#
# x = np.array([x0,y0,z0])
# tspan = (0,100)
# t = np.arange(0,100+0.05,0.05)
#
# sol1 = scipy.integrate.solve_ivp(model3, tspan,x, t_eval=t)
#
# x2 = np.array([x0,y0+1e-5,z0])
# sol2 = scipy.integrate.solve_ivp(model3, tspan,x2, t_eval=t)
#
# plot.plot(t, sol1.y[0,:])
# plot.plot(t, sol2.y[0,:])
# plot.show()

A8 = 0

### Problem 3
### Part 1: Finite Differences
### Use finite differences to solve the BVP
### Be careful about the shape of the vectors, you may have to transpose to
### get the correct shape.  It's a good idea to print the solutions out to
### make sure the shape is correct.

dt = 0.1
t = np.arange(0,6+dt,dt)

v = -2 * np.ones(t.size - 2)
u = np.ones(t.size - 3)
A = (1 / dt ** 2) * (np.diag(v) + np.diag(u, 1) + np.diag(u, -1))
I = np.diag(np.ones(t.size-2).reshape(-1))

# f''(t) +f(t) = 5*cos(4t)
b = 5*np.cos(4*t[1:-1])

b[0] = b[0] - 1/ dt ** 2
b[-1] = b[-1] - 0.5/ dt ** 2
b = b.reshape((-1, 1))

A9 = A+I
A10 = b
x_int = la.solve(A9, A10)

A11 = np.zeros(t.size)
A11[0] = 1
A11[-1] = 0.5
A11[1:-1] = x_int.reshape(-1)
A11 = A11.reshape([61,1])

x_true = (1/2+1/3*np.cos(24)-4/3*np.cos(6))/np.sin(6)*np.sin(t) + 4/3*np.cos(t)-1/3*np.cos(4*t)

plot.plot(t, x_true, t, A11, '--')

A12 = np.max(np.abs(A11-x_true))

### Part 2: Bisection
### Use the shooting method to solve the BVP
### It's a good idea to test out a few in the command window first to make
### sure that your initial conditions gets you to different sides of the right
### boundary condition.
### Use the plot to help you figure out what your choices of initial
### conditions should be

# Let f'(t) = y(t)
def model4(t,x):
    I = 5*np.cos(4*t)
    dx1= x[1]
    dx2= I - x[0]
    return np.array([dx1,dx2])

t = np.arange(0, 6.1, 0.1)

v_left = 2
v_right = 3
x0 = 1
xT = 0.5

v_mid = (v_left+v_right)/2

sol_mid = scipy.integrate.solve_ivp(model4, (0,6), np.array([x0,v_mid]), t_eval = t)
x_mid = sol_mid.y[0,:]
sol_left = scipy.integrate.solve_ivp(model4, (0,6), np.array([x0,v_left]), t_eval = t)
x_left = sol_left.y[0,:]
sol_right = scipy.integrate.solve_ivp(model4, (0,6), np.array([x0,v_right]), t_eval = t)
x_right = sol_right.y[0,:]

while np.abs(x_mid[-1]-xT) > 1e-8:
    if x_mid[-1] == xT:
        break
    elif np.sign(x_mid[-1]-xT) == np.sign(x_left[-1] - xT):
        v_left = v_mid
        sol_left = scipy.integrate.solve_ivp(model4, (0,6), np.array([x0,v_left]), t_eval = t)
        x_left = sol_left.y[0,:]
    else:
        v_right = v_mid
        sol_right = scipy.integrate.solve_ivp(model4, (0,6), np.array([x0,v_right]), t_eval = t)
        x_right = sol_right.y[0,:]

    v_mid = (v_left+v_right)/2
    sol_mid = scipy.integrate.solve_ivp(model4, (0, 6), np.array([x0, v_mid]), t_eval=t)
    x_mid = sol_mid.y[0,:]

A13 = x_mid.reshape([61,1])
A14 = np.max(np.abs(A13-x_true))
A15 = np.max(np.abs(A13-A11))
print(A15)
print(A12)

# print(A12)
# print(A15)
# print(A14)
# plot.plot(t,x_mid,'b',t, x_true, 'g',t, A11, 'r')
#
# plot.show()
# plot.plot(t, x)
# plot.show()