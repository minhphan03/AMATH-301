import numpy as np
import scipy.integrate
import math
######################

### Problem 1
### Initialize t, and x_true

upper = 10
delta = 0.1

t = np.arange(0, upper+delta, delta)
x_true = 1/2*(np.cos(t)+np.sin(t)+np.exp(-t))

f = lambda t,x: np.cos(t)-x

### Forward Euler
### Write a forward Euler scheme

x = np.zeros(t.size)
x[0] = 1
for n in range(1, 101):
    x[n] = x[n-1]+delta*f(t[n-1],x[n-1])

A1 = x.reshape([1,101])
A2 = np.abs(x-x_true).reshape([1,101])

### Backward Euler
### Write a backward Euler scheme

f = lambda t,x,delta: (x+delta*np.cos(t))/(1+delta)

x = np.zeros(t.size)
x[0] = 1
for n in range(1,101):
    x[n] = f(n*delta,x[n-1],0.1)

A3 = x.reshape([1,101])
A4 = np.abs(x-x_true).reshape([1,101])

### Built-in Solver
### Use scipy.integrate.solve_ivp
### Don't forget to reshape the solution you get from scipy.integrate.solve_ivp

t = np.arange(0, 10+0.1, 0.1)
f = lambda t,x: np.cos(t)-x
tspan = (0, 10)
sol = scipy.integrate.solve_ivp(f, tspan, np.array([1]),t_eval=t)
x = sol.y[0,:]
A5 = x.reshape([1,101])
A6 = np.abs(A5-x_true).reshape([1,101])

### Problem 2
### Initialize the parameters

T = 2

### Forward Euler for dt = 0.01
### Initialize t and x_true
### Write a forward Euler scheme

dt = 0.01
t = np.arange(0,dt+T, dt)
f = lambda t,x: 8*np.sin(x)

x = np.zeros(t.size)
x[0] = np.pi/4
for n in range(1,201):
    x[n] = x[n-1]+dt*f(t[n-1],x[n-1])

A7 = x.reshape([1,201])
# true solution
x_true = 2*np.arctan(np.exp(8*t)/(1+math.sqrt(2)))

A8 = np.max(np.abs(x_true - x))

### Forward Euler dt = 0.001
### Reinitialize t and x_true
### Write a forward Euler scheme

dt = 0.001
t = np.arange(0,dt+T, dt)
x = np.zeros(t.size)
x[0] = np.pi/4
for n in range(1,2001):
    x[n] = x[n-1]+dt*f(t[n-1],x[n-1])

x_true = 2*np.arctan(np.exp(8*t)/(1+math.sqrt(2)))
A9 = A8/np.max(np.abs(x_true-x))

### Predictor-Corrector dt = 0.01
### Reinitialize t and x_true
### Write a forward Euler scheme and a backward Euler scheme in the same loop.
### The forward Euler scheme is the predictor.  The answer from forward
### Euler will be plugged into the sin(x_n+1) in the backward Euler scheme.

dt = 0.01
t = np.arange(0,dt+T, dt)
f = lambda t,x: 8*np.sin(x)

x_forward = np.pi/4

x_backward = np.zeros(t.size)
x_backward[0] = np.pi/4

x_true = 2*np.arctan(np.exp(8*t)/(1+math.sqrt(2)))

for n in range(1,201):
    x_forward = x_backward[n-1]+dt*f(t[n],x_backward[n-1])
    x_backward[n] = x_backward[n-1]+dt*f(t[n],x_forward)
A10 = x_backward.reshape([1,201])
A11 = np.max(np.abs(x_true - x_backward))
### Predictor-Corrector dt = 0.001
### Reinitialize t and x_true
### Write a forward Euler scheme and a backward Euler scheme in the same loop.
### The forward Euler scheme is the predictor.  The answer from forward
### Euler will be plugged into the sin(x_n+1) in the backward Euler scheme.

dt = 0.001
t = np.arange(0,dt+T, dt)
f = lambda t,x: 8*np.sin(x)

x_forward = np.pi/4

x_backward = np.zeros(t.size)
x_backward[0] = np.pi/4

x_true = 2*np.arctan(np.exp(8*t)/(1+math.sqrt(2)))

for n in range(1,2001):
    x_forward = x_backward[n-1]+dt*f(t[n],x_backward[n-1])
    x_backward[n] = x_backward[n-1]+dt*f(t[n],x_forward)

A12 = A11/np.max(np.abs(x_true - x_backward))

### Builtin Solver
### Reinitialize t and x_true
### Use scipy.integrate.solve_ivp
### Don't forget to reshape the solution you get from scipy.integrate.solve_ivp

t = np.arange(0, 2.01, 0.01)
x_true = 2*np.arctan(np.exp(8*t)/(1+math.sqrt(2)))
f = lambda t,x: 8*np.sin(x)
tspan = (0, 2)

sol = scipy.integrate.solve_ivp(f, tspan, np.array([np.pi/4]),t_eval=t)
x = sol.y[0,:]

A13 = x.reshape([1,201])
A14 = np.max(np.abs(x-x_true))

t = np.arange(0, 2.001, 0.001)
x_true = 2*np.arctan(np.exp(8*t)/(1+math.sqrt(2)))

sol = scipy.integrate.solve_ivp(f, tspan, np.array([np.pi/4]),t_eval=t)
A15 = A14/np.max(np.abs(sol.y[0,:]-x_true))

print(A10)
print(A11)
print(A12)
