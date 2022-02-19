import numpy as np
import math
import matplotlib.pyplot as plt

### Problem 1
### Implement the Bisection method as we did in the Week 5 Coding Lecture

x = np.arange(0, 10, 0.1)
c = 1.3*(np.exp(-x/11)-np.exp(-4*x/3))
dc = 1.3*(-1/11*np.exp(-x/11)+4/3*np.exp(-4*x/3))

# set boundaries
a = 1
b = 3
mid = (a+b)/2
dc_mid = 1.3*(-1/11*np.exp(-mid/11)+4/3*np.exp(-4*mid/3))

dc_a = 1.3 * (-1 / 11 * np.exp(-a / 11) + 4 / 3 * np.exp(-4 * a / 3))
dc_b = 1.3 * (-1 / 11 * np.exp(-b / 11) + 4 / 3 * np.exp(-4 * b / 3))

while np.abs(dc_mid) > 1e-8:
    if dc_mid == 0:
        break
    elif np.sign(dc_mid) == np.sign(dc_a):
        a = mid
    else:
        b = mid

    mid = (a+b)/2

    dc_mid = 1.3 * (-1 / 11 * np.exp(-mid / 11) + 4 / 3 * np.exp(-4 * mid / 3))
    dc_a = 1.3 * (-1 / 11 * np.exp(-a / 11) + 4 / 3 * np.exp(-4 * a / 3))
    dc_b = 1.3 * (-1 / 11 * np.exp(-b / 11) + 4 / 3 * np.exp(-4 * b / 3))

A1 = mid
A2 = 1.3*(np.exp(-mid/11)-np.exp(-4*mid/3))
A3 = np.abs(dc_mid)

### Problem 2
### Implement Newton's method as we did in the Week 5 Coding Lecture

# initialize parameters
x = 2
dx = 2*x
i = 1
# y = x^2
while abs(dx) > 1e-8:
    i = i + 1
    x = x - (2*x)/2
    dx = 2*x

A4 = i
A5 = x

# reinitialization
x = 2
dx = (500*math.pow(x,499))
i = 1
# y = x^500
while abs(dx) > 1e-8:
    i += 1
    x = x - (500*math.pow(x,499))/(500*499*math.pow(x,498))
    dx = (500*math.pow(x,499))

A6 = i
A7 = x

# reinitialization
x = 2
dx = 1000*math.pow(x,999)
i = 1
# y = x^1000
while abs(dx) > 1e-8:
    i += 1
    x = x - (1000*math.pow(x,999))/(1000*999*math.pow(x,998))
    dx = 1000*math.pow(x,999)


A8 = i
A9 = x


# # plot
# plt.plot(x,c, x, dc)
# plt.axhline(c="black")
# plt.axvline(c="black")
# plt.show()

