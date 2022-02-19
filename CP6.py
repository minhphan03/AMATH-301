import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
import scipy.linalg as la

np.set_printoptions(precision = 9, linewidth=150)

### Problem 1
data = np.genfromtxt('lynx.csv', delimiter=',')
t = data[0, :]
pop = data[1, :]
### Don't delete the lines above when submitting to gradescope

### Replace the value of the population for the years given in the assignment file and save it as A1
pop[1956-1946] = 34
pop[1974-1946] = 27
A1 = pop.copy().reshape(1,49)
print(A1)

### Calculate the value of the cubic spline interpolation of the data at t = 24.5 using the interp1d function.  Save this as A2.
xplot1 = np.arange(0, t.size-0.5, 0.5)
interp_func = si.interp1d(t, pop, kind='cubic')

yplot1 = interp_func(xplot1)
# print(yplot1)
A2 = yplot1[int(24.5*2)]
# plt.plot(xplot1, yplot1, t, pop, 'ko')
# plt.axvline(x = 24.5)
# plt.show()

### Use polyfit to calculate the coefficients for A3, A5, and A7
### Use norm to calculate the error for A4, A6, and A8

coeffs1 = np.polyfit(t,pop, 1)
A3 = coeffs1.reshape(1,2)
A4 = la.norm(np.polyval(coeffs1,t) - pop)

coeffs2 = np.polyfit(t,pop, 2)
A5 = coeffs2.reshape(1,3)
A6 = la.norm(np.polyval(coeffs2,t) - pop)

coeffs10 = np.polyfit(t,pop, 10)
A7 = coeffs10.reshape(1,11)
A8 = la.norm(np.polyval(coeffs10,t) - pop)

### Problem 2
data = np.genfromtxt('CO2_data.csv', delimiter=',')
t = data[0, :]
co2 = data[1, :]
### Don't delete the lines above when submitting to gradescope

### Use polyfit to calculate the coefficients for A9
### Use norm to calculate the error for A10
coeffs1 = np.polyfit(t,co2, 1)
A9 = coeffs1.reshape(1,2)
A10 = la.norm(np.polyval(coeffs1,t) - co2)


### Fit the exponential
b = 260
co2_proc = np.log(co2 - b)
coeffs1 = np.polyfit(t, co2_proc, 1)

a = np.exp(coeffs1[1])
r = coeffs1[0]

A11 = np.array([a,r,b]).reshape(1,3)
A12 = la.norm(np.exp(np.polyval(coeffs1, t))+b - co2)

### Fit the sinusoidal
### There are a few different ways to do this, and we will refrain from giving away the answer to this part.
### The class has been doing loops for a while now, so this part should be doable, albeit a little tricky.
### We can however check to see if there are any bugs that we can spot.
co2_proc = co2 - (np.exp(np.polyval(coeffs1, t))+b)

sum = 0
peaks = np.zeros(63)
troughs = peaks.copy()

j = 20
count = 0
#j stands for the index of the peak of the year before (the first year does not have the "yesteryear" so set j >0.5)
for i in range(len(t)-2):
    if co2_proc[i] < co2_proc[i+1] and co2_proc[i+1] > co2_proc[i+2] and np.abs(t[j]- t[i]) > 0.75:
        peaks[count] = co2_proc[i+1]
        count +=1
        j = i

j = 20
count = 0
for i in range(len(t) - 2):
    if co2_proc[i] > co2_proc[i + 1] and co2_proc[i + 1] < co2_proc[i + 2] and np.abs(t[j] - t[i]) > 0.75:
        troughs[count] = co2_proc[i + 1]
        count += 1
        j = i


year = np.arange(0,62,1)

# remove last year data
peaks = np.delete(peaks, [62])
print(peaks)
troughs = np.delete(troughs, [62])
print(troughs)
A = (np.sum(peaks)-np.sum(troughs))/(62*2)
B = 2*np.pi

A13 = np.array([A,B]).reshape(1,2)
print(A13)
A14 = la.norm((np.exp(np.polyval(coeffs1, t))+b + A*np.sin(B*t)) - co2)
print(A14)
plt.plot(t, co2_proc)
# plt.plot(t, co2, '.', t, np.exp(np.polyval(coeffs1, t))+b)
# plt.plot(t, co2, '.', t, np.exp(np.polyval(coeffs1, t))+b + A*np.sin(B*t))
plt.show()