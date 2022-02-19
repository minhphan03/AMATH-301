import numpy as np
import scipy.integrate
######################

# Problem 1
data = np.genfromtxt('population.csv', delimiter=',')
t = data[0, :]
N = data[1, :]
######################


### Determine your stepsize dt from the array t





### Use the appropriate second order differences from the Theory Lecture






### For dN/dt you will need to use a combination of the above differences,
### but the choice will be obvious based on which direction you can/cannot
### go in the horizontal axis.  Whenever possible use central difference;
### only use forward or backward when central is not possible.






# Problem 2
data = np.genfromtxt('brake_pad.csv', delimiter=',')
r = data[0, :]
T = data[1, :]
######################

### Determine your stepsize dr from the array r



### Use the LHR formula from the coding lecture





### Use the RHR formula from the coding lecture





### Use the Trapezoid rule formula or the trapz function from the coding lecture





### Problem 3
### You'll have to use anonymous functions here.  You can see the syntax in
### the Numerical Integration coding lecture where the builtin function
### "integrate.quad" is used.



