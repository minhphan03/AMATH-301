import scipy.io
import numpy as np
import scipy.integrate
import scipy.linalg
import math
#########################


### Problem 1
### First model the problem, and then solve it using solve_ivp




### Problem 2
### Use finite differences for boundary value problems and loop to iterate each timestep.



### Problem 3
### You need the following code to load the matrices
M1 = scipy.io.loadmat('CP10_M1.mat')
M1 = np.squeeze(M1['M1'].astype('float32'))
M2 = scipy.io.loadmat('CP10_M2.mat')
M2 = np.squeeze(M2['M2'].astype('float32'))
M3 = scipy.io.loadmat('CP10_M3.mat')
M3 = np.squeeze(M3['M3'].astype('float32'))
######################################
