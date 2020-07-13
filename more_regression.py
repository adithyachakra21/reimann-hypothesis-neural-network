# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:23:29 2020

@author: mythkc
"""

import numpy as np
import sympy
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D
import os
if not os.path.isdir('images'):
    os.mkdir('images')
    
    
############################### PRINTING (INDEXED) PRIMES FROM 2 TO N ################################    

def primesfrom2to(n):
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n//3 + (n%6==2), dtype=np.bool)
    for i in range(1,int(n**0.5)//3+1):
        if sieve[i]:
            k=3*i+1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    return list (np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)] )
        

def indexedprimesfrom2to(n):
    k = primesfrom2to(n)
    output = []
       
    for i in range( len(k)):
        output += [[i+1,k[i]]]
    
    return output

Q = indexedprimesfrom2to(100000) # training data

X = np.array(Q)
 
df = pd.DataFrame(Q, columns=['index', 'prime'])
ax1 = df.plot.scatter(x='index',
                       y='prime',
                       c ='DarkBlue')

################################### NEURAL NETWORK ####################################################

neural_net = MLPRegressor([500, 500], random_state=9, max_iter=2000).fit(X[:,:-1], X[:,-1])

###################################  TESTING DATA ##################################################
testingdata = indexedprimesfrom2to(1000000)

residuals = []

Y = []
percent_error = []

for i in range (0, len(testingdata)):
    nnresult = float(neural_net.predict([[i]]))
    actualnumber = testingdata[i][1]
    
    Y += [[i+1, nnresult]]
    residuals += [[i+1, nnresult - actualnumber]]
    percent_error += [[i+1, 100*abs(nnresult - actualnumber)/actualnumber]]


df2 = pd.DataFrame(percent_error, columns=['index', 'percenterror'])
ax2 = df2.plot.scatter(x='index',
                       y='percenterror',
                       c ='DarkBlue')

print (percent_error)










