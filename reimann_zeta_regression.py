# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:59:40 2020

@author: mythkc
"""


import numpy as np
import sympy
import pandas as pd

from sklearn.neural_network import MLPRegressor

#################################### TRAINING DATA ##################

Q = []
count = 1

for line in open('100000zeroes.txt'):
    Q += [[count, float(line.rstrip('\n'))]]
    count += 1

X = np.array(Q [0:10000])
print(X)


neural_net = MLPRegressor([500, 500], random_state=9, max_iter=2000).fit(X[:,:-1], X[:,-1])


###################################  TESTING DATA ##############################################################
"""

Q2 = []
count = 1

for line in open('TESTINGDATA.txt'):
    Q2 += [count, float(line.rstrip('\n'))]
    count += 1

print (Q2)
"""

testingdata = np.array(Q)


residuals = []

Y = []
percent_error = []

for i in range (0, len(testingdata)):
    nnresult = float(neural_net.predict([[i]]))
    actualnumber = testingdata[i][1]
    
    Y += [[i+1, nnresult]]
    residuals += [[i+1, nnresult - actualnumber]]
    percent_error += [[i+1, 100*abs(nnresult - actualnumber)/actualnumber]]


df2 = pd.DataFrame(percent_error, columns=['index', 'percent error'])
ax2 = df2.plot.scatter(x='index',
                       y='percent error',
                       c ='DarkBlue')

print (percent_error)





