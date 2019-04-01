#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 16:28:21 2017

@author: quang
"""

'''import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers

# Generate data
"""X0 = np.array (([[6, 2], [3,1], [6, 2], [4, 3],[2, 3],[1, 5],[1, 6],[3, 4]])).T
X1 = np.array (([[11, 6],[4, 10],[2, 9],[3, 8],[6, 9],[8, 6],[8, 8],[6, 6]])).T

plt.plot (X0[0, :], X0[1, :], 'or')
plt.plot (X1[0, :], X1[1, :], 'ob')

X = np.concatenate ((X0, X1), axis = 1)
y = np.concatenate ((np.ones ((1, X0.shape[1])), -1 * np.ones ((1, X1.shape[1]))), axis = 1)
"""
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # class 1
X1 = np.random.multivariate_normal(means[1], cov, N) # class -1 
X = np.concatenate((X0.T, X1.T), axis = 1) # all data 
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels 


V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V)) # see definition of V, K near eq (8)

p = matrix(-np.ones((X0.shape[1] + X1.shape[1], 1))) # all-one vector 
print (p.size[0], p.size[1])
# build A, b, G, h 
G = matrix(-np.eye(X0.shape[1] + X1.shape[1])) # for all lambda_n >= 0
h = matrix(np.zeros((X0.shape[1] + X1.shape[1], 1)))
A = matrix(y) # the equality constrain is actually y^T lambda = 0
b = matrix(np.zeros((1, 1))) 
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
print('lambda = ')
print(l.T)'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

X0 = np.array (([[6, 2], [3,1], [6, 2], [4, 3],[2, 3],[1, 5],[1, 6],[3, 4]])).T
X1 = np.array (([[11, 6],[4, 10],[2, 9],[3, 8],[6, 9],[8, 6],[8, 8],[6, 6]])).T

plt.plot (X0[0, :], X0[1, :], 'or')
plt.plot (X1[0, :], X1[1, :], 'ob')

X = np.concatenate ((X0, X1), axis = 1)
y = np.concatenate ((np.ones ((1, X0.shape[1])), -1 * np.ones ((1, X1.shape[1]))), axis = 1)

y1 = y.reshape((16,))
X1 = X.T # each sample is one row
clf = SVC(kernel = 'linear', tol = 1e-50)#, C = 1e5) # just a big number 

clf.fit(X1, y1) 

w = clf.coef_
b = clf.intercept_
print('w = ', w)
print('b = ', b)

x_ = np.linspace (0, 15, 100)
y_= -(b + w[0][0] * x_) / w[0][1]
plt.plot (x_, y_)
plt.show()