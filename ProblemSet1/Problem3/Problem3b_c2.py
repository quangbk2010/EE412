#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np 
from cvxopt import matrix, solvers

N = 8
X0 = np.array (([[6., 2.], [3.,1.], [6., 2.], [4., 3.],[2., 3.],[1., 5.],[1., 6.],[3., 4.]])) # class -1
X1 = np.array (([[11., 6.],[4., 10.],[2., 9.],[3., 8.],[6., 9.],[8., 6.],[8., 8.],[6., 6.]])) # class 11 
X = np.concatenate((X0, X1), axis = 0).T # all data 
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # labels 



# build K
V = np.concatenate((X0.T, -X1.T), axis = 1)
K = matrix(V.T.dot(V)) # see definition of V, K near eq (8)

p = matrix(-np.ones((2*N, 1))) # all-one vector 

# build A, b, G, h 
G = matrix(-np.eye(2*N)) # for all lambda_n >= 0
h = matrix(np.zeros((2*N, 1)))
A = matrix(y) # the equality constrain is actually y^T lambda = 0
b = matrix(np.zeros((1, 1))) 
solvers.options['show_progress'] = False
sol = solvers.qp(K, p, G, h, A, b)

l = np.array(sol['x'])
print('lambda = ')
for i in range (len(l)):
    print("%d\t%.4f" %(i, l.T[0][i]))

epsilon = 1e-6 # just a small number, greater than 1e-9
S = np.where(l > epsilon)[0]

VS = V[:, S]
XS = X[:, S]
yS = y[:, S]
lS = l[S]
# calculate w and b
w = VS.dot(lS)
b = np.mean(yS.T - w.T.dot(XS))

print('w = ', w.T)
print('b = ', b)