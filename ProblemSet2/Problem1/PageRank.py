# -*- coding: utf-8 -*-

import numpy as np

H = np.array ([[0,1/2,1/2,0,0,0,0],[1/2,0,0,0,1/2,0,0],[0,1/3,0,1/3,1/3,0,0],[0,0,0,0,0,1/2,1/2],[0,0,0,1/2,0,1/2,0], [1/7,1/7,1/7,1/7,1/7,1/7,1/7], [0,0,0,0,1/2,1/2,0]])

J = 1/7 * np.ones ((7,7))

G = 0.8 * H + 0.2 * J

def mul_matrix (M, n):
    M_bar = np.identity (7)
    for i in range (n):
        M_bar = np.dot (M_bar, M)
    return M_bar

def rank_pages (m):
    mu0 = np.ones ((1, 7))
    for i in range (m):
        mu = np.dot (mu0, mul_matrix (G, i))
        print ("iteration: ", i, "---", mu)
        
rank_pages (20)