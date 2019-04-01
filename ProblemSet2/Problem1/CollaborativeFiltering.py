# -*- coding: utf-8 -*-

import numpy as np

R = np.array ([[5,4,4,-1,5], [-1,3,5,3,4], [5,2,-1,2,3], [-1,2,3,1,2], [4,-1,5,4,5], [5,3,-1,3,5], [3,2,3,2,-1], [5,3,4,-1,5], [4,2,5,4,-1], [5,-1,5,3,4]])
R1 = np.array ([[5,4,4,-1,0], [-1,3,5,0,4], [5,2,-1,0,3], [-1,0,3,1,2], [4,-1,0,4,5], [0,3,-1,3,5], [3,0,3,2,-1], [5,0,4,-1,5], [0,2,5,4,-1], [0,-1,5,3,4]])

bu = np.array ([0.62,0.42,-0.28,-1.78,0.52,0.49,-1.24,0.45,0.4,0.23])

bi = np.array ([0.72,-1.2,0.6,-0.6,0.33])

rbar = 3.67

Rhat = np.zeros ((10, 5))
for u in range (10):
    for i in range (5):
        if R[u][i] != -1:
            Rhat[u][i] = rbar + bu[u] + bi[i]
            
print ("Rhat", Rhat[6:,:])

def get_rmse (actual_R, predict_R):
    deltaR = actual_R - predict_R
    #print (deltaR)
    train_mse = 0
    test_mse = 0
    for u in range (10):
        for i in range (5):
            if R1[u][i] == 0:
                test_mse += deltaR[u][i] ** 2
                #print (deltaR[u][i])
            elif R1[u][i] != -1:
                train_mse += deltaR[u][i] ** 2

    return (np.sqrt (train_mse/10), np.sqrt (test_mse/10))

print ("baseline: rmse = ", get_rmse (R, Rhat))

for count in range (2):
    Rwave = np.zeros ((10, 5))
    for u in range (10):
        for i in range (5):
            if R1[u][i] != -1 and R1[u][i] != 0:
                Rwave[u][i] = R[u][i] - Rhat[u][i]
    
    #print ("Rwave", Rwave)
    
    D = np.zeros ((5,5))
    for i in range (5):
        for j in range (5):
            a = Rwave[:, i]
            b = Rwave[:, j]
            
            tu = np.dot (a, b)
            a_ = []
            b_ = []
            for k in range (10):
                if a[k] * b[k] != 0:
                    a_.append (a[k])
                    b_.append (b[k])
                    
            if (i != j):
                D[i][j] = tu / (np.linalg.norm (a_) * np.linalg.norm (b_))
    
            
    #print ("D", D)
    
    
    Rhat_neighbor = np.zeros ((10, 5))
    for u in range (10):
        for i in range (5):
            a = 0
            b = 0
            for j in range (5):
                a += D[i][j] * Rwave[u][j]
                if Rwave[u][j] != 0:
                    b += np.abs (D[i][j])
            #if i == 3 and u == 2:
                #print ("Rhat[u][i]", Rhat[u][i], "a",a, "b", b)
                
            Rhat_neighbor[u][i] = Rhat[u][i] + a/b
            
    print ("Rhat_neighbor:", Rhat_neighbor[6:])
    Rhat = Rhat_neighbor
    print ("Neighbor" + str (count) + ": rmse = ", get_rmse (R, Rhat))