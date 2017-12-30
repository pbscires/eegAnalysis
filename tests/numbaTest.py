'''
Created on Dec 29, 2017

@author: pb8xe
'''
from numba import vectorize, float32, guvectorize, int64
import numpy as np
from numpy import dtype

@vectorize(["float32(float32, float32)"], target='cuda')
def VectorAdd(a,b):
    return a+b

@guvectorize(['void(int64[:], int64[:], int64[:])', 
              'void(int32[:], int32[:], int32[:])'], '(n),(n)->(n)', target='cuda')
def g(x, y, res):
    for i in range(x.shape[0]):
        res[i] = x[i] + y[i]

def main():
    N = 32000000
    
    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.zeros(N, dtype=np.float32)
    print(C)
    C = VectorAdd(A, B)
    
    print(C)
    
    a = np.arange(5)
    b = np.zeros(len(a), dtype=np.int64)
    c = g(a, b)
    print (c)
#     a = np.arange(6).reshape(2, 3)
#     print (g(a,10))
    
if __name__ == '__main__':
    main()