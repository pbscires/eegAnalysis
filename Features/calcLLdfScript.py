'''
Created on Dec 28, 2017

@author: pb8xe
'''
from numba import guvectorize
from numba import float64, float32

@guvectorize([(float64[:,:], float64[:], float64, float64[:,:])], '(n,n),(n),()->(n,n)', target='cuda')
def calcLLdf(sigDiffs, startingRowsArr, numSamplesPerEpoch, llMat):
#         timer2 = ElapsedTime()
#         timer2.reset()
        llMat = sigDiffs[startingRowsArr,]
        for j in range(1, numSamplesPerEpoch):
            startingRowsArr = startingRowsArr + 1
            llMat = llMat + sigDiffs[startingRowsArr,]
#             if (j % 256 == 0):
#                 print("j=", j, ", timer2 elapsed time=", timer2.timeDiff())
        llMat = llMat / numSamplesPerEpoch