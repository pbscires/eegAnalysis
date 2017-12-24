#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pranavburugula
"""

import os
import numpy as np
import pandas as pd
import re
import sys
import pyedflib
from multiprocessing import Pool
import matplotlib.pyplot as plt
from util.ElapsedTime import ElapsedTime




epochLength = 10000 # In milliseconds
slidingWindowLength = 2000 # In milliseconds

# Sliding window length should be < epocLength
if (slidingWindowLength > epochLength):
    print ("Invalid values for sliding window length and/or epoch length")
    exit(-1)

def calculateLineLength(filename):
    global numSamplesPerEpoch
    global allChannelsDF

    if (re.search('\.edf', filename) != None):
        f = pyedflib.EdfReader(sys.argv[1])
        numChannels = f.signals_in_file
        print ("number of signals in file = ", numChannels)
        signal_labels = f.getSignalLabels()
        print ("signal labels = ", signal_labels)

        # numSamples = 3600 * 256 = 921,600
        numSamples = f.getNSamples()[0]
        # sampleFrequency = 256
        sampleFrequency = f.getSampleFrequency(0)
        numSamplesPerEpoch = int(sampleFrequency * epochLength / 1000)
        sigbufs = np.zeros((numChannels, numSamples))
        for i in np.arange(numChannels):
            sigbufs[i, :] = f.readSignal(i)
        # sigbufs above is a 23 x 921600 matrix
        # transpose it so that it becomes 921600 x 23 matrix
        sigbufs = sigbufs.transpose()
        allChannelsDF = pd.DataFrame(data = sigbufs, columns = signal_labels)
        print ("Shape of allChannelsDf = ", allChannelsDF.shape)
        print (allChannelsDF.head(10))
        
        sigDiffs = np.delete(sigbufs, numSamples-1, 0) - np.delete(sigbufs, 0, 0)
        sigDiffs = np.absolute(sigDiffs)
        print ("Shape of sigDiffs = ", sigDiffs.shape)
        lastRowInd = numSamples - 2
        endOfHeadInd = 0
        begOfTailInd = lastRowInd-numSamplesPerEpoch+1

        startingRowsArr = np.arange(0, numSamples, slidingWindowLength)
        print (startingRowsArr.shape)
#         print(startingRowsArr)
        j = len(startingRowsArr)
        j -= 1
        print ("j=", j, "startingRowsArr[", j, "] = ", startingRowsArr[j])
        lastEpochIndex = numSamples - (epochLength*sampleFrequency / 1000)
        print ("lastEpochIndex = ", lastEpochIndex)
        while (startingRowsArr[j] > lastEpochIndex):
            j -= 1
            print ("j=", j, "startingRowsArr[", j, "] = ", startingRowsArr[j])
        startingRowsArr = np.delete(startingRowsArr, range(j,len(startingRowsArr)))
        print ("j=", j)
        print (startingRowsArr.shape)
        
        llMat = np.delete(sigDiffs, list(range(begOfTailInd, lastRowInd+1)), 0)
        timer2 = ElapsedTime()
        for j in range(1, numSamplesPerEpoch):
            timer2.reset()
            endOfHeadInd += 1
            begOfTailInd += 1
            rowsToDel = list(range(endOfHeadInd)) + list(range(begOfTailInd, lastRowInd+1))
#             print ("Removing the rows", rowsToDel)
            llMat = llMat + np.delete(sigDiffs, rowsToDel, 0)
            print("j=", j, ", timer2 elapsed time=", timer2.timeDiff())
        llDf = pd.DataFrame(data = llMat, columns = signal_labels)
                
        print ("Printing Line Length Data frame for file", filename)
        print (llDf.head())
        print (llDf.shape)
        plt.figure()
        plt.plot(llDf)
        plt.show()
    return


# p = Pool()
#p.map(calculateLineLength, [filesList[0]])
#print (filesList[0])
#calculateLineLength(filesList[0])
timer1 = ElapsedTime()
timer1.reset()
print (sys.argv[1])
calculateLineLength(sys.argv[1])
print(timer1.timeDiff())
