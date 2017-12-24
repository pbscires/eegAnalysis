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




epochLength = 1000 # In milliseconds
slidingWindowLength = 10 # In number of epochs

def lineLenForTimestamp(rowIndex):
    global numSamplesPerEpoch
    global allChannelsDF
    global rowDiffs

    if (rowIndex < numSamplesPerEpoch):
        return
    else:
        row = allChannelsDF.iloc[rowIndex-1] - allChannelsDF.iloc[rowIndex]
        for j in range(2, numSamplesPerEpoch):
            row = row + (allChannelsDF.iloc[rowIndex-j] - 
                         allChannelsDF.iloc[rowIndex-j+1])
        return row


def calculateLineLength(filename):
    global numSamplesPerEpoch
    global allChannelsDF
    global rowDiffsDf

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
        numSamplesPerEpoch = int(sampleFrequency * 1000 / epochLength)
        sigbufs = np.zeros((numChannels, numSamples))
        for i in np.arange(numChannels):
            sigbufs[i, :] = f.readSignal(i)
        # sifbufs above is a 23 x 921600 matrix
        # transpose it so that it becomes 921600 x 23 matrix
        sigbufs = sigbufs.transpose()
        allChannelsDF = pd.DataFrame(data = sigbufs, columns = signal_labels)
        rowDiffsDf = pd.DataFrame(columns = signal_labels)
        print ("Shape of allChannelsDf = ", allChannelsDF.shape)
        print (allChannelsDF.head(10))
        
        sigDiffs = np.delete(sigbufs, numSamples-1, 0) - np.delete(sigbufs, 0, 0)
        sigDiffs = np.absolute(sigDiffs)
        print ("Shape of sigDiffs = ", sigDiffs.shape)
        lastRowInd = numSamples - 2
        endOfHeadInd = 0
        begOfTailInd = lastRowInd-numSamplesPerEpoch+1
        llMat = np.delete(sigDiffs, range(begOfTailInd, lastRowInd+1), 0)
        for j in range(1, numSamplesPerEpoch):
            endOfHeadInd += 1
            begOfTailInd += 1
            rowsToDel = list(range(endOfHeadInd)) + list(range(begOfTailInd, lastRowInd+1))
#             print ("Removing the rows", rowsToDel)
            llMat = llMat + np.delete(sigDiffs, rowsToDel, 0)
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
    
print (sys.argv[1])
calculateLineLength(sys.argv[1])
