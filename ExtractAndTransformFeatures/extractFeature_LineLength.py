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




epochLength = 1000 # In milliseconds
slidingWindowLength = 10 # In number of epochs

def lineLenForTimestamp(rowIndex):
    global numSamplesPerEpoch
    global allChannelsDF

    if (rowIndex < numSamplesPerEpoch):
        return
    else:
        row = allChannelsDF.iloc[rowIndex-1] - allChannelsDF.iloc[rowIndex]
        for j in range(2, numSamplesPerEpoch):
            row = row + (allChannelsDF.iloc[rowIndex-j] - 
                         allChannelsDF.iloc[rowIndex-j+1])
        return row


def calculateLineLength(filename, p):
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
        numSamplesPerEpoch = int(sampleFrequency * 1000 / epochLength)
        sigbufs = np.zeros((numChannels, numSamples))
        for i in np.arange(numChannels):
            sigbufs[i, :] = f.readSignal(i)
        # sifbufs above is a 23 x 921600 matrix
        # transpose it so that it becomes 921600 x 23 matrix
        sigbufs = sigbufs.transpose()
        allChannelsDF = pd.DataFrame(data = sigbufs[:,:], columns = signal_labels)
        llDf = pd.DataFrame(columns = signal_labels)
        print (allChannelsDF.shape)
#        for i in range(20):
#            allChannelsDF = allChannelsDF.add(other = sig[i, :])
        print (allChannelsDF.head())
        
        lineLenForTimestamp
        for i in range(1000):
            if (i > numSamplesPerEpoch):
                row = allChannelsDF.iloc[i-1] - allChannelsDF.iloc[i]
                for j in range(2, numSamplesPerEpoch):
                    row = row + (allChannelsDF.iloc[i-j] - allChannelsDF.iloc[i-j+1])
#                llDf = llDf.append(allChannelsDF.iloc[i] - allChannelsDF.iloc[i+1], 
#                                   ignore_index=True)
                llDf = llDf.append(row, ignore_index=True)
                
        print ("Printing Line Length Data frame for file", filename)
        print (llDf.head())
    return


p = Pool()
#p.map(calculateLineLength, [filesList[0]])
#print (filesList[0])
#calculateLineLength(filesList[0])
    
print (sys.argv[1])
calculateLineLength(sys.argv[1])
