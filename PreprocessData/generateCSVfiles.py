#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:45:36 2017

@author: rsburugula
"""

import wfdb
import os
import numpy as np
import re
import sys
from multiprocessing import Pool

mitDir = sys.argv[1]
print ("mitDir = ", mitDir)
os.chdir(mitDir)
channelsList = list(range(23))

filesList = os.listdir(mitDir)
# print (filesList)

def generateCSVPerFile(filename):
    if (re.search('\.dat', filename) != None):
        fileBasename = os.path.splitext(filename)[0]
        print ("fileBasename = ", fileBasename)
        try:
            sig, fields = wfdb.srdsamp(fileBasename)
        except ValueError:
            print("ValueError occurred for fieBasename ", fileBasename)
            return
        numChannels = len(fields['signame'])
        for chIndex in range(numChannels):
            channelName = fields['signame'][chIndex]
            perChannelFilename = '.'.join([fileBasename, channelName, 'csv'])
            print ("perChannelFilename=", perChannelFilename)
            
            numSamples = fields['fs'] * 3600
            fileHandle = open(perChannelFilename, 'w')
            for chIndex in range(numChannels):
                for i in range(numSamples):
                    timeStamp = i * fields['fs']
                    fileHandle.write(','.join([str(timeStamp), str(sig[i, chIndex])]))
                    fileHandle.write('\n')
            fileHandle.close()
    return


p = Pool()
p.map(generateCSVPerFile, filesList)
#for filename in filesList:
#    generateCSVPerFile(filename)

#
#sig, fields = wfdb.srdsamp('../../Data/tmp/chb01_01')
#print (fields)
#print (type(sig))
#print (sig.shape, sig.dtype)

#numSamples = fields['fs'] * 3600 # ts contains the samples per second

#print (sig)
#
#for i in range(23):
#    for j in range(numSamples):
#        print (sig[j, i])