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

mitDir = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/mit/chbmit01_12/chb01'

channelsList = list(range(23))

os.chdir(mitDir)
filesList = os.listdir(mitDir)
# print (filesList)

for filename in filesList:
    if (re.search('\.dat', filename) != None):
        fileBasename = os.path.splitext(filename)[0]
        print ("fileBasename = ", fileBasename)
        try:
            sig, fields = wfdb.srdsamp(fileBasename)
        except ValueError:
            print("ValueError occurred for fieBasename ", fileBasename)
            continue
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