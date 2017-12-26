'''

'''

import sys
from Features.LineLength import LineLength
import pyedflib
import re
import numpy as np

if __name__ == '__main__':
    edfFilePath  = sys.argv[1]
    print ("edfFilePath = ", edfFilePath)
    llObj = LineLength()
    
    if (re.search('\.edf$', edfFilePath) != None):
        f = pyedflib.EdfReader(edfFilePath)
        numChannels = f.signals_in_file
        print ("number of signals in file = ", numChannels)
        signal_labels = f.getSignalLabels()
        print ("signal labels = ", signal_labels)

        # numSamples = 3600 * 256 = 921,600
        numSamples = f.getNSamples()[0]
        # sampleFrequency = 256
        sampleFrequency = f.getSampleFrequency(0)
        sigbufs = np.zeros((numChannels, numSamples))
        for i in np.arange(numChannels):
            sigbufs[i, :] = f.readSignal(i)
        # sigbufs above is a 23 x 921600 matrix
        # transpose it so that it becomes 921600 x 23 matrix
        sigbufs = sigbufs.transpose()
    
    llObj.extractFeature(sigbufs, signal_labels, sampleFrequency)
    llObj.plotLLdf(signal_labels)
    llObj.saveLLdf(sys.argv[2])
