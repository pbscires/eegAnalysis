'''

'''
import numpy as np
import pyedflib
import re

class EEGRecord(object):
    '''
    This class represents a single EEG record. i.e. one .edf file wich contains
    values for 23 channels over a period of 1 hour. 
    '''


    def __init__(self, filePath, subjectName):
        '''
        Constructor
        '''
        self.filePath = filePath
        self.subjectName = subjectName
    
    def loadFile(self):
        f = pyedflib.EdfReader(self.filePath)
        self.numChannels = f.signals_in_file
        print ("number of signals in file = ", self.numChannels)
        self.signal_labels = f.getSignalLabels()
        print ("signal labels = ", self.signal_labels)

        # numSamples = 3600 * 256 = 921,600
        self.numSamples = f.getNSamples()[0]
        # sampleFrequency = 256
        self.sampleFrequency = f.getSampleFrequency(0)
        self.sigbufs = np.zeros((self.numChannels, self.numSamples))
        for i in np.arange(self.numChannels):
            self.sigbufs[i, :] = f.readSignal(i)
        # sigbufs above is a 23 x 921600 matrix
        # transpose it so that it becomes 921600 x 23 matrix
        self.sigbufs = self.sigbufs.transpose()
        
        # cleanup the sigbufs and signal_labels so that duplicate signals
        # are eliminated.
        columnsToDel = []
        labelsRead = []
        for i in range(len(self.signal_labels)):
            if (re.search('\w+\d+-\w+\d+', self.signal_labels[i]) == None):
                columnsToDel.append(i)
            if (self.signal_labels[i] in labelsRead):
                if (i not in columnsToDel):
                    columnsToDel.append[i]
        
        # Now remove the columns from sig_labels and sigbufs
        self.sigbufs = np.delete(self.sigbufs, columnsToDel, axis=1)
        self.signal_labels = np.delete(self.signal_labels, columnsToDel, axis=0)
        print ("new signal labels = ", self.signal_labels)
    
    def getSigbufs(self):
        return self.sigbufs
    
    def getSingalLabels(self):
        return self.signal_labels
    
    def getSampleFrequency(self):
        return self.sampleFrequency

        