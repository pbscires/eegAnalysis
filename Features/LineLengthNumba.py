'''
Created on Dec 25, 2017

'''
import numpy as np
import pandas as pd
from util.ElapsedTime import ElapsedTime
import matplotlib.pyplot as plt
from numba import vectorize, float64, guvectorize

@guvectorize(["void(float64[:,:], int64[:], int64, float64[:,:])"],
             "(m,n),(p),()->(p,n)", target='cpu')
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
#     return llMat



class LineLengthNumba(object):
    '''
    Contains methods to extract LineLength feature
    '''


    def __init__(self, epochLength=10000, slidingWindowLen=2000):
        '''
        Constructor
        '''
        # Sliding window length should be < epocLength
        print ("epochLength = ", epochLength, "slidingWindowLen = ", slidingWindowLen)
        if (slidingWindowLen > epochLength):
            print ("Invalid values for sliding window length and/or epoch length")
            exit(-1)
        self.epochLength = epochLength
        self.slidingWindowLen = slidingWindowLen
    
    def extractFeature(self, sigbufs, signal_labels, sampleFrequency):
        '''
        Extract the Line length feature from the given input file to the given output file.
        Input file is expected to be in the EDF file format.
        Output file is CSV file with 2 values per row -- epoch number and LineLength value
        '''
        self.signal_labels = signal_labels
        numSamples = sigbufs.shape[0]
        numSamplesPerEpoch = int(sampleFrequency * self.epochLength / 1000)
        print ("numSamples = ", numSamples, ", sampleFrequency = ", sampleFrequency)
        sigDiffs = np.delete(sigbufs, numSamples-1, 0) - np.delete(sigbufs, 0, 0)
        sigDiffs = np.absolute(sigDiffs)
        print ("Shape of sigDiffs = ", sigDiffs.shape)

        startingRowsArr = np.arange(0, numSamples, int(self.slidingWindowLen*sampleFrequency/1000))
        
        # Identify the value of the last element in the startingRowsArr array.
        #  The requirement is that there should be enough samples left to contain
        #   an epoch.
        j = len(startingRowsArr)
        j -= 1
        print ("j=", j, "startingRowsArr[", j, "] = ", startingRowsArr[j])
        lastEpochIndex = numSamples - numSamplesPerEpoch
        print ("lastEpochIndex = ", lastEpochIndex)
        while (startingRowsArr[j] > lastEpochIndex):
            j -= 1
            print ("j=", j, "startingRowsArr[", j, "] = ", startingRowsArr[j])
        startingRowsArr = np.delete(startingRowsArr, range(j,len(startingRowsArr)))
        print ("Shape of startingRowsArr = ", startingRowsArr.shape)

        timer3 = ElapsedTime()
        timer3.reset()
#         timer2 = ElapsedTime()
#         timer2.reset()
#         llMat = sigDiffs[startingRowsArr,]
#         for j in range(1, numSamplesPerEpoch):
#             startingRowsArr = startingRowsArr + 1
#             llMat = llMat + sigDiffs[startingRowsArr,]
#             if (j % 256 == 0):
#                 print("j=", j, ", timer2 elapsed time=", timer2.timeDiff())
#         llMat = llMat / numSamplesPerEpoch
#         self.llDf = pd.DataFrame(data = llMat, columns = signal_labels)

        llMat = calcLLdf(sigDiffs, startingRowsArr, numSamplesPerEpoch)
        self.llDf = pd.DataFrame(data = llMat, columns = self.signal_labels)
        print (self.llDf.head())
        print (self.llDf.shape)
    
    def plotLLdf(self, channels=None):
        plt.figure()
        if (channels != None):
            plt.plot(self.llDf[channels])
        plt.show()

    def saveLLdf(self, outFilePath):
        '''
        Save the Line Length feature in the given file in CSV format
        '''
        numEpochs = self.llDf.shape[0]
        numChannels = self.llDf.shape[1]
        columnsDone = dict()
        print ("numEpochs = ", numEpochs, ", numChannels = ", numChannels)
        for columnName in self.llDf.columns.values:
#         for columnName in ['T8-P8']:
            if (columnName == 'T8-P8'):
                continue
            if (columnName not in columnsDone):
                columnsDone[columnName] = True
            else:
                continue
            fileToWrite = ''.join([outFilePath, '.', columnName, '.csv'])
            f = open(fileToWrite, "w")
            for i in range(numEpochs):
                strToWrite = ','.join([str(i), str(self.llDf.loc[i, columnName])])
                f.write(strToWrite)
                f.write("\n")
            f.close()
    
    def saveLLdfWithSeizureInfo(self, outFilePath, seizures, recordFile):
        '''
        Save the Line Length feature in the given file in CSV format;
        Include the seizures information as +1 (True) or -1 (False)
        '''
        numEpochs = self.llDf.shape[0]
        numChannels = self.llDf.shape[1]
        print ("numEpochs = ", numEpochs, ", numChannels = ", numChannels)
        fileToWrite = '.'.join([outFilePath, recordFile, 'csv'])
        f = open(fileToWrite, "w")
        for i in range(numEpochs):
            if seizures.isSeizurePresent(recordFile, i, self.epochLength, self.slidingWindowLen):
                seizureValue = 1
            else:
                seizureValue = -1 
            strToWrite = ''
            columnsDone = dict()
            for columnName in self.llDf.columns.values:
                if (columnName == 'T8-P8'):
                    continue
                if (columnName not in columnsDone):
                    columnsDone[columnName] = True
                else:
                    continue
                if (strToWrite == ''):
                    strToWrite = str(self.llDf.loc[i, columnName])
                else:
                    strToWrite = ','.join([strToWrite, str(self.llDf.loc[i, columnName])])

            strToWrite = ','.join([str(i), strToWrite, str(seizureValue)])
            f.write(strToWrite)
            f.write("\n")
        f.close()

#         for columnName in self.llDf.columns.values:
#             if (columnName == 'T8-P8'):
#                 continue
#             if (columnName not in columnsDone):
#                 columnsDone[columnName] = True
#             else:
#                 continue
#             fileToWrite = ''.join([outFilePath, '.', recordFile, '.', columnName, '.csv'])
#             f = open(fileToWrite, "w")
#             for i in range(numEpochs):
#                 if seizures.isSeizurePresent(recordFile, i, self.epochLength, self.slidingWindowLen):
#                     seizureValue = 1
#                 else:
#                     seizureValue = -1 
#                 strToWrite = ','.join([str(i), str(self.llDf.loc[i, columnName]), str(seizureValue)])
#                 f.write(strToWrite)
#                 f.write("\n")
#             f.close()
