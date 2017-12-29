'''
Generate the CSV file for the given classifier or regressor
'''
from util.JsonReader import JsonReader
from util.ConfigJsonReader import ConfigJsonReader
import sys
import os
import re
from Features.LineLength import LineLength
from Features.FastFourierTransform import FFT
from DataRepresentation.EEGRecord import EEGRecord
from DataRepresentation.Seizures import Seizures
from util.ElapsedTime import ElapsedTime
pgmName = sys.argv[0]
if (len(sys.argv) > 1):
    cfgFilename = sys.argv[1]
else:
    cfgFilename = pgmName.replace(".py", ".json")

def readArguments():
    global jsonData
    global subjectNames
    global configFile
    global seizuresFile
    global features
    global outFilePath

    print ("cfgFilename = ", cfgFilename)
    jsonData = JsonReader(cfgFilename)
    subjectNames = jsonData.get_value("SubjectNames")
    configFile = jsonData.get_value("ConfigFile")
    seizuresFile = jsonData.get_value("SeizuresFile")
    features = jsonData.get_value("Features")
    outFilePath = jsonData.get_value("OutFilePath")
    
    print ("subjectNames = ", subjectNames)
    print ("configFile = ", configFile)
    print ("seizuresFile = ", seizuresFile)
    print ("featues = ", features)
    print ("OutFilePath = ", outFilePath)
    return

def extractFeature(featureName, eegRecord):
    if (featureName == "LineLength"):
        ll = LineLength(epochLength=10000, slidingWindowLen=2000)
        ll.extractFeature(eegRecord.sigbufs, eegRecord.signal_labels, eegRecord.sampleFrequency)

def getEEGRecordFiles(subjectName):
    global cfgReader
    
    eegRecordFiles = []
    subjectDir = cfgReader.getSubjectDir(subjectName)
    print ("subjectDir = ", subjectDir)

    allFiles = os.listdir(subjectDir)
    for file in allFiles:
        if (re.search('\.edf$', file) != None):
            eegRecordFiles.append(os.path.join(subjectDir, file))
    eegRecordFiles = sorted(eegRecordFiles)
    print ("len(eegRecordFiles) = ", len(eegRecordFiles))
    print ("eegRecordFiles = ", eegRecordFiles)
    return eegRecordFiles


if __name__ == '__main__':
    global cfgReader
    global subjectNames
    global configFile
    global seizuresFile
    global outFilePath
    global features


    print ("pgmName = ", pgmName)
    readArguments()
    cfgReader = ConfigJsonReader(configFile)
    
    seizuresInfo = Seizures(cfgReader)
    seizuresInfo.loadSeizuresFile(seizuresFile)
    
#     exit(0)
    timer1 = ElapsedTime()
    for subject in subjectNames:
        timer1.reset()
        eegRecordFiles = getEEGRecordFiles(subject)
        for filePath in eegRecordFiles:
#             if (re.search('chb01_03.edf$', filePath) == None):
#                 continue
            print ("working on subject ", subject, ", record = ", filePath)
            eegRecord = EEGRecord(filePath, subject)
            eegRecord.loadFile()
            sigbufs = eegRecord.getSigbufs()
            signal_labels = eegRecord.getSingalLabels()
            sampleFrequency = eegRecord.getSampleFrequency()
            if ("LineLength" in features):
                llFeature = LineLength()
                llFeature.extractFeature(sigbufs, signal_labels, sampleFrequency)
                llFeature.saveLLdfWithSeizureInfo(outFilePath, seizuresInfo, os.path.basename(filePath))
            if ("FFT" in features):
                fftObj = FFT()
                fftObj.extractFeatureMultiProcessing(sigbufs, signal_labels, sampleFrequency)
#                 fftObj.extractFeature(sigbufs, signal_labels, sampleFrequency)
                fftObj.saveFFTWithSeizureInfo(outFilePath, seizuresInfo, os.path.basename(filePath))
        print ("Time took for subject %s is %f" % subject, timer1.timeDiff())