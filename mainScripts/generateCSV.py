'''
Generate the CSV file for the given classifier or regressor
'''
from util.JsonReader import JsonReader
from util.ConfigJsonReader import ConfigJsonReader
import sys
import os
import re
from Features.LineLength import LineLength
from DataRepresentation.EEGRecord import EEGRecord

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

    print ("cfgFilename = ", cfgFilename)
    jsonData = JsonReader(cfgFilename)
    subjectNames = jsonData.get_value("SubjectNames")
    configFile = jsonData.get_value("ConfigFile")
    seizuresFile = jsonData.get_value("SeizuresFile")
    features = jsonData.get_value("Features")
    
    print ("subjectNames = ", subjectNames)
    print ("configFile = ", configFile)
    print ("seizuresFile = ", seizuresFile)
    print ("featues = ", features)
    return

def extractFeature(featureName, eegRecord):
    if (featureName == "LineLength"):
        ll = LineLength(epochLength=10000, slidingWindowLen=2000)
        ll.extractFeature(eegRecord.sigbufs, eegRecord.signal_labels, eegRecord.sampleFrequency)

def getEEGRecordFiles(subjectName):
    global cfgReader
    
    eegRecordFiles = []
    cfgReader = ConfigJsonReader(configFile)
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

    print ("pgmName = ", pgmName)
    readArguments()

    for subject in subjectNames:
        eegRecordFiles = getEEGRecordFiles(subject)
        for filePath in eegRecordFiles:
            eegRecord = EEGRecord(filePath, subject)
            eegRecord.loadFile()
            sigbufs = eegRecord.getSigbufs()
            signal_labels = eegRecord.getSingalLabels()
            sampleFrequency = eegRecord.getSampleFrequency()
            llFeature = LineLength()
            llFeature.extractFeature(sigbufs, signal_labels, sampleFrequency)