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
from multiprocessing import Pool

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
    global outDir
    global subjectFiles

    print ("cfgFilename = ", cfgFilename)
    jsonData = JsonReader(cfgFilename)
    subjectNames = jsonData.get_value("SubjectNames")
    configFile = jsonData.get_value("ConfigFile")
    seizuresFile = jsonData.get_value("SeizuresFile")
    features = jsonData.get_value("Features")
    outDir = jsonData.get_value("OutDir")
    
    if (not os.path.isdir(outDir)):
        print ("Given Path {} is not a directory".format(outDir))
        exit (-1)
    
    print ("subjectNames = ", subjectNames)
    print ("configFile = ", configFile)
    print ("seizuresFile = ", seizuresFile)
    print ("features = ", features)
    print ("OutDir = ", outDir)
    
    subjectFiles = jsonData.get_value("Filenames")
    print ("Subject Files = ", subjectFiles)
    return

def extractFeature(featureName, eegRecord):
    if (featureName == "LineLength"):
        ll = LineLength(epochLength=10000, slidingWindowLen=2000)
        ll.extractFeature(eegRecord.sigbufs, eegRecord.signal_labels, eegRecord.sampleFrequency)

def getEEGRecordFiles(subjectName, subjectFiles):
    global cfgReader
    
    eegRecordFiles = []
    subjectDir = cfgReader.getSubjectDir(subjectName)
    print ("subjectDir = ", subjectDir)
    
    if (subjectName not in subjectFiles):
        return eegRecordFiles

    regExpsForSubject = subjectFiles[subjectName]
    print ("RegExpsFrSubject = ", regExpsForSubject)

    allFiles = os.listdir(subjectDir)
    for file in allFiles:
        for regexp in regExpsForSubject:
#             print ("regexp = {}, file = {}".format(regexp, file))
#             pattern = re.compile(regexp)
            if (re.match(regexp, file) != None):
                eegRecordFiles.append(os.path.join(subjectDir, file))
    eegRecordFiles = sorted(eegRecordFiles)
    print ("len(eegRecordFiles) = ", len(eegRecordFiles))
    print ("eegRecordFiles = ", eegRecordFiles)
    return eegRecordFiles

def getOutFilePath(outDir, filePath, featurename):
    # Remove the .edf from filePath and add it to outDir
#     filename = os.path.basename(filePath)
#     (recordFile, extension) = os.path.splitext(filename)
    filename = '.'.join([featurename, os.path.basename(filePath), 'csv'])
    outFilePath = os.path.join(outDir, filename)
    return (outFilePath)

def generateInParallel(filePath):
    global outDir

    # Check if the output .csv file already exists.
    # If if the file has already been processed no need to do it again.
    outFilePath = getOutFilePath(outDir, filePath, "LineLength")
    if (os.path.exists(outFilePath)):
        print ("The file ", filePath, " is already processed")
        return

    # Extract the subject name from the record filePath
    basename = os.path.basename(filePath)
    m = re.search('(chb\d+)_\d+.edf$', basename)
    if (m == None):
        print ("Error. Could not retrieve the subject name from ", basename)
        return
    subject = m.group(1)
    print ("working on subject ", subject, ", record = ", filePath)
    
    eegRecord = EEGRecord(filePath, subject)
    eegRecord.loadFile()
    sigbufs = eegRecord.getSigbufs()
    signal_labels = eegRecord.getSingalLabels()
    sampleFrequency = eegRecord.getSampleFrequency()
    llFeature = LineLength()
    llFeature.extractFeature(sigbufs, signal_labels, sampleFrequency)
#     print ("outFilePath = ", outFilePath)
    llFeature.saveLLdfWithSeizureInfo(outFilePath, seizuresInfo, os.path.basename(filePath))

if __name__ == '__main__':
    global cfgReader
    global subjectNames
    global configFile
    global seizuresFile
    global outDir
    global features
    global subjectFiles

    print ("pgmName = ", pgmName)
    readArguments()
#     exit(0)
    cfgReader = ConfigJsonReader(configFile)
    
    seizuresInfo = Seizures(cfgReader)
    seizuresInfo.loadSeizuresFile(seizuresFile)
    
#     exit(0)
    timer1 = ElapsedTime()
    timer2 = ElapsedTime()
    # Initialize the process pool once at the beginning so that 
    # we do not get "Too many open file handles" error
    #  Note: 
#     if ("LineLength" in features):
#         p = Pool()
#         print ("Number of processes = ", p._processes)

    for subject in subjectNames:
        timer1.reset()
        eegRecordFiles = getEEGRecordFiles(subject, subjectFiles)
        # Line Length Feature extraction is parallelized at a "coarse" level
        #  i.e.  each EEG record is processed by one single thread.
        # FFT feature extraction is parallelized at "fine" level.
        #  i.e. each EEG record is processed by a pool of worker processes.
        if ("LineLength" in features):
            p = Pool()
            print ("Number of processes = ", p._processes)
            p.map(generateInParallel, eegRecordFiles)
            p.close()
            p.join()
        if ("FFT" in features):
            for filePath in eegRecordFiles:
                timer2.reset()
                print ("working on subject ", subject, ", record = ", filePath)
                eegRecord = EEGRecord(filePath, subject)
                eegRecord.loadFile()
                sigbufs = eegRecord.getSigbufs()
                signal_labels = eegRecord.getSingalLabels()
                sampleFrequency = eegRecord.getSampleFrequency()
                fftObj = FFT()
                fftObj.extractFeatureMultiProcessing(sigbufs, signal_labels, sampleFrequency)
#                 fftObj.extractFeature(sigbufs, signal_labels, sampleFrequency)
                outFilePath = getOutFilePath(outDir, filePath, "FFT")
                print ("outFilePath = ", outFilePath)
                fftObj.saveFFTWithSeizureInfo(outFilePath, seizuresInfo, os.path.basename(filePath))
                print ("Time taken for one file = ", timer2.timeDiff())
        print ("Time took for subject {} is {}".format(subject, timer1.timeDiff()))