'''
'''
from util.JsonReader import JsonReader
import sys
import os
import re
import datetime
import time
import numpy as np
import shutil


pgmName = sys.argv[0]
if (len(sys.argv) > 1):
    cfgFilename = sys.argv[1]
else:
    cfgFilename = pgmName.replace(".py", ".json")

def readArguments():
    global subjectNames
    global configFile
    global subjectFiles
    global csvInDir
    global outDir
    global summaryDir
    global numFeatures
    global preictalSeconds, epochLen, slidingWindowLen

    print ("cfgFilename = ", cfgFilename)
    jsonData = JsonReader(cfgFilename)
    subjectNames = jsonData.get_value("SubjectNames")
    subjectFiles = jsonData.get_value("Filenames")
    csvInDir = jsonData.get_value("CSVInDir")
    summaryDir = jsonData.get_value("SummaryDir")
    outDir = jsonData.get_value("OutDir")
    numFeatures = int(jsonData.get_value("numFeatures"))
    preictalSeconds_str = jsonData.get_value("PreIctalOffsets")
    preictalSeconds = [int(x) for x in preictalSeconds_str] # convert strings to integers
    epochLen = int(jsonData.get_value("epochLenth"))
    slidingWindowLen = int(jsonData.get_value("slidingWindowLength"))

    print ("subjectNames = ", subjectNames)
    print ("Subject Files = ", subjectFiles)
    print ("csvInDir = ", csvInDir)
    print ("OutDir = ", outDir)
    print ("SummaryDir = ", summaryDir)
    print ("numFeatures = ", numFeatures)
    print ('preictalSeconds = {}, epochLen = {}, slidingWindowLen = {}'.format(preictalSeconds, epochLen, slidingWindowLen))

    return

def readSummaryFile(filePath=''):
    global startTimes, endTimes

    fd1 = open(filePath, 'r')
    re1 = re.compile('File Name: (chb\d+)_(\d+)\.edf')
    startRe = re.compile('File Start Time: (\d+-\d+-\d+ \d+:\d+:\d+)')
    endRe = re.compile('File End Time: (\d+-\d+-\d+ \d+:\d+:\d+)')
    startTimes = dict()
    endTimes = dict()
    summaryStarted = False
    for line in fd1.readlines():
#         print ("line = ", line)
        line = line.strip()
        if (len(line) <= 0):
#             print ("\n\n    Starting a new record")
            summaryStarted = False

        m1 = re1.search(line)
        if (m1 != None):
            summaryStarted = True
            subjectName = m1.group(1)
            recordNum = m1.group(2)
#             print ("subjectName = ", subjectName, ", recordNum = ", recordNum)
            continue

        m2 = startRe.search(line)
        if (m2 != None and summaryStarted):
            # Check if the hour filed is 24 or above
            timeStampStr = m2.group(1)
            startTime = datetime.datetime.strptime(timeStampStr, '%Y-%m-%d %H:%M:%S')
            startTimes[recordNum] = startTime
#             print ("startTime = ", startTime)
            continue

        m3 = endRe.search(line)
        if (m3 != None and summaryStarted):
            # Check if the hour filed is 24 or above
            timeStampStr = m3.group(1)
            endTime = datetime.datetime.strptime(timeStampStr, '%Y-%m-%d %H:%M:%S')
            endTimes[recordNum] = endTime
#             print ("endTime = ", endTime)
            continue

    fd1.close()
    print ("filePath = ", filePath)
    prev_endTime = datetime.datetime.now()
    for i in startTimes.keys():
        gapTime = startTimes[i] - prev_endTime
        print ('gapTime={}, startTimes[{}]={}, endTimes[{}]={}'.format(gapTime, i, startTimes[i], i, endTimes[i]))
        prev_endTime = endTimes[i]

def determinePerSubjectFiles():
    global subjectNames
    global subjectFiles
    global csvInDir
    global filesPerSubject
    global numFeatures

    filesPerSubject = {}
    for subjectName in subjectNames:
        filesPerSubject[subjectName] = []
        regExpsForSubject = subjectFiles[subjectName]
        print ("RegExpsForSubject = ", regExpsForSubject)
        allFiles = os.listdir(csvInDir)
        for filePath in allFiles:
            for regexp in regExpsForSubject:
    #             print ("regexp = {}, file = {}".format(regexp, filePath))
    #             pattern = re.compile(regexp)
                if (re.match(regexp, filePath) != None):
#                     print ("filePath ", filePath, " is matched by regexp ", regexp)
                    filesPerSubject[subjectName].append(filePath)
        filesPerSubject[subjectName] = sorted(filesPerSubject[subjectName])
#         print(filesPerSubject[subjectName])
                    
    return

def verifyCSVFiles():
    global subjectNames
    global configFile
    global subjectFiles
    global csvInDir
    global outDir
    global filesPerSubject
    global numFeatures
    
    for subjectName in filesPerSubject.keys():
        arr_numRows = []
        arr_numCols = []
        filesToBeRemoved = []
        for filename in filesPerSubject[subjectName]:
            filePath = os.path.join(csvInDir, filename)
            arr = np.genfromtxt(filePath, delimiter=',')
            if (arr.shape[0] not in arr_numRows):
                arr_numRows.append(arr.shape[0])
            if (arr.shape[1] not in arr_numCols):
                arr_numCols.append(arr.shape[1])
            if (arr.shape[1] != (numFeatures+1)):
                print ("subject ", subjectName, "file ", filename, "has non-standard number of features: ", arr.shape[1])
                filesToBeRemoved.append(filename)
        print ("subjectName = ", subjectName, "arr_numRows = ", arr_numRows, "arr_numCols = ", arr_numCols)
        for filename in filesToBeRemoved:
            print ("Removing the filename", filename)
            filesPerSubject[subjectName].remove(filename)
        
#         print ("filesPerSubject[", subjectName,"] = ", filesPerSubject[subjectName])

    return

def concatPerSubjectFiles(subjectName):
    global subjectNames
    global configFile
    global subjectFiles
    global csvInDir
    global outDir
    global filesPerSubject
    global numFeatures
    
    concatFilePath = os.path.join(outDir, '.'.join([subjectName, 'csv']))
    # if the file exists, just return; no need to redo the work
    if (os.path.exists(concatFilePath)):
        return

    fd1 = open(concatFilePath, 'w')
    for filename in filesPerSubject[subjectName]:
        filePath = os.path.join(csvInDir, filename)
        with open(filePath, 'r') as rfd1:
            shutil.copyfileobj(rfd1, fd1)
    fd1.close()

def shiftResultsForPreIctal(subjectName, preIctalSeconds,  slidingWindowLen):
    '''
    Shift the last column (the classification) up by the given number of rows
    '''
    global outDir

    print ("shiftResultsForPreIctal(): subjectName = ", subjectName,
           "preIctalSeconds = ", preIctalSeconds, "slidingWindowLen = ", slidingWindowLen)
    concatFilePathOld = os.path.join(outDir, '.'.join([subjectName, 'csv']))
    arr = np.genfromtxt(concatFilePathOld, delimiter=',')
    X = arr[:,:-1]
    y = arr[:,-1]
    
    for preIctalOffset in preIctalSeconds:
        concatFilePathNew = os.path.join(outDir, '.'.join([subjectName, 'preictal', str(preIctalOffset), 'csv']))
        # If the work was already done before, just return
        if (os.path.exists(concatFilePathNew)):
            return
        # Roll the y array to shift the column by given number of rows
        preIctalOffset = int(preIctalOffset / slidingWindowLen)
        y_new = np.roll(y, arr.shape[0] - preIctalOffset)
        new_arr = np.concatenate((X, np.reshape(y_new, (-1, 1))), axis=1)
        print ('Shape of the arrays: X={}, y_new={}, arr={}, new_arr={}'.format(X.shape, y_new.shape, arr.shape, new_arr.shape))
        fmtStr = '%f,' * numFeatures
        fmtStr = ''.join([fmtStr, '%d'])
        np.savetxt(concatFilePathNew, new_arr, fmt=fmtStr, delimiter=',')



if __name__ == '__main__':
    global subjectNames
    global configFile
    global subjectFiles
    global csvInDir
    global outDir
    global summaryDir
    global numFeatures
    global preictalSeconds, epochLen, slidingWindowLen
    global startTimes, endTimes
    
    print ("pgmName = ", pgmName)
    readArguments()

    filesPerSubject = {}
    print ("Determining the files per each subject...")
    determinePerSubjectFiles()
    print ("...done")
    # 1. Make sure all the csv files have same number of columns
    print ("Verifying the CSV files...")
    verifyCSVFiles()
    print ("...done")
    for subject in subjectNames:
        print ("Concatenating the per-subject files...")
        concatPerSubjectFiles(subject)
        print ("...done")
        print ("Shifting the results for Pre-ictal analysis...")
        shiftResultsForPreIctal(subject, preictalSeconds, slidingWindowLen)
        print ("...done")
