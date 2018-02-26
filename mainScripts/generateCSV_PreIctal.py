'''
'''
from util.JsonReader import JsonReader
import sys
import os
import re
import datetime
import time

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

    print ("cfgFilename = ", cfgFilename)
    jsonData = JsonReader(cfgFilename)
    subjectNames = jsonData.get_value("SubjectNames")
    subjectFiles = jsonData.get_value("Filenames")
    csvInDir = jsonData.get_value("CSVInDir")
    summaryDir = jsonData.get_value("SummaryDir")
    outDir = jsonData.get_value("OutDir")
    numFeatures = int(jsonData.get_value("numFeatures"))

    print ("subjectNames = ", subjectNames)
    print ("Subject Files = ", subjectFiles)
    print ("csvInDir = ", csvInDir)
    print ("OutDir = ", outDir)
    print ("SummaryDir = ", summaryDir)
    print ("numFeatures = ", numFeatures)

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


def readSummaryFile_old(filePath=''):
    fd1 = open(filePath, 'r')
    re1 = re.compile('File Name: (chb\d+)_(\d+)\.edf')
    startRe = re.compile('File Start Time: (\d+:\d+:\d+)')
    endRe = re.compile('File End Time: (\d+:\d+:\d+)')
    startTimes = dict()
    endTimes = dict()
    summaryStarted = False
    num23Starts = 0
    num23Ends = 0
    days = ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04']
    next_day_index = 0
    for line in fd1.readlines():
#         print ("line = ", line)
        line = line.strip()
        if (len(line) <= 0):
            print ("\n\n    Starting a new record")
            summaryStarted = False

        m1 = re1.search(line)
        if (m1 != None):
            summaryStarted = True
            subjectName = m1.group(1)
            recordNum = m1.group(2)
            day_index = next_day_index
#             print ("subjectName = ", subjectName, ", recordNum = ", recordNum)
            continue

        m2 = startRe.search(line)
        if (m2 != None and summaryStarted):
            # Check if the hour filed is 24 or above
            timeStampStr = m2.group(1)
            m2_hh = re.search('(\d+):(\d+):(\d+)', timeStampStr)
            if (int(m2_hh.group(1)) == 23):
                num23Starts += 1
                next_day_index += 1
            if (int(m2_hh.group(1)) >= 24):
                new_hh = int(m2_hh.group(1)) - 24
                new_timeStampStr = timeStampStr.replace(m2_hh.group(1), str(new_hh))
                new_timeStampStr = ' '.join([days[day_index], new_timeStampStr])
                startTime = datetime.datetime.strptime(new_timeStampStr, '%Y-%m-%d %H:%M:%S')
                # TODO:  Add 1 to the datetime
            else:
                timeStampStr = ' '.join([days[day_index], timeStampStr])
                startTime = datetime.datetime.strptime(timeStampStr, '%Y-%m-%d %H:%M:%S')
            startTimes[recordNum] = startTime
#             print ("startTime = ", startTime)
            continue
        
        m3 = endRe.search(line)
        if (m3 != None and summaryStarted):
            # Check if the hour filed is 24 or above
            timeStampStr = m3.group(1)
            m3_hh = re.search('(\d+):(\d+):(\d+)', timeStampStr)
            if (int(m3_hh.group(1)) == 23):
                num23Ends += 1
            if (int(m3_hh.group(1)) >= 24):
                new_hh = int(m3_hh.group(1)) - 24
                new_timeStampStr = timeStampStr.replace(m3_hh.group(1), str(new_hh))
                new_timeStampStr = ' '.join([days[day_index], new_timeStampStr])
                endTime = datetime.datetime.strptime(new_timeStampStr, '%Y-%m-%d %H:%M:%S')
                # TODO:  Add 1 to the datetime
            else:
                timeStampStr = ' '.join([days[day_index], timeStampStr])
                endTime = datetime.datetime.strptime(timeStampStr, '%Y-%m-%d %H:%M:%S')
            endTimes[recordNum] = endTime
#             print ("endTime = ", endTime)
            continue

    fd1.close()
    print ("filePath = ", filePath, ", num23Starts = ", num23Starts, ", num23Ends = ", num23Ends)
    for i in startTimes.keys():
        print ('startTimes[{}]={}, endTimes[{}]={}\n'.format(i, startTimes[i], i, endTimes[i]))

if __name__ == '__main__':
    global subjectNames
    global configFile
    global subjectFiles
    global csvInDir
    global outDir
    global summaryDir
    global numFeatures
    global startTimes, endTimes
    
    print ("pgmName = ", pgmName)
    readArguments()
    for subject in subjectNames:
        summaryFilename = ''.join([subject, '-summary.txt'])
#         filePath = os.path.join(summaryDir, subject, summaryFilename)
        filePath = os.path.join(summaryDir, summaryFilename)
        print ("Trying to read the summary from...", filePath)
        readSummaryFile(filePath)
