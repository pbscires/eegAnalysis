'''

'''

import json
from datetime import datetime, timedelta
import re

class SeizureInfo():
    def __init__(self, filePath, fileStartTime, fileEndTime, seizureStartTimes, seizureEndTimes):
        self.filePath = filePath
        self.fileStartTime = fileStartTime
        self.fileEndTime = fileEndTime
        self.seizureStartTimes = seizureStartTimes
        self.seizureEndTimes = seizureEndTimes

    def __str__(self):
        return ('\n'.join([self.filePath, str(self.fileStartTime), str(self.fileEndTime),
                           str(self.seizureStartTimes), str(self.seizureEndTimes)]))

class Seizures(object):
    '''
    Represents the seizures information
    '''
    def __init__(self, cfgReader):
        '''
        Constructor
        '''
        self.seizureFiles = {}
        self.cfgReader = cfgReader
        self.seizures = []
        # We are assuming that all the experiments are done on 2010-01-01
        # It does not really matter; this is needed for python datetime representation
        self.expDateStr = "2010-01-01"

    def isSeizurePresent(self, recordFile, epochNum, epochLen, slidingWindowLen):
        '''
        epochLen and slidingWindowLen are in milliseconds
        '''
        # Convert epochNum to start and end datetime objects
#         print ("recordFile = ", recordFile, ", epochNum =", epochNum, ", epochLen = ", epochLen,
#                ", slidingWindowLen = ", slidingWindowLen)
        epochStartSeconds = int(epochNum * slidingWindowLen / 1000)
        epochEndSeconds = epochStartSeconds + int(epochLen / 1000)
        seizureInfoObj = self.getSeizureInfoObj(recordFile)
#         print ("seizureInfoObj = ", seizureInfoObj)
        if (seizureInfoObj != None):
            epochStart = self.getDateTimeObjFromSeconds(seizureInfoObj.fileStartTime, epochStartSeconds)
            epochEnd = self.getDateTimeObjFromSeconds(seizureInfoObj.fileStartTime, epochEndSeconds)
            for i in range(len(seizureInfoObj.seizureStartTimes)):
#                 if ((recordFile == "chb01_03.edf") and (epochNum > 1495) and (epochNum < 1520)):
#                     print("epochStart = ", epochStart, ", epochEnd = ", epochEnd)
#                     print ("seizureStart = ", seizureInfoObj.seizureStartTimes[i],
#                            "seizureEnd = ", seizureInfoObj.seizureEndTimes[i])
                if (( (epochStart >= seizureInfoObj.seizureStartTimes[i]) and
                      (epochStart <= seizureInfoObj.seizureEndTimes[i])) or
                     ( (epochEnd >= seizureInfoObj.seizureStartTimes[i]) and
                      (epochEnd <= seizureInfoObj.seizureEndTimes[i]))):
                    return True
        return False

    def getSeizureInfoObj(self, recordFile):
        for seizureInfoObj in self.seizures:
#             print ("filePath = ", seizureInfoObj.filePath, ", recordFile = ", recordFile)
            if (seizureInfoObj.filePath == recordFile):
                return (seizureInfoObj)
        return None
            

    def getDateTimeObj(self, timeStr):
        dateStr = "2010-01-01"
        m = re.match(r'(\d+):(\d+):(\d+)', timeStr)
        if (m != None):
            (hh, mm, ss) = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
#             print ("hh mm ss = ", hh, mm, ss)
            if (hh >= 24):
                hh -= 24
                dateStr = "2010-01-02"
#             print ("hh mm ss = ", hh, mm, ss)
            
            dateTimeStr = dateStr + " " + ':'.join([str(hh), str(mm), str(ss)])
            datetimeObj = datetime.strptime(dateTimeStr, "%Y-%m-%d %H:%M:%S")
            return datetimeObj
        else:
            return None
    
    def getDateTimeObjFromSeconds(self, startTime, n_seconds):
        dateTimeObj = startTime + timedelta(seconds=n_seconds)
        return dateTimeObj    

    def loadSeizuresFile(self, filePath):
        f = open(filePath, 'r')
        self.jsonRoot = json.load(f)
        f.close()
        subjectsList = self.jsonRoot.keys()
#         print ("Subjects list = ", subjectsList)
        for subject in subjectsList:
            seizuresJsonArr = self.jsonRoot[subject]
            for seizureElem in seizuresJsonArr:
                fileName = seizureElem['FileName']
                fileStartTime = self.getDateTimeObj(seizureElem['StartTime'])
                fileEndTime = self.getDateTimeObj(seizureElem['EndTime'])
                seizureStartTimes = []
                seizureEndTimes = []
                for timeInSeconds in seizureElem['SeizureStartTimes']:
                    datetimeObj = self.getDateTimeObjFromSeconds(fileStartTime, int(timeInSeconds))
                    seizureStartTimes.append(datetimeObj)
                for timeInSeconds in seizureElem['SeizureEndTimes']:
                    datetimeObj = self.getDateTimeObjFromSeconds(fileStartTime, int(timeInSeconds))
                    seizureEndTimes.append(datetimeObj)

                seizureInfoElem = SeizureInfo(fileName, fileStartTime, fileEndTime, seizureStartTimes, seizureEndTimes)
#                 print (seizureInfoElem)
                self.seizures.append(seizureInfoElem)
            
            
        
        