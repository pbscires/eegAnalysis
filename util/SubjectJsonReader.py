'''

'''

import json

class SubjectJsonReader(object):
    '''
    Reads the subject summary information -- e.g. seizure start and end times
    '''


    def __init__(self, subjectsJsonFile):
        '''
        
        '''
        self.sunjectJsonFile = subjectsJsonFile
        f = open(subjectsJsonFile, 'r')
        self.jsonData = json.load(f)
        f.close()
    
    def getSeizureTimes(self, subjectName, sampleFileName):
        subjectInfoArr = self.jsonData[subjectName]
        seizureStartTimes = []
        seizureEndTimes = []
        for i in range(len(subjectInfoArr)):
            record_i = subjectInfoArr[i]
            if (sampleFileName == record_i['FileName']):
                seizureStartTimes = record_i['SeizureStartTime']
                seizureEndTimes = record_i['SeizureEndTime']
        return (seizureStartTimes, seizureEndTimes)
    
    def convertSecondsToEpochs(self, time_sec):
        slidingWindowLen = 2000
        return (int(time_sec * 1000 / slidingWindowLen))