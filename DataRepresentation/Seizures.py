'''

'''

import json
from datetime import datetime

class SeizureInfo():
    def __init__(self):
        self.filePath = ''
        self.fileStartTime = datetime()
        self.fileEndTime = datetime()
        self.seizureStartTimes = []
        self.seizureEndTimes = []

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
    
    def loadSeizuresFile(self, filePath):
        f = open(filePath, 'r')
        self.jsonRoot = json.load(f)
        f.close()
        subjectsList = self.jsonRoot.keys()
        for subject in subjectsList:
            seizuresJsonArr = self.jsonRoot[subject]
            for seizureElem in seizuresJsonArr:
                fileName = seizureElem['FileName']
                fileStartTime = seizureElem['StartTime']
                fileEndTime = seizureElem['EndTime']
                seizureStartTimes = seizureElem['SeizureStartTimes']
                seizureEndTimes = seizureElem['SeizureEndTimes']
            
        
        