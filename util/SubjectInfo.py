'''

'''
from util.ConfigJsonReader import ConfigJsonReader

class SubjectInfo(object):
    '''
    Contains information per subject
    '''


    def __init__(self, subjectName, configJsonReader):
        '''
        Constructor
        '''
        self.subjectName = subjectName
        self.subjectDir = configJsonReader.getSubjectDir(subjectName)