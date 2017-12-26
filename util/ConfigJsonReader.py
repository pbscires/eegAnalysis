'''

'''
import json
import os

class ConfigJsonReader(object):
    '''
    Contains methods to read the *config.json files
    '''


    def __init__(self, configFile):
        '''
        Read the given config file into a json dictionary
        '''
        self.configFile = configFile
        f = open(configFile, 'r')
        self.jsonData = json.load(f)
        f.close()
    
    def getRootDir(self):
        return self.jsonData['RootDirectory']
    
    def getSubjectDir(self, subjectName):
        subDir = self.jsonData['SubjectDirectories'][subjectName]
        subjectDir = os.path.join(self.getRootDir(), subDir)
        return subjectDir
    
    def getSubjectsList(self):
        subjectNames = self.jsonData['SubjectDirectories'].keys()
        return subjectNames
