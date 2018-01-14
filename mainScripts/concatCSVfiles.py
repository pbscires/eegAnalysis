'''

'''
from util.JsonReader import JsonReader
import sys
import os
import re
import numpy as np
from sklearn.cross_validation import train_test_split


pgmName = sys.argv[0]
if (len(sys.argv) > 1):
    cfgFilename = sys.argv[1]
else:
    cfgFilename = pgmName.replace(".py", ".json")

def readArguments():
    global subjectNames
    global configFile
    global subjectFiles
    global inDir
    global outDir
    global trainPercent
    global numFeatures

    print ("cfgFilename = ", cfgFilename)
    jsonData = JsonReader(cfgFilename)
    subjectNames = jsonData.get_value("SubjectNames")
    subjectFiles = jsonData.get_value("Filenames")
    inDir = jsonData.get_value("InDir")
    outDir = jsonData.get_value("OutDir")
    trainPercent = float(jsonData.get_value("trainPercent"))
    numFeatures = int(jsonData.get_value("numFeatures"))

    print ("subjectNames = ", subjectNames)
    print ("Subject Files = ", subjectFiles)
    print ("InDir = ", inDir)
    print ("OutDir = ", outDir)
    print ("trainPercent = ", trainPercent)
    print ("numFeatures = ", numFeatures)

    return

def determinePerSubjectFiles():
    global subjectNames
    global subjectFiles
    global inDir
    global filesPerSubject
    global numFeatures

    filesPerSubject = {}
    for subjectName in subjectNames:
        filesPerSubject[subjectName] = []
        regExpsForSubject = subjectFiles[subjectName]
        print ("RegExpsForSubject = ", regExpsForSubject)
        allFiles = os.listdir(inDir)
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
    global inDir
    global outDir
    global trainPercent
    global filesPerSubject
    global numFeatures
    
    for subjectName in filesPerSubject.keys():
        arr_numRows = []
        arr_numCols = []
        filesToBeRemoved = []
        for filename in filesPerSubject[subjectName]:
            filePath = os.path.join(inDir, filename)
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

def trainTestSplit():
    global inDir
    global outDir
    global filesPerSubject
    for subjectName in filesPerSubject.keys():
        Xy_train_path = os.path.join(outDir, '.'.join([subjectName, 'Xy_train', 'csv']))
        Xy_test_path = os.path.join(outDir, '.'.join([subjectName, 'Xy_test', 'csv']))
        y_train_path = os.path.join(outDir, '.'.join([subjectName, 'y_train', 'csv']))
        y_test_path = os.path.join(outDir, '.'.join([subjectName, 'y_test', 'csv']))
        Xy_train_fHandle = open(Xy_train_path, 'w')
        Xy_test_fHandle = open(Xy_test_path, 'w')
        y_train_fHandle = open(y_train_path, 'w')
        y_test_fHandle = open(y_test_path, 'w')
        for filename in filesPerSubject[subjectName]:
            filePath = os.path.join(inDir, filename)
            arr = np.genfromtxt(filePath, delimiter=',')
            X = arr[:,:-1]
            y = arr[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            print ("subjectName = ", subjectName, "filename=  ", filename, "filePath = ", filePath)
            print ("X_train.shape = ", X_train.shape, "X_test.shape = ", X_test.shape,
                   "y_train.shape = ", y_train.shape, "y_test.shape = ", y_test.shape)
            
#             y_temp = np.reshape(y_train, (-1, 1))
#             print ("y_temp.shape = ", y_temp.shape)
            Xy_train_arr = np.concatenate((X_train, np.reshape(y_train, (-1, 1))), axis=1)
            Xy_test_arr = np.concatenate((X_test, np.reshape(y_test, (-1, 1))), axis=1)
            y_train_arr = np.concatenate((y_train, y_train), axis=0)
            y_test_arr = np.concatenate((y_test, y_test), axis=0)

            # Write to the files
            np.savetxt(Xy_train_fHandle, Xy_train_arr, delimiter=',')
            np.savetxt(Xy_test_fHandle, Xy_test_arr, delimiter=',')
            np.savetxt(y_train_fHandle, y_train_arr, delimiter=',')
            np.savetxt(y_test_fHandle, y_test_arr, delimiter=',')
#             for i in range(X_train.shape[0]):
#                 strToWrite = ','.join([','.join(str(X_train[i])), str(y_train[i])])
#                 Xy_train_fHandle.write(strToWrite)
#                 Xy_train_fHandle.write('\n')
#                 strToWrite = ','.join([X_test[i], y_test[i]])
#                 Xy_test_fHandle.write(strToWrite)
#                 Xy_test_fHandle.write('\n')
#                 strToWrite = ','.join([y_train[i], y_train[i]])
#                 y_train_fHandle.write(strToWrite)
#                 y_train_fHandle.write('\n')
#                 strToWrite = ','.join([y_test[i], y_test[i]])
#                 y_test_fHandle.write(strToWrite)
#                 y_test_fHandle.write('\n')
        Xy_train_fHandle.close()
        Xy_test_fHandle.close()
        y_train_fHandle.close()
        y_test_fHandle.close()

    return

if __name__ == '__main__':
    global subjectNames
    global configFile
    global subjectFiles
    global inDir
    global outDir
    global trainPercent
    global filesPerSubject
    global numFeatures

    print ("pgmName = ", pgmName)
    readArguments()

    filesPerSubject = {}
    determinePerSubjectFiles()
    # 1. Make sure all the csv files have same number of columns
    verifyCSVFiles()
    # 2. Create train-test rows
    trainTestSplit()