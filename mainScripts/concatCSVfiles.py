'''

'''
from util.JsonReader import JsonReader
import sys
import os
import re
import numpy as np
import subprocess
from sklearn.cross_validation import train_test_split
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
    global numFeatures

    for subjectName in filesPerSubject.keys():
        Xy_train_path = os.path.join(outDir, '.'.join([subjectName, 'Xy_train', 'csv']))
        Xy_test_path = os.path.join(outDir, '.'.join([subjectName, 'Xy_test', 'csv']))
        y_train_path = os.path.join(outDir, '.'.join([subjectName, 'y_train', 'csv']))
        y_test_path = os.path.join(outDir, '.'.join([subjectName, 'y_test', 'csv']))
        train_numRows = 0
        test_numRows = 0
        for filename in filesPerSubject[subjectName]:
            filePath = os.path.join(inDir, filename)
            arr = np.genfromtxt(filePath, delimiter=',')
            X = arr[:,:-1]
            y = arr[:,-1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            print ("subjectName = ", subjectName, "filename=  ", filename, "filePath = ", filePath)
#             print ("X_train.shape = ", X_train.shape, "X_test.shape = ", X_test.shape,
#                    "y_train.shape = ", y_train.shape, "y_test.shape = ", y_test.shape)
            
            train_numRows += X_train.shape[0]
            test_numRows += X_test.shape[0]
            
            Xy_train_arr = np.concatenate((X_train, np.reshape(y_train, (-1, 1))), axis=1)
            Xy_test_arr = np.concatenate((X_test, np.reshape(y_test, (-1, 1))), axis=1)
            y_train_arr = np.concatenate((y_train, y_train), axis=0)
            y_test_arr = np.concatenate((y_test, y_test), axis=0)

            fmtStr = '%f,' * numFeatures
            fmtStr = ''.join([fmtStr, '%d'])
#             print ("fmtStr = ", fmtStr)
            # Write to the files
            cur_Xy_train_path = os.path.join(outDir, '.'.join([filename, 'Xy_train', 'csv']))
            cur_Xy_test_path = os.path.join(outDir, '.'.join([filename, 'Xy_test', 'csv']))
            cur_y_train_path = os.path.join(outDir, '.'.join([filename, 'y_train', 'csv']))
            cur_y_test_path = os.path.join(outDir, '.'.join([filename, 'y_test', 'csv']))
            np.savetxt(cur_Xy_train_path, Xy_train_arr, fmt=fmtStr, delimiter=',')
            np.savetxt(cur_Xy_test_path, Xy_test_arr, fmt=fmtStr, delimiter=',')
            np.savetxt(cur_y_train_path, y_train_arr, fmt='%d', delimiter=',')
            np.savetxt(cur_y_test_path, y_test_arr, fmt='%d', delimiter=',')
            
        print('train_numRows = ', train_numRows, ', test_numRows = ', test_numRows)
        strToWrite = ','.join([str(train_numRows), str(numFeatures), 'no_seizure', 'seizure'])
        fd1 = open(Xy_train_path, 'w')
        fd1.write(strToWrite)
        fd1.write('\n')
#             fd1.close()
        strToWrite = ','.join([str(test_numRows), str(numFeatures), 'no_seizure', 'seizure'])
        fd2 = open(Xy_test_path, 'w')
        fd2.write(strToWrite)
        fd2.write('\n')
#             fd2.close()
        strToWrite = ','.join([str(train_numRows), '1', 'no_seizure', 'seizure'])
        fd3 = open(y_train_path, 'w')
        fd3.write(strToWrite)
        fd3.write('\n')
#             fd3.close()
        strToWrite = ','.join([str(test_numRows), '1', 'no_seizure', 'seizure'])
        fd4 = open(y_test_path, 'w')
        fd4.write(strToWrite)
        fd4.write('\n')
#             fd4.close()
        for filename in filesPerSubject[subjectName]:
            cur_Xy_train_path = os.path.join(outDir, '.'.join([filename, 'Xy_train', 'csv']))
            cur_Xy_test_path = os.path.join(outDir, '.'.join([filename, 'Xy_test', 'csv']))
            cur_y_train_path = os.path.join(outDir, '.'.join([filename, 'y_train', 'csv']))
            cur_y_test_path = os.path.join(outDir, '.'.join([filename, 'y_test', 'csv']))
            print ("cur_Xy_train_path = ", cur_Xy_train_path)
            with open(cur_Xy_train_path, 'r') as rfd1:
                shutil.copyfileobj(rfd1, fd1)
            with open(cur_Xy_test_path, 'r') as rfd2:
                shutil.copyfileobj(rfd2, fd2)
            with open(cur_y_train_path, 'r') as rfd3:
                shutil.copyfileobj(rfd3, fd3)
            with open(cur_y_test_path, 'r') as rfd4:
                shutil.copyfileobj(rfd4, fd4)
            # Remove the intermediate files
            os.remove(cur_Xy_train_path)
            os.remove(cur_Xy_test_path)
            os.remove(cur_y_train_path)
            os.remove(cur_y_test_path)
        fd1.close()
        fd2.close()
        fd3.close()
        fd4.close()
            
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
#     exit(0)
    # 2. Create train-test rows
    trainTestSplit()