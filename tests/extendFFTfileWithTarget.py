'''

'''

import os
import numpy as np

if __name__ == '__main__':
    LL_inDir = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/LineLength'
    FFT_inDir = '/Users/rsburugula/Documents/Etc/Pranav/YHS/ScienceResearch/Data/output/FFT'
    LLfiles = os.listdir(LL_inDir)
    print ("number of files in ", LL_inDir, " = ", len(LLfiles))
    for LLfilename in LLfiles:
        FFTfilename = LLfilename.replace('LineLength', 'FFT')
        LLfilepath = os.path.join(LL_inDir, LLfilename)
        FFTfilepath = os.path.join(FFT_inDir, FFTfilename)
        arr1 = np.genfromtxt(LLfilepath, delimiter=',')
        arr2 = np.genfromtxt(FFTfilepath, delimiter=',')
        if (arr1.shape[0] != arr2.shape[0]):
            print('Number of rows in arr1 and arr2 are different')
        else:
            print('Number of rows in arr1 and arr2 are same')
            print("Modifying the file", FFTfilename, "using the file", LLfilename)
            y = arr1[:, -1]
            arr3 = np.concatenate((arr2, np.reshape(y, (-1, 1))), axis=1)
            fmtStr = '%f,' * arr2.shape[1]
            fmtStr = ''.join([fmtStr, '%d'])
            np.savetxt(FFTfilepath, arr3, fmt=fmtStr, delimiter=',')