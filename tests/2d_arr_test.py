'''

'''
import numpy as np

def add1(nparr):
    return

if __name__ == '__main__':
    fftMat = np.zeros((2,23*8))
    for i in range(fftMat.shape[0]):
        for j in range(23):
            for k in range(8):
                fftMat[i,(j*8)+k] = ((i * 23 * 8)  + (j * 8) + k)
    print(fftMat)
    for i in range(fftMat.shape[0]):
        strToWrite = ''
#         for j in range(fftMat.shape[1]):
        strToWrite = ','.join(['%d' % num for num in fftMat[i,:]])
#             strToWrite = ','.join([strToWrite, newStr])
#     strToWrite = ','.join([str(i), strToWrite, str(seizureValue)])

        print ("strToWrite = %s" % strToWrite)