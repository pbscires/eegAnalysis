'''

'''
import sys
from Features.FastFourierTransform import FFT
from DataRepresentation.EEGRecord import EEGRecord


if __name__ == '__main__':
    edfFile = sys.argv[1]
    print ("edfFile = ", edfFile)
    eegRecObj = EEGRecord(edfFile, "chb03")
    eegRecObj.loadFile()
    sigbufs = eegRecObj.getSigbufs()
    sampleFrequency = eegRecObj.getSampleFrequency()
    signal_labels = eegRecObj.getSingalLabels()

    fftObj = FFT()
    fftObj.extractFeature(sigbufs, signal_labels, sampleFrequency)