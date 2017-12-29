'''
Created on Dec 26, 2017

@author: pb8xe
'''

from TrainAndTest.KNNClassifier import KNNClassifier
import sys

if __name__ == '__main__':
    print(sys.argv[1])
    classifier = KNNClassifier(sys.argv[1])
    classifier.create_arrays()
    classifier.preprocess()
    classifier.train_test()