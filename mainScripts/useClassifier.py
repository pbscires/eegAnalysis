'''
Created on Dec 26, 2017

@author: pb8xe
'''

from TrainAndTest.KNNClassifier import KNNClassifier
import sys
from TrainAndTest.DNNClassifier import DNNClassifier

if __name__ == '__main__':
    print(sys.argv[1])
#     knn_classifier = KNNClassifier(sys.argv[1], k=5)
#     knn_classifier.create_arrays()
#     knn_classifier.preprocess()
#     knn_classifier.train_test()
    
    dnn_classifier = DNNClassifier(sys.argv[1], sys.argv[2])
#     dnn_classifier.create_arrays()
#     dnn_classifier.preprocess()
    dnn_classifier.train()
    dnn_classifier.test()