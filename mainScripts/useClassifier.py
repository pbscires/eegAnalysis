'''
Created on Dec 26, 2017

@author: pb8xe
'''

from TrainAndTest.KNNClassifier import KNNClassifier
import sys
from TrainAndTest.DNNClassifier import DNNClassifier
from TrainAndTest.SVMClassifier import SVMClassifier
if __name__ == '__main__':
    print(sys.argv[1])
    print(sys.argv[2])
#     knn_classifier = KNNClassifier(sys.argv[1], sys.argv[2], k=5)
#     knn_classifier.train()
#     knn_classifier.test()
# 
#     svm_classifier = SVMClassifier(sys.argv[1], sys.argv[2])
#     svm_classifier.train()
#     svm_classifier.test()
#     
    dnn_classifier = DNNClassifier(sys.argv[1], sys.argv[2])
    dnn_classifier.train()
    dnn_classifier.test()

#     svm_classifier = SVMClassifier(sys.argv[1], sys.argv[2])
#     svm_classifier.validation()