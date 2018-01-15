'''
Created on Dec 26, 2017

@author: pb8xe
'''

import numpy as np
import time
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing.data import StandardScaler
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNNClassifier():
    '''
    classdocs
    '''


    def __init__(self, csv_path_train, csv_path_test, k):
        '''
        Constructor
        '''
        self.csv_path_train = csv_path_train
        self.csv_path_test = csv_path_test
        self.classifier = KNeighborsClassifier(n_neighbors=k, p=2, metric='minkowski')
    
    def create_arrays(self):
        arr_train = np.genfromtxt(self.csv_path_train, delimiter=',', skip_header=1)
        self.X_train = np.delete(arr_train, [arr_train.shape[1]-1], axis=1)
        self.y_train = np.delete(arr_train, list(range(arr_train.shape[1]-1)), axis=1)
        
        arr_test = np.genfromtxt(self.csv_path_test, delimiter=',', skip_header=1)
        self.X_test = np.delete(arr_test, [arr_test.shape[1]-1], axis=1)
        self.y_test = np.delete(arr_test, list(range(arr_test.shape[1]-1)), axis=1)
    
    def preprocess(self):
#         X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)
    
    def train(self):
        self.create_arrays()
        self.preprocess()
        self.classifier.fit(self.X_train_std, self.y_train.ravel())
    
    def test(self):
        y_pred = self.classifier.predict(self.X_test_std)
        print('Accuracy: %.2f' % accuracy_score(self.y_test, y_pred))