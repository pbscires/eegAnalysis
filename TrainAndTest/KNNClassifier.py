'''
Created on Dec 26, 2017

@author: pb8xe
'''

import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing.data import StandardScaler
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class KNNClassifier():
    '''
    classdocs
    '''


    def __init__(self, csv_path, k):
        '''
        Constructor
        '''
        self.csv_path = csv_path
        self.k = k
    
    def create_arrays(self):
        arr = np.genfromtxt(self.csv_path, delimiter=',')
        print(arr.shape)
        self.X = np.delete(arr, [arr.shape[1]-1], axis=1)
        self.X = np.delete(self.X, [0], axis=0)
        print(self.X.shape)
        self.y = np.delete(arr, list(range(arr.shape[1]-1)), axis=1)
        self.y = np.delete(self.y, [0], axis=0)
        print(self.y.shape)
    
    def preprocess(self):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        sc = StandardScaler()
        sc.fit(X_train)
        self.X_train_std = sc.transform(X_train)
        self.X_test_std = sc.transform(X_test)
    
    def train_test(self):
        knn = KNeighborsClassifier(n_neighbors=self.k, p=2, metric='minkowski')
        knn.fit(self.X_train_std, self.y_train.ravel())
        y_pred = knn.predict(self.X_test_std)
        print('Accuracy: %.2f' % accuracy_score(self.y_test, y_pred))