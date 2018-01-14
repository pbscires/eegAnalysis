'''
Created on Jan 14, 2018

@author: pb8xe
'''
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
class SVMClassifier(object):
    '''
    classdocs
    '''


    def __init__(self, csv_path):
        '''
        Constructor
        '''
        self.csv_path = csv_path
        self.classifier = SVC(kernel='rbf', random_state=0, gamma=.1, C=10.0)
    
    def create_arrays(self):
        arr = np.genfromtxt(self.csv_path, delimiter=',')
        print(arr.shape)
        self.X = np.delete(arr, [arr.shape[1]-1], axis=1)
        self.X = np.delete(self.X, 0, axis=0)
        print(self.X.shape)
        self.y = np.delete(arr, list(range(arr.shape[1]-1)), axis=1)
        self.y = np.delete(self.y, 0, axis=0)
        print(self.y.shape)
    
    def preprocess(self):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        sc = StandardScaler()
        sc.fit(X_train)
        self.X_train_std = sc.transform(X_train)
        self.X_test_std = sc.transform(X_test)
    
    def train(self):
        self.classifier.fit(self.X_train_std, self.y_train.ravel())
    
    def test(self):
        y_pred = self.classifier.predict(self.X_test_std)
        print("Accuracy: %.2f" % accuracy_score(self.y_test, y_pred))