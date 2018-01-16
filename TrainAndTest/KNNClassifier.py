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
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc
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
    
    def test(self, f, patient_num, total_fpr, total_tpr):
        y_pred = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = accuracy_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        print("Accuracy: %.2f" % accuracy)
        print("Precision: %.2f" % precision)
        print("Recall: %.2f" % recall)
        line = str(accuracy)+","+str(precision)+","+str(recall)
        f.write(line)
        f.write("\n")
        confmat = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.savefig("D:\\Documents\\KNN\\LineLength\\chb"+patient_num+"_confmat.png")
        plt.close()
        fpr, tpr, threshholds = roc_curve(self.y_test, y_pred)
        print("fpr", fpr)
        print("tpr", tpr)
        total_fpr[1]+=fpr[len(fpr)-2]
        total_tpr[1]+=tpr[len(tpr)-2]
        print(total_fpr)
        print(total_tpr)
        roc_auc = auc(fpr, tpr)
        plt.title('ROC Curve')
        plt.plot(fpr, tpr, 'b', label='AUC = %.2F' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0,1], [0,1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("D:\\Documents\\KNN\\LineLength\\chb"+patient_num+"roc.png")
        plt.close()
        return total_fpr, total_tpr