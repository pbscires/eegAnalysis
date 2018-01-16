'''
Created on Dec 31, 2017

@author: pb8xe
'''
import numpy as np
import collections
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing.data import StandardScaler
import tensorflow as tf
from util.ElapsedTime import ElapsedTime
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
from tensorflow.contrib.metrics.python.metrics.classification import accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

class DNNClassifier(object):
    '''
    classdocs
    '''


    def __init__(self, csv_path_train, csv_path_test):
        '''
        Constructor
        '''
        self.csv_path_train = csv_path_train
        self.csv_path_test = csv_path_test
        self.Dataset = collections.namedtuple('Dataset', ['data', 'target'])
        
        # Specify that all features have real-value data
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=184)]
        # Build 3 layer DNN with 10, 20, 10 units respectively.
        self.classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                    hidden_units=[10, 20, 10],
                                                    n_classes=2,
                                                    model_dir="DNNClassifier_Data")
            
        print('Initialized')

    # Define the training inputs
    def get_train_inputs(self):
        x = tf.constant(self.train_dataset.data)
        y = tf.constant(self.train_dataset.target)
    
        return x, y

    def create_arrays(self):
        arr_train = np.genfromtxt(self.csv_path_train, delimiter=',', skip_header=1)
        print(arr_train.shape)
        self.X_train = np.delete(arr_train, [arr_train.shape[1]-1], axis=1)
        print(self.X_train.shape)
        self.y_train = np.delete(arr_train, list(range(arr_train.shape[1]-1)), axis=1)
        print(self.y_train.shape)
        
        arr_test = np.genfromtxt(self.csv_path_test, delimiter=',', skip_header=1)
        print(arr_test.shape)
        self.X_test = np.delete(arr_test, [arr_test.shape[1]-1], axis=1)
        print(self.X_test.shape)
        self.y_test = np.delete(arr_test, list(range(arr_test.shape[1]-1)), axis=1)
        print(self.y_test.shape)
    
    def preprocess(self):
        sc = StandardScaler()
        sc.fit(self.X_train)
        X_train_std = sc.transform(self.X_train)
        X_test_std = sc.transform(self.X_test)
        self.train_dataset = self.Dataset(data=self.X_train, target=self.y_train)
        self.test_dataset = self.Dataset(data=self.X_test, target=self.y_test)
    
    def train(self):
        self.create_arrays()
        self.preprocess()
        timer = ElapsedTime()
        timer.reset()
        print('Started training')

# Fit model.
        self.classifier.fit(input_fn=self.get_train_inputs, steps=2000)
        print('Ended in: ', timer.timeDiff())
        
    def get_test_inputs(self):
        x = tf.constant(self.test_dataset.data)
        y = tf.constant(self.test_dataset.target)
        
        return x, y

    def get_test_inputs_only(self):
        return tf.constant(self.test_dataset.data)
    
    def test(self, f, patient_num, total_fpr, total_tpr):
        print('Started testing')

        score = list(self.classifier.predict_classes(input_fn=self.get_test_inputs_only))
        y_pred = np.array(score).astype(int)
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
        plt.savefig("D:\\Documents\\DNN\\FFT\\chb"+patient_num+"_confmat.png")
        plt.close()
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)
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
        plt.savefig("D:\\Documents\\DNN\\FFT\\chb"+patient_num+"roc.png")
        plt.close()
        return total_fpr, total_tpr