'''
Created on Dec 31, 2017

@author: pb8xe
'''
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing.data import StandardScaler
import tensorflow as tf
from util.ElapsedTime import ElapsedTime

class DNNClassifier(object):
    '''
    classdocs
    '''


    def __init__(self, csv_path_train, csv_path_test):
        '''
        Constructor
        '''
        self.csv_path = csv_path_train
        self.training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
            filename=csv_path_train,
            target_dtype=np.int,
            features_dtype=np.float64)
        self.test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
                filename=csv_path_test,
                target_dtype=np.int,
                features_dtype=np.float64)
        with tf.device('/cpu:0'):
            feature_columns = [tf.feature_column.numeric_column("x", shape=[21])]
            self.classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[100,10],
                                                n_classes=2,
                                                model_dir="DNNClassifier_data")
        
    def create_arrays(self):
        arr = np.genfromtxt(self.csv_path, delimiter=',')
        print(arr.shape)
        self.X = np.delete(arr, [arr.shape[1]-1], axis=1)
        print(self.X.shape)
        self.y = np.delete(arr, list(range(arr.shape[1]-1)), axis=1)
        print(self.y.shape)
    
    def preprocess(self):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        sc = StandardScaler()
        sc.fit(X_train)
        self.X_train_std = sc.transform(X_train)
        self.X_test_std = sc.transform(X_test)
    
    def train(self):
        timer = ElapsedTime()
        timer.reset()
        with tf.device('/cpu:0'):
            train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x" : np.array(self.training_set.data)}, y=self.training_set.target, num_epochs=None, shuffle=False)
        
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            self.classifier.train(input_fn=train_input_fn, steps=1)
        
        print(timer.timeDiff())
        
    def test(self):
        with tf.device('/cpu:0'):
            test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x" : np.array(self.test_set.data)}, y=self.test_set.target, num_epochs=None, shuffle=False)
        
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            accuracy_score = self.classifier.evaluate(input_fn=test_input_fn)['accuracy']
        
        print("Accuracy score: " + accuracy_score)