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
#         self.training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#             filename=csv_path_train,
#             target_dtype=np.int,
#             features_dtype=np.float32)
#         self.training_set_target = tf.contrib.learn.datasets.base.load_csv_with_header(
#             filename=csv_path_train_target,
#             features_dtype=np.int,
#             target_dtype=np.int)
#         self.test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#                 filename=csv_path_test,
#                 features_dtype=np.float32,
#                 target_dtype=np.int)
#         self.test_set_target = tf.contrib.learn.datasets.base.load_csv_with_header(
#             filename=csv_path_test_target,
#             features_dtype=np.int,
#             target_dtype=np.int)
        
        # Specify that all features have real-value data
        feature_columns = [tf.contrib.layers.real_valued_column("", dimension=184)]
        # Build 3 layer DNN with 10, 20, 10 units respectively.
        self.classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                    hidden_units=[10, 20, 10],
                                                    n_classes=2,
                                                    model_dir="DNNClassifier_Data")
#         with tf.device('/gpu:0'):
#             feature_columns = [tf.feature_column.numeric_column("x", shape=[21])]
#             self.classifier = SKCompat(tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
#                                                 hidden_units=[100,10],
#                                                 n_classes=2,
#                                                 model_dir="DNNClassifier_data"))
            
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
#         X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=0)
        sc = StandardScaler()
        sc.fit(self.X_train)
        X_train_std = sc.transform(self.X_train)
        X_test_std = sc.transform(self.X_test)
        self.train_dataset = self.Dataset(data=X_train_std, target=self.y_train)
        self.test_dataset = self.Dataset(data=X_test_std, target=self.y_test)
    
    def train(self):
        self.create_arrays()
        self.preprocess()
        timer = ElapsedTime()
        timer.reset()
        print('Started training')
#         with tf.device('/gpu:0'):
#             train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x" : np.array(self.training_set.data)}, y=self.training_set.target, num_epochs=None, shuffle=False)
#         
#         with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#             self.classifier.fit(x={"x" : np.array(self.training_set.data)}, y=self.training_set.target, steps=1)
# Fit model.
        self.classifier.fit(input_fn=self.get_train_inputs, steps=2000)
        print('Ended in: ', timer.timeDiff())
        
    def get_test_inputs(self):
        x = tf.constant(self.test_dataset.data)
        y = tf.constant(self.test_dataset.target)
        
        return x, y

    def test(self):
        print('Started testing')
#         with tf.device('/gpu:0'):
#             print('in test data creation')
#             test_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x" : np.array(self.test_set.data)}, y=self.test_set.target, num_epochs=None, shuffle=False)
#         
#         with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
#             print('In session')
# #             accuracy_score = self.classifier.score(x={"x" : np.array(self.test_set.data)}, y=self.test_set.target, metrics={'accuracy' : accuracy})
#             accuracy_score = accuracy(self.classifier.predict(x={"x" : np.array(self.test_set.data)}), labels=self.test_set.target)
#             print('Done testing')
#           # Define the test inputs
        # Evaluate accuracy.
        accuracy_score = self.classifier.evaluate(input_fn=self.get_test_inputs,
                                             steps=1)["accuracy"]
        print("Accuracy score: ", accuracy_score)