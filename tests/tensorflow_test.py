from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import tensorflow as tf
import numpy as np

from datetime import datetime

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

raw = urlopen(IRIS_TRAINING_URL).read()
with open(IRIS_TRAINING, 'wb') as f:
    f.write(raw)

raw = urlopen(IRIS_TEST_URL).read()
with open(IRIS_TEST, 'wb') as f:
    f.write(raw)

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

with tf.device('/gpu:0'):
    feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10,20,10],
                                            n_classes=3,
                                            model_dir="iris_model6")
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(training_set.data)},
            y=np.array(training_set.target),
            num_epochs=None,
            shuffle=True)
    
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(test_set.data)},
            y=np.array(test_set.target),
            num_epochs=1,
            shuffle=False)

startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    train = classifier.train(input_fn=train_input_fn, steps=1000)
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print(accuracy_score)
    
print('Time:', datetime.now()-startTime)