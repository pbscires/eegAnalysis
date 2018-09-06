import  tensorflow  as tf
import numpy as np
import pandas as pd
import shutil
from tensorflow.contrib.learn import ModeKeys
import tensorflow.contrib.rnn as rnn

SEQ_LEN = 100
N_COLUMNS = SEQ_LEN
DEFAULTS = [[0.0] for x in range(0, N_COLUMNS)]
BATCH_SIZE = 100
TIMESERIES_COL = 'rawdata'
N_OUTPUTS = 10  # in each sequence, 1-18 are features, and 19-20 is label
N_INPUTS = SEQ_LEN - N_OUTPUTS

# read data and convert to needed format
def read_dataset(filename, mode=ModeKeys.TRAIN):
    def _input_fn():
        num_epochs = 1
        
        # could be a path to one file or a file pattern.
        input_file_names = tf.train.match_filenames_once(filename, name='match_fname')
        
    
        filename_queue = tf.train.string_input_producer(
            input_file_names, num_epochs=num_epochs, shuffle=True)
        reader = tf.TextLineReader(name='txtLineReader')
        _, value = reader.read_up_to(filename_queue, num_records=BATCH_SIZE, name='read_upto')
    
        value_column = tf.expand_dims(value, -1, name='value_column')
        # value_column = tf.Print(value_column, [value_column], message='printing value column')
        #value_column = value
        print('readcsv, after expand_dims={}'.format(value_column))
        # tf.summary.scalar('value_column', value_column)

        # all_data is a list of tensors
        all_data = tf.decode_csv(value_column, record_defaults=DEFAULTS)  
        inputs = all_data[:len(all_data)-N_OUTPUTS]  # first few values
        label = all_data[len(all_data)-N_OUTPUTS : ] # last few values
        
        # from list of tensors to tensor with one more dimension
        inputs = tf.concat(inputs, axis=1)
        label = tf.concat(label, axis=1)
        print('inputs={}'.format(inputs))
        # inputs = tf.Print(inputs, [inputs], message='printing inputs')
    
        return {TIMESERIES_COL: inputs}, label   # dict of features, label

    return _input_fn


LSTM_SIZE = 3  # number of hidden layers in each of the LSTM cells

# create the inference model
def simple_rnn(features, labels, mode, params):
    print("features=", features, "labels = ", labels, "mode = ", mode,
          "params = ", params)

    # 0. Reformat input shape to become a sequence
    # The last parameter '0' refers to the axis of the data
    x = tf.split(features[TIMESERIES_COL], N_INPUTS, 1)
    # print ('x={}'.format(x))

    # 1. configure the RNN
    lstm_cell = rnn.BasicLSTMCell(LSTM_SIZE, forget_bias=1.0)
    outputs, _ = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # slice to keep only the last cell of the RNN
    outputs = outputs[-1]
    print ('last outputs={}'.format(outputs))
  
    # output is result of linear activation of last layer of RNN
    weight = tf.Variable(tf.random_normal([LSTM_SIZE, N_OUTPUTS]))
    bias = tf.Variable(tf.random_normal([N_OUTPUTS]))
    predictions = tf.matmul(outputs, weight) + bias
    
    # 2. loss function, training/eval ops
    if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(labels, predictions)
        train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=0.01,
                optimizer="SGD")
        eval_metric_ops = {
                "rmse": tf.metrics.root_mean_squared_error(labels, predictions)
                }
    else:
        loss = None
        train_op = None
        eval_metric_ops = None
  
    # 3. Create predictions
    predictions_dict = {"predicted": predictions}

    # 4. Create export outputs  
    export_outputs = {"predicted": tf.estimator.export.PredictOutput(predictions)}
  
  
    # 5. return ModelFnOps
    return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions_dict,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            export_outputs=export_outputs)


def get_train():
    return read_dataset('chb03/LineLength.chb03_04.edf_feature_FP1-F7_train.csv', mode=ModeKeys.TRAIN)

def get_valid():
    return read_dataset('chb03/LineLength.chb03_04.edf_feature_FP1-F7_valid.csv', mode=ModeKeys.EVAL)

def serving_input_receiver_fn():
    feature_placeholders = {
        TIMESERIES_COL: tf.placeholder(tf.float32, [None, N_INPUTS])
    }

    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }

    features[TIMESERIES_COL] = tf.squeeze(features[TIMESERIES_COL], axis=[2], name='timeseries')
  
    print('serving: features={}'.format(features[TIMESERIES_COL]))

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

def experiment_fn(output_dir):
    train_spec = tf.estimator.TrainSpec(input_fn=get_train(), max_steps=1000)
    exporter = tf.estimator.FinalExporter('timeseries',
                                          serving_input_receiver_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=get_valid(), 
                                      exporters=[exporter])

    estimator = tf.estimator.Estimator(model_fn=simple_rnn, model_dir=output_dir)
          
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':

    print (tf.__version__)
    
    OUTPUT_DIR = 'outputdir7'
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True) # start fresh each time
    
    experiment_fn(OUTPUT_DIR)

