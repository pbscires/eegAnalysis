import numpy as np
import pandas as pd
import sys
import os

# Reorganize the chb<nn>.csv file into multiple smaller
# .csv files each containing the time series data for one channel.

# Inputs: Number of epochs per row, column_headers
# Outputs: chb03_feature01_epochs.csv
#          chb03_feature02_epochs.csv

feature_names = ["FP1-F7", "F7-T7", "T7-P7",
        "P7-O1", "FP1-F3", "F3-C3", "C3-P3", 
        "P3-O1", "FP2-F4", "F4-C4", "C4-P4", 
        "P4-O2", "FP2-F8", "F8-T8", "T8-P8", 
        "P8-O2", "FZ-CZ", "CZ-PZ", "P7-T7", 
        "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8"]

sequenceLen = 100
numChannels = 19

def read_input_file(filename):
    print ("filename = ", filename)
    df = pd.read_csv(filename, usecols=range(0,numChannels), names=[feature_names[i] for i in range(numChannels)])
    print (df.head(n=10))
    return (df)

def generateCSVperFeature(channelNum, feature_files, df):
    print ("generating CSV file ", feature_files[channelNum])
    # df = df.splice()
    feature_series = df[feature_names[channelNum]]
    print ("feature_series shape = ", feature_series.shape)
    # Reshape the feature_series data frame into n_rows x sequenceLen matrix,
    # where n_rows = feature_series.shape[0] // seuquenceLen
    last_row = feature_series.shape[0] - sequenceLen
    # feature_df = pd.DataFrame(np.zeros((last_row, sequenceLen), dtype=np.float))
    feature_df = pd.DataFrame(dtype=np.float)
    print ("feature_df.shape = ", feature_df.shape)
    for i in range(0, last_row):
        feature_df[i] = np.array(feature_series[i:(i+sequenceLen)])
    feature_df = feature_df.transpose()
    print ("feature_df.shape = ", feature_df.shape)
    # return
    # n_rows = feature_series.shape[0] // sequenceLen
    # n_cols = sequenceLen
    # print ("n_rows * n_cols = ", n_rows * n_cols)
    # feature_series = feature_series.truncate(after = (n_rows * n_cols - 1))
    # print (feature_series.shape)
    # feature_series = pd.DataFrame(feature_series.values.reshape(n_rows, n_cols))
    # print (feature_series.shape)
    # feature_df = pd.DataFrame(data=feature_series)
    feature_df.to_csv(feature_files[channelNum], header=False, index=False)

if __name__ == '__main__':
    print ("in main")
    csv_path = sys.argv[1]
    print ("csv file to read = ", csv_path)
    basedir = os.path.dirname(csv_path)
    basename = os.path.basename(csv_path)
    print ("basedir = ", basedir, "basename = ", basename)
    (basefilename, ext) = os.path.splitext(basename)
    print ("basefilename = ", basefilename, "ext = ", ext)
    feature_files = []
    for i in range(0, numChannels):
        fname = basefilename + '_feature_' + feature_names[i] + ext
        feature_files.append(os.path.join(basedir, fname))
    
    df = read_input_file(csv_path)

    # print (feature_files)
    for i in range(numChannels):
        generateCSVperFeature(i, feature_files, df)
