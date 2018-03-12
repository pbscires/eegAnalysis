'''
Created on Dec 26, 2017

@author: pb8xe
'''

from TrainAndTest.KNNClassifier import KNNClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing.data import StandardScaler
import sys
import os
import numpy as np
from TrainAndTest.DNNClassifier import DNNClassifier
from TrainAndTest.SVMClassifier import SVMClassifier
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn.metrics.ranking import roc_auc_score
from scipy.interpolate.fitpack2 import UnivariateSpline
from numpy import dtype, float32
from util.JsonReader import JsonReader
from util.ConfigJsonReader import ConfigJsonReader


pgmName = sys.argv[0]
if (len(sys.argv) > 1):
    cfgFilename = sys.argv[1]
else:
    cfgFilename = pgmName.replace(".py", ".json")

def readArguments():
    global jsonData
    global csvInDir, resultsFile
    global subjectNames
    global seizuresFile
    global features
    global outDir
    global subjectFiles
    global preictalOffsets
    

    print ("cfgFilename = ", cfgFilename)
    jsonData = JsonReader(cfgFilename)
    subjectNames = jsonData.get_value("SubjectNames")
    csvInDir = jsonData.get_value("CSVInDir")
    resultsFile = jsonData.get_value("resultsFile")
    preictalOffsets = jsonData.get_value("PreIctalOffsets")

    print ("subjectNames = ", subjectNames)
    print ("csvInDir = ", csvInDir)
    print ("resultsFile = ", resultsFile)

    return

def processSubject(subject):
    global csvInDir, resultsFile, f
    global total_fpr_knn, total_tpr_knn, total_confmat_knn, knn_accuracies, knn_f1s, knn_precisions, knn_recalls
    global total_fpr_dnn, total_tpr_dnn, total_confmat_dnn, dnn_accuracies, dnn_f1s, dnn_precisions, dnn_recalls
    global total_fpr_svm, total_tpr_svm, total_confmat_svm, svm_accuracies, svm_f1s, svm_precisions, svm_recalls

    print ("Processing per-subject files for ", subject)
    subjectPath = os.path.join(csvInDir, subject+'.csv')
    print ("subject path = ", subjectPath)
#     root="D:\\Documents\\LL_PreIctal\\chb"
#     f = open('D:\\Documents\\multiple\\resultsLL.csv', 'w')

#         filepath = root + "04" + ".csv"

    arr = np.genfromtxt(subjectPath, delimiter=',')
    X = np.delete(arr, [arr.shape[1]-1], axis=1)
    y = np.delete(arr, list(range(arr.shape[1]-1)), axis=1)
    Xns = []
    yns = []
    Xs = []
    ys = []

    for i in range(len(y)):
        if y[i] == 0:
            Xns.append(X[i])
            yns.append(y[i])
        elif y[i] == 1:
            Xs.append(X[i])
            ys.append(y[i])
    Xs = np.array(Xs)
    ys = np.array(ys)
    Xns = np.array(Xns)
    yns = np.array(yns)
    
    if Xns.shape[0] > 10000:
        print("Reshaping")
        Xns.resize((10000, Xns.shape[1]))
        yns.resize((10000, yns.shape[1]))
    
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.3, random_state=0)
    Xns_train, Xns_test, yns_train, yns_test = train_test_split(Xns, yns, test_size=0.3, random_state=0)
    
    X_train = np.vstack((Xns_train, Xs_train))
    y_train = np.vstack((yns_train, ys_train))
    X_test = np.vstack((Xns_test, Xs_test))
    y_test = np.vstack((yns_test, ys_test))

    print("Train:")
    print("Xs_train: ", Xs_train.shape)
    print("Xns_train: ", Xns_train.shape)
    print("ys_train: ", ys_train.shape)
    print("yns_train: ", yns_train.shape)
    print("X_train: ", X_train)
    print("y_train: ", y_train)
    
    print("Test:")
    print("Xs_test: ", Xs_test.shape)
    print("Xns_test: ", Xns_test.shape)
    print("ys_test: ", ys_test.shape)
    print("yns_test: ", yns_test.shape)
    print("X_test: ", X_test)
    print("y_test: ", y_test)
    
#     exit(0)
    
#    for i in range(y_test.shape[0]):
#        if y_test[i]==1:
#            print("true")
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    knn_classifier = KNNClassifier(X_train_std, y_train, X_test_std, y_test, k=5)
    knn_classifier.train()
    knn_accuracy, knn_precision, knn_recall, knn_f1, total_confmat_knn, total_fpr_knn, total_tpr_knn = knn_classifier.test(f, subject, total_confmat_knn, total_fpr_knn, total_tpr_knn)
    knn_accuracies+=knn_accuracy
    knn_precisions+=knn_precision
    knn_recalls+=knn_recall
    knn_f1s+=knn_f1
         
    dnn_classifier = DNNClassifier(X_train_std, y_train, X_test_std, y_test)
    dnn_classifier.train()
    dnn_accuracy, dnn_precision, dnn_recall, dnn_f1, total_confmat_dnn, total_fpr_dnn, total_tpr_dnn = dnn_classifier.test(f, subject, total_confmat_dnn, total_fpr_dnn, total_tpr_dnn)
    dnn_accuracies+=dnn_accuracy
    dnn_precisions+=dnn_precision
    dnn_recalls+=dnn_recall
    dnn_f1s+=dnn_f1
      
    svm_classifier = SVMClassifier(X_train_std, y_train, X_test_std, y_test)
    svm_classifier.train()
    svm_accuracy, svm_precision, svm_recall, svm_f1, total_confmat_svm, total_fpr_svm, total_tpr_svm = svm_classifier.test(f, subject, total_confmat_svm, total_fpr_svm, total_tpr_svm)
    svm_accuracies+=svm_accuracy
    svm_precisions+=svm_precision
    svm_recalls+=svm_recall
    svm_f1s+=svm_f1



if __name__ == '__main__':
    global csvInDir, resultsFile
    global subjectNames
    global f
    global preictalOffsets
    global total_fpr_knn, total_tpr_knn, total_confmat_knn, knn_accuracies, knn_f1s, knn_precisions, knn_recalls
    global total_fpr_dnn, total_tpr_dnn, total_confmat_dnn, dnn_accuracies, dnn_f1s, dnn_precisions, dnn_recalls
    global total_fpr_svm, total_tpr_svm, total_confmat_svm, svm_accuracies, svm_f1s, svm_precisions, svm_recalls

    print ("pgmName = ", pgmName)
    readArguments()
#     exit(0)
    
    f = open(resultsFile, 'w')
    total_fpr_knn=None
    total_tpr_knn=None
    total_confmat_knn=np.zeros([2,2])
    knn_accuracies=0.0
    knn_precisions=0.0
    knn_recalls=0.0
    knn_f1s = 0.0

    total_fpr_svm=None
    total_tpr_svm=None
    total_confmat_svm=np.zeros([2,2])
    svm_accuracies=0.0
    svm_precisions=0.0
    svm_recalls=0.0
    svm_f1s = 0.0

    total_fpr_dnn=None
    total_tpr_dnn=None
    total_confmat_dnn=np.zeros([2,2])
    dnn_accuracies=0.0
    dnn_precisions=0.0
    dnn_recalls=0.0
    dnn_f1s = 0.0
    
    for subject in subjectNames:
        processSubject(subject)
        processSubject(subject+".preictal."+str(preictalOffsets[0]))
        processSubject(subject+".preictal."+str(preictalOffsets[1]))
        processSubject(subject+".preictal."+str(preictalOffsets[2]))
        processSubject(subject+".preictal."+str(preictalOffsets[3]))
        processSubject(subject+".preictal."+str(preictalOffsets[4]))
        processSubject(subject+".preictal."+str(preictalOffsets[5]))
        
    f.write("\n")
    f.write(str(knn_accuracies/22)+","+str(knn_precisions/22)+","+str(knn_recalls/22)+","+str(knn_f1s/22)+","+
            str(dnn_accuracies/22)+","+str(dnn_precisions/22)+","+str(dnn_recalls/22)+","+str(dnn_f1s/22)+","+
            str(svm_accuracies/22)+","+str(svm_precisions/22)+","+str(svm_recalls/22)+","+str(svm_f1s/22))
#     f.write(str(svm_accuracies/22)+","+str(svm_precisions/22)+","+str(svm_recalls/22)+","+str(svm_f1s/22))
      
    f.close()
    
#     exit(0)

    
#     for index in range(0,23):
#         num = index+1
#         if(num==16):
#             continue
#         if num<10:
#             num_string="0" + str(num)
#         else:
#             num_string=str(num)
#         filepath = root + num_string + ".csv"
#          
#         arr = np.genfromtxt(filepath, delimiter=',')
#         X = np.delete(arr, [arr.shape[1]-1], axis=1)
#         y = np.delete(arr, list(range(arr.shape[1]-1)), axis=1)
#         Xns = []
#         yns = []
#         Xs = []
#         ys = []
#         
#         for i in range(len(y)):
#             if y[i] == 0:
#                 Xns.append(X[i])
#                 yns.append(y[i])
#             elif y[i] == 1:
#                 Xs.append(X[i])
#                 ys.append(y[i])
#         Xs = np.array(Xs)
#         ys = np.array(ys)
#         Xns = np.array(Xns)
#         yns = np.array(yns)
#         Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, test_size=0.3, random_state=0)
#         Xns_train, Xns_test, yns_train, yns_test = train_test_split(Xns, yns, test_size=0.3, random_state=0)
#         X_train = np.concatenate(Xns_train, Xs_train)
#         y_train = np.concatenate(yns_train, ys_train)
#         X_test = np.concatenate(Xns_test, Xs_test)
#         y_test = np.concatenate(yns_test, ys_test)
#         sc = StandardScaler()
#         sc.fit(X_train)
#         X_train_std = sc.transform(X_train)
#         X_test_std = sc.transform(X_test)
#          
#         print("Currently on chb"+num_string)
#           
#         knn_classifier = KNNClassifier(X_train_std, y_train, X_test_std, y_test, k=5)
#         knn_classifier.train()
#         knn_accuracy, knn_precision, knn_recall, knn_f1, total_confmat_knn, total_fpr_knn, total_tpr_knn, total_auc_knn = knn_classifier.test(f, num_string, total_confmat_knn, total_fpr_knn, total_tpr_knn, total_auc_knn)
#         knn_accuracies+=knn_accuracy
#         knn_precisions+=knn_precision
#         knn_recalls+=knn_recall
#         knn_f1s+=knn_f1
#             
#         dnn_classifier = DNNClassifier(X_train_std, y_train, X_test_std, y_test)
#         dnn_classifier.train()
#         dnn_accuracy, dnn_precision, dnn_recall, dnn_f1, total_confmat_dnn, total_fpr_dnn, total_tpr_dnn, total_auc_dnn = dnn_classifier.test(f, num_string, total_confmat_dnn, total_fpr_dnn, total_tpr_dnn, total_auc_dnn)
#         dnn_accuracies+=dnn_accuracy
#         dnn_precisions+=dnn_precision
#         dnn_recalls+=dnn_recall
#         dnn_f1s+=dnn_f1
#          
#         svm_classifier = SVMClassifier(X_train_std, y_train, X_test_std, y_test)
#         svm_classifier.train()
#         svm_accuracy, svm_precision, svm_recall, svm_f1, total_confmat_svm, total_fpr_svm, total_tpr_svm, total_auc_svm = svm_classifier.test(f, num_string, total_confmat_svm, total_fpr_svm, total_tpr_svm, total_auc_svm)
#         svm_accuracies+=svm_accuracy
#         svm_precisions+=svm_precision
#         svm_recalls+=svm_recall
#         svm_f1s+=svm_f1
#          
#     f.write("\n")
#     f.write(str(knn_accuracies/22)+","+str(knn_precisions/22)+","+str(knn_recalls/22)+","+str(knn_f1s/22)+","+
#             str(dnn_accuracies/22)+","+str(dnn_precisions/22)+","+str(dnn_recalls/22)+","+str(dnn_f1s/22)+","+
#             str(svm_accuracies/22)+","+str(svm_precisions/22)+","+str(svm_recalls/22)+","+str(svm_f1s/22))
# #     f.write(str(svm_accuracies/22)+","+str(svm_precisions/22)+","+str(svm_recalls/22)+","+str(svm_f1s/22))
#      
#     f.close()
#      
#     s_knn = UnivariateSpline(total_fpr_knn, total_tpr_knn, k=2)
#     xs = np.linspace(0.0, 1.0, 200)
#     ys = s_knn(xs)
#       
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.matshow(total_confmat_knn, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(total_confmat_knn.shape[0]):
        for j in range(total_confmat_knn.shape[1]):
            ax.text(x=j, y=i, s=total_confmat_knn[i,j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.savefig("D:\\Documents\\KNN3\\LL\\total_confmat.png")
    plt.close()
    
    total_fpr_knn.sort()
    total_tpr_knn.sort()
    roc_auc_knn = auc(total_fpr_knn, total_tpr_knn)
    plt.title('KNN LL ROC Curve')
    plt.plot(total_fpr_knn, total_tpr_knn, 'b', label='AUC = %.2F' % roc_auc_knn)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("D:\\Documents\\KNN3\\LL\\total_roc.png")
    plt.close()
# #     
#     s_dnn = UnivariateSpline(total_fpr_dnn, total_tpr_dnn, k=2)
#     xs = np.linspace(0.0, 1.0)
#     ys = s_knn(xs)
#       
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.matshow(total_confmat_dnn, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(total_confmat_dnn.shape[0]):
        for j in range(total_confmat_dnn.shape[1]):
            ax.text(x=j, y=i, s=total_confmat_dnn[i,j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.savefig("D:\\Documents\\DNN3\\LL\\total_confmat.png")
    plt.close()
#       
    total_fpr_dnn.sort()
    total_tpr_dnn.sort()
    print(total_fpr_dnn)
    print(total_tpr_dnn)
    roc_auc_dnn = auc(total_fpr_dnn, total_tpr_dnn)
    plt.title('DNN ROC Curve')
    plt.plot(total_fpr_dnn, total_tpr_dnn, 'b', label='AUC = %.2F' % roc_auc_dnn)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("D:\\Documents\\DNN3\\LL\\total_roc.png")
    plt.close()
#      
#     s_svm = UnivariateSpline(total_fpr_svm, total_tpr_svm, k=2)
#     xs = np.linspace(0.0, 1.0, 200)
#     ys = s_svm(xs)
#      
    fig, ax = plt.subplots(figsize=(3.0, 3.0))
    ax.matshow(total_confmat_svm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(total_confmat_svm.shape[0]):
        for j in range(total_confmat_svm.shape[1]):
            ax.text(x=j, y=i, s=total_confmat_svm[i,j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.savefig("D:\\Documents\\SVM3\\LL\\total_confmat.png")
    plt.close()
#      
    total_fpr_svm.sort()
    total_tpr_svm.sort()
    roc_auc_svm = auc(total_fpr_svm, total_tpr_svm)
    plt.title('SVM LL ROC Curve')
    plt.plot(total_fpr_svm, total_tpr_svm, 'b', label='AUC = %.2F' % roc_auc_svm)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("D:\\Documents\\SVM3\\LL\\total_roc.png")
    plt.close()
