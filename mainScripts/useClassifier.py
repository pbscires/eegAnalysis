'''
Created on Dec 26, 2017

@author: pb8xe
'''

from TrainAndTest.KNNClassifier import KNNClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing.data import StandardScaler
import sys
import numpy as np
from TrainAndTest.DNNClassifier import DNNClassifier
from TrainAndTest.SVMClassifier import SVMClassifier
from sklearn.metrics import auc
import matplotlib.pyplot as plt
if __name__ == '__main__':
    root="D:\\Documents\\csv_combined_ll\chb"
    f = open('D:\\Documents\\multiple\\resultsLL.csv', 'w')
    total_fpr_knn=[0.0, 0.0, 1.0]
    total_tpr_knn=[0.0, 0.0, 1.0]
    knn_accuracies=[]
    knn_precisions=[]
    knn_recalls=[]
    knn_f1s = []
    
    total_fpr_svm=[0.0, 0.0, 1.0]
    total_tpr_svm=[0.0, 0.0, 1.0]
    svm_accuracies=[]
    svm_precisions=[]
    svm_recalls=[]
    svm_f1s = []
    
    total_fpr_dnn=[0.0, 0.0, 1.0]
    total_tpr_dnn=[0.0, 0.0, 1.0]
    dnn_accuracies=[]
    dnn_precisions=[]
    dnn_recalls=[]
    dnn_f1s = []
    
    for index in range(0,23):
        if(index==16):
            continue
        num = index+1
        if num<10:
            num_string="0" + str(num)
        else:
            num_string=str(num)
        filepath = root + num_string + ".csv"
        
        arr = np.genfromtxt(filepath, delimiter=',')
        X = np.delete(arr, [arr.shape[1]-1], axis=1)
        y = np.delete(arr, list(range(arr.shape[1]-1)), axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)
        
        print("Currently on chb"+num_string)
         
        knn_classifier = KNNClassifier(X_train_std, y_train, X_test_std, y_test, k=5)
        knn_classifier.train()
        knn_accuracy, knn_precision, knn_recall, knn_f1 = knn_classifier.test(f, num_string)
        knn_accuracies.append(knn_accuracy)
        knn_precisions.append(knn_precision)
        knn_recalls.append(knn_recall)
        knn_f1s.append(knn_f1)
      
        svm_classifier = SVMClassifier(X_train_std, y_train, X_test_std, y_test)
        svm_classifier.train()
        svm_accuracy, svm_precision, svm_recall, svm_f1 = svm_classifier.test(f, num_string)
        svm_accuracies.append(svm_accuracy)
        svm_precisions.append(svm_precision)
        svm_recalls.append(svm_recall)
        svm_f1s.append(svm_f1)
          
        dnn_classifier = DNNClassifier(X_train_std, y_train, X_test_std, y_test)
        dnn_classifier.train()
        dnn_accuracy, dnn_precision, dnn_recall, dnn_f1 = dnn_classifier.test(f, num_string)
        dnn_accuracies.append(dnn_accuracy)
        dnn_precisions.append(dnn_precision)
        dnn_recalls.append(dnn_recall)
        dnn_f1s.append(dnn_f1)
    
    f.close()
    
    plt.title('KNN Avg Precision-Recall Curve')
    plt.plot(knn_precisions, knn_recalls, 'b')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig("D:\\Documents\\KNN3\\LL\\avgprc.png")
    plt.close()
    
#     total_fpr_dnn[1]/=22
#     total_fpr_knn[1]/=22
#     total_fpr_svm[1]/=22
#     total_tpr_dnn[1]/=22
#     total_tpr_knn[1]/=22
#     total_tpr_svm[1]/=22
#     
#     roc_auc_knn = auc(total_fpr_knn, total_tpr_knn)
#     plt.title('KNN Avg ROC Curve')
#     plt.plot(total_fpr_knn, total_tpr_knn, 'b', label='AUC = %.2F' % roc_auc_knn)
#     plt.legend(loc='lower right')
#     plt.plot([0,1], [0,1], 'r--')
#     plt.xlim([-0.1, 1.2])
#     plt.ylim([-0.1, 1.2])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.savefig("D:\\Documents\\KNN\\FFT\\avgroc.png")
#     plt.close()
#     
#     roc_auc_svm = auc(total_fpr_svm, total_tpr_svm)
#     plt.title('SVM Avg ROC Curve')
#     plt.plot(total_fpr_svm, total_tpr_svm, 'b', label='AUC = %.2F' % roc_auc_svm)
#     plt.legend(loc='lower right')
#     plt.plot([0,1], [0,1], 'r--')
#     plt.xlim([-0.1, 1.2])
#     plt.ylim([-0.1, 1.2])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.savefig("D:\\Documents\\SVM\\FFT\\avgroc.png")
#     plt.close()
#     
#     roc_auc_dnn = auc(total_fpr_dnn, total_tpr_dnn)
#     plt.title('DNN Avg ROC Curve')
#     plt.plot(total_fpr_dnn, total_tpr_dnn, 'b', label='AUC = %.2F' % roc_auc_dnn)
#     plt.legend(loc='lower right')
#     plt.plot([0,1], [0,1], 'r--')
#     plt.xlim([-0.1, 1.2])
#     plt.ylim([-0.1, 1.2])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.savefig("D:\\Documents\\DNN\\FFT\\avgroc.png")
#     plt.close()
