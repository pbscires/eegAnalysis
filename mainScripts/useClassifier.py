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
from sklearn.metrics.ranking import roc_auc_score
from scipy.interpolate.fitpack2 import UnivariateSpline
if __name__ == '__main__':
    root="D:\\Documents\\LL_PreIctal\\chb"
    f = open('D:\\Documents\\multiple\\resultsLL.csv', 'w')
    total_fpr_knn=[]
    total_tpr_knn=[]
    total_auc_knn=[]
    total_confmat_knn=np.zeros([2,2])
    knn_accuracies=[]
    knn_precisions=[]
    knn_recalls=[]
    knn_f1s = []
    
    total_fpr_svm=[]
    total_tpr_svm=[]
    total_auc_svm=[]
    total_confmat_svm=np.zeros([2,2])
    svm_accuracies=[]
    svm_precisions=[]
    svm_recalls=[]
    svm_f1s = []
    
    total_fpr_dnn=[]
    total_tpr_dnn=[]
    total_auc_dnn=[]
    total_confmat_dnn=np.zeros([2,2])
    dnn_accuracies=[]
    dnn_precisions=[]
    dnn_recalls=[]
    dnn_f1s = []
    
    for index in range(0,23):
        num = index+1
        if(num==16):
            continue
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
        knn_accuracy, knn_precision, knn_recall, knn_f1, total_confmat_knn, total_fpr_knn, total_tpr_knn, total_auc_knn = knn_classifier.test(f, num_string, total_confmat_knn, total_fpr_knn, total_tpr_knn, total_auc_knn)
        knn_accuracies+=knn_accuracy
        knn_precisions+=knn_precision
        knn_recalls+=knn_recall
        knn_f1s+=knn_f1
           
        dnn_classifier = DNNClassifier(X_train_std, y_train, X_test_std, y_test)
        dnn_classifier.train()
        dnn_accuracy, dnn_precision, dnn_recall, dnn_f1, total_confmat_dnn, total_fpr_dnn, total_tpr_dnn, total_auc_dnn = dnn_classifier.test(f, num_string, total_confmat_dnn, total_fpr_dnn, total_tpr_dnn, total_auc_dnn)
        dnn_accuracies+=dnn_accuracy
        dnn_precisions+=dnn_precision
        dnn_recalls+=dnn_recall
        dnn_f1s+=dnn_f1
        
        svm_classifier = SVMClassifier(X_train_std, y_train, X_test_std, y_test)
        svm_classifier.train()
        svm_accuracy, svm_precision, svm_recall, svm_f1, total_confmat_svm, total_fpr_svm, total_tpr_svm, total_auc_svm = svm_classifier.test(f, num_string, total_confmat_svm, total_fpr_svm, total_tpr_svm, total_auc_svm)
        svm_accuracies+=svm_accuracy
        svm_precisions+=svm_precision
        svm_recalls+=svm_recall
        svm_f1s+=svm_f1
        
    f.write("\n")
    f.write(str(knn_accuracies/22)+","+str(knn_precisions/22)+","+str(knn_recalls/22)+","+str(knn_f1s/22)+","+
            str(dnn_accuracies/22)+","+str(dnn_precisions/22)+","+str(dnn_recalls/22)+","+str(dnn_f1s/22)+","+
            str(svm_accuracies/22)+","+str(svm_precisions/22)+","+str(svm_recalls/22)+","+str(svm_f1s/22))
#     f.write(str(svm_accuracies/22)+","+str(svm_precisions/22)+","+str(svm_recalls/22)+","+str(svm_f1s/22))
    
    f.close()
    
    s_knn = UnivariateSpline(total_fpr_knn, total_tpr_knn, k=2)
    xs = np.linspace(0.0, 1.0, 200)
    ys = s_knn(xs)
     
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(total_confmat_knn, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(total_confmat_knn.shape[0]):
        for j in range(total_confmat_knn.shape[1]):
            ax.text(x=j, y=i, s=total_confmat_knn[i,j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.savefig("D:\\Documents\\KNN3\\LL\\total_confmat.png")
    plt.close()
     
    roc_auc_knn = auc(xs, ys)
    plt.title('KNN LL ROC Curve')
    plt.plot(total_fpr_knn, total_tpr_knn, 'o')
    plt.plot(xs, ys, 'b', label='AUC = %.2F' % roc_auc_knn)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("D:\\Documents\\KNN3\\LL\\roc.png")
    plt.close()
#     
    s_dnn = UnivariateSpline(total_fpr_dnn, total_tpr_dnn, k=2)
    xs = np.linspace(0.0, 1.0)
    ys = s_knn(xs)
     
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(total_confmat_dnn, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(total_confmat_dnn.shape[0]):
        for j in range(total_confmat_dnn.shape[1]):
            ax.text(x=j, y=i, s=total_confmat_dnn[i,j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.savefig("D:\\Documents\\DNN3\\LL\\total_confmat.png")
    plt.close()
     
    roc_auc_dnn = roc_auc_score(total_fpr_dnn, total_tpr_dnn)
    plt.title('DNN ROC Curve')
    plt.plot(total_tpr_dnn, total_fpr_dnn, 'o')
    plt.plot(xs, ys, 'b', label='AUC = %.2F' % roc_auc_dnn)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("D:\\Documents\\DNN3\\LL\\roc.png")
    plt.close()
    
    s_svm = UnivariateSpline(total_fpr_svm, total_tpr_svm, k=2)
    xs = np.linspace(0.0, 1.0, 200)
    ys = s_svm(xs)
    
    fig, ax = plt.subplots(figsize=(2.5, 2.5))
    ax.matshow(total_confmat_svm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(total_confmat_svm.shape[0]):
        for j in range(total_confmat_svm.shape[1]):
            ax.text(x=j, y=i, s=total_confmat_svm[i,j], va='center', ha='center')
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.savefig("D:\\Documents\\SVM3\\LL\\total_confmat.png")
    plt.close()
    
    roc_auc_svm = auc(xs, ys)
    plt.title('SVM LL ROC Curve')
    plt.plot(total_fpr_svm, total_tpr_svm, 'o')
    plt.plot(xs, ys, 'b', label='AUC = %.2F' % roc_auc_svm)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("D:\\Documents\\SVM3\\LL\\roc.png")
    plt.close()
