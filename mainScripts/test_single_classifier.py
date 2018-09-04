'''
Created on Dec 26, 2017

@author: pb8xe
'''

from TrainAndTest.KNNClassifier import KNNClassifier
import sys
from TrainAndTest.DNNClassifier import DNNClassifier
from TrainAndTest.SVMClassifier import SVMClassifier
from TrainAndTest.RNNClassifier import RNNClassifier
from sklearn.metrics import auc
import matplotlib.pyplot as plt
if __name__ == '__main__':
    root="D:\\Documents\\ReadyForTensorFlow\\FFT\\chb"
    f = open('D:\\Documents\\resultsFFTtest.csv', 'w')
    total_fpr_knn=[0.0, 0.0, 1.0]
    total_tpr_knn=[0.0, 0.0, 1.0]
    
    total_fpr_svm=[0.0, 0.0, 1.0]
    total_tpr_svm=[0.0, 0.0, 1.0]
    
    total_fpr_dnn=[0.0, 0.0, 1.0]
    total_tpr_dnn=[0.0, 0.0, 1.0]

    total_fpr_rnn=[0.0, 0.0, 1.0]
    total_tpr_rnn=[0.0, 0.0, 1.0]
    
    for index in range(3,4):
        num = index+1
        if num<10:
            num_string="0" + str(num)
        else:
            num_string=str(num)
        train_path = root + num_string + ".Xy_train.csv"
        test_path = root + num_string + ".Xy_test.csv"
        
        print("Currently on chb"+num_string)
         
        knn_classifier = KNNClassifier(train_path, test_path, k=5)
        knn_classifier.train()
        total_fpr_knn, total_tpr_knn = knn_classifier.test(f, num_string, total_fpr_knn, total_tpr_knn)
      
        svm_classifier = SVMClassifier(train_path, test_path)
        svm_classifier.train()
        total_fpr_svm, total_tpr_svm = svm_classifier.test(f, num_string, total_fpr_svm, total_tpr_svm)
          
        dnn_classifier = DNNClassifier(train_path, test_path)
        dnn_classifier.train()
        total_fpr_dnn, total_tpr_dnn = dnn_classifier.test(f, num_string, total_fpr_dnn, total_tpr_dnn)

        rnn_classifier = RNNClassifier(train_path, test_path)
        rnn_classifier.train()
        total_fpr_rnn, total_tpr_rnn = rnn_classifier.test(f, num_string, total_fpr_rnn, total_tpr_rnn)
    
    f.close()
    
    total_fpr_dnn[1]/=22
    total_fpr_knn[1]/=22
    total_fpr_svm[1]/=22
    total_fpr_rnn[1]/=22
    total_tpr_dnn[1]/=22
    total_tpr_knn[1]/=22
    total_tpr_svm[1]/=22
    total_tpr_rnn[1]/=22
    
    roc_auc_knn = auc(total_fpr_knn, total_tpr_knn)
    plt.title('KNN Avg ROC Curve')
    plt.plot(total_fpr_knn, total_tpr_knn, 'b', label='AUC = %.2F' % roc_auc_knn)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("D:\\Documents\\KNN\\FFTtest\\avgroc.png")
    plt.close()
    
    roc_auc_svm = auc(total_fpr_svm, total_tpr_svm)
    plt.title('SVM Avg ROC Curve')
    plt.plot(total_fpr_svm, total_tpr_svm, 'b', label='AUC = %.2F' % roc_auc_svm)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("D:\\Documents\\SVM\\FFTtest\\avgroc.png")
    plt.close()
    
    roc_auc_dnn = auc(total_fpr_dnn, total_tpr_dnn)
    plt.title('DNN Avg ROC Curve')
    plt.plot(total_fpr_dnn, total_tpr_dnn, 'b', label='AUC = %.2F' % roc_auc_dnn)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("D:\\Documents\\DNN\\FFTtest\\avgroc.png")
    plt.close()

    roc_auc_rnn = auc(total_fpr_rnn, total_tpr_rnn)
    plt.title('RNN Avg ROC Curve')
    plt.plot(total_fpr_rnn, total_tpr_rnn, 'b', label='AUC = %.2F' % roc_auc_rnn)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("D:\\Documents\\RNN\\FFTtest\\avgroc.png")
    plt.close()