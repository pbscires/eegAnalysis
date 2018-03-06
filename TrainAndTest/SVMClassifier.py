'''
Created on Jan 14, 2018

@author: pb8xe
'''
from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from xlrd.timemachine import fprintf
from sklearn.metrics.classification import f1_score
class SVMClassifier(object):
    '''
    classdocs
    '''


    def __init__(self, X_train, y_train, X_test, y_test):
        '''
        Constructor
        '''
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.classifier = SVC(random_state=0, gamma=.1, C=0.1)
    
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
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)
    
    def train(self):
        self.classifier.fit(self.X_train, self.y_train.ravel())
    
    def test(self, f, patient_num, total_confmat, total_fpr, total_tpr, total_auc):
        y_pred = self.classifier.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        print("Accuracy: %.2f" % accuracy)
        print("Precision: %.2f" % precision)
        print("Recall: %.2f" % recall)
        print("F1: %.2f" % f1)
        line = str(accuracy)+","+str(precision)+","+str(recall)+","+str(f1)
        f.write(line)
        f.write("\n")
        confmat = confusion_matrix(self.y_test, y_pred)
        for i in range(0,2):
            for j in range(0,2):
                total_confmat[i,j]+=confmat[i,j]
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(confmat.shape[0]):
            for j in range(confmat.shape[1]):
                ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
        plt.xlabel('predicted label')
        plt.ylabel('true label')
        plt.savefig("D:\\Documents\\SVM3\\LL\\chb"+patient_num+"_confmat.png")
        plt.close()
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred)
        for i in range(len(fpr)):
            if fpr[i]*1==0:
                fpr[i]=0.0
            elif fpr[i]*1==1:
                fpr[i]=1.0
        for i in range(len(tpr)):
            if tpr[i]*1==0:
                tpr[i]=0.0
            elif tpr[i]*1==1:
                tpr[i]=1.0
        print("fpr", fpr)
        print("tpr", tpr)
        for coor in fpr:
            if (coor!=0.0) and (coor!=1.0):
                total_fpr.append(coor)
            else:
                total_fpr.append(0.5)
        for coor in tpr:
            if (coor!=0.0) and (coor!=1.0):
                total_tpr.append(coor)
            else:
                total_tpr.append(0.5)
        
        total_auc.append(roc_auc_score(self.y_test, y_pred))
#         plt.title('ROC Curve')
#         plt.plot(fpr, tpr, 'b', label='AUC = %.2F' % roc_auc)
#         plt.legend(loc='lower right')
#         plt.plot([0,1], [0,1], 'r--')
#         plt.xlim([-0.1, 1.2])
#         plt.ylim([-0.1, 1.2])
#         plt.ylabel('True Positive Rate')
#         plt.xlabel('False Positive Rate')
#         plt.savefig("D:\\Documents\\KNN\\FFT\\chb"+patient_num+"roc.png")
#         plt.close()
        return accuracy, precision, recall, f1, total_confmat, total_fpr, total_tpr, total_auc
        
    def validation(self):
        self.create_arrays()
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        train_scores,test_scores = validation_curve(estimator=self.classifier, X=self.X_train, y=self.y_train.ravel(), param_name='gamma', param_range=param_range, cv=3)
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
        plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
        plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        plt.grid()
        plt.xscale('log')
        plt.legend(loc='lower right')
        plt.xlabel('Parameter C')
        plt.ylabel('Accuracy')
        plt.ylim([0.8, 1.0])
        plt.show()
        
    def grid_search(self):
        self.create_arrays()
        pipe_svc = Pipeline([('scl', StandardScaler()), ('clf', SVC(random_state=1))])
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [{'clf__C' : param_range, 'clf__gamma': param_range, 'clf__kernel': ['rbf']}]
        gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=5, n_jobs=-1)
        gs = gs.fit(self.X_train, self.y_train.ravel())
        print(gs.best_score_)
        print(gs.best_params_)