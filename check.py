from sklearn import svm
import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
from RVM import RVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import KFold
from random import shuffle
global Parameters
global numClasses
global clf
import sys
Params={
    'SVM':{
        'gamma' : [1]
    },
    'RVM':{
        'alpha':[1e-06],
        'beta' : [1e-06]
    },
    'GPR':{
        'length_scale' : [10]
    }
}

def PreProcessing(file_name):

    target = loadmat(file_name)["Proj2TargetOutputsSet1"]
    number_labels = []
    for ars in target:
        if np.all(ars == [1, -1, -1, -1, -1]):
            ars = 1
            number_labels.append(ars)
        elif np.all(ars == [-1, 1, -1, -1, -1]):
            ars = 2
            number_labels.append(ars)
        elif np.all(ars == [-1, -1, 1, -1, -1]):
            ars = 3
            number_labels.append(ars)
        elif np.all(ars == [-1, -1, -1, 1, -1]):
            ars = 4
            number_labels.append(ars)
        elif np.all(ars == [-1, -1, -1, -1, 1]):
            ars = 5
            number_labels.append(ars)

    return np.asarray(number_labels)

def SVMTraining(XEstimate,XValidate,Parameters,class_labels):

    svcClassifier = SVC(kernel='rbf',  probability=True)
    gridSearcher = GridSearchCV(svcClassifier, Parameters)
    clf = OneVsRestClassifier(gridSearcher)

    clf.fit(XEstimate, class_labels)
    Yvalidate=clf.predict(XValidate)
    
    EstParameters=clf.get_params()

    mini = 1
    for i in clf.predict_proba(XValidate):
        mini = min(max(i), mini)
    print(mini)
    return {"Yvalidate": Yvalidate,
            "EstParameters": EstParameters,
            "clf": clf}


def RVMTraining(XEstimate,XValidate,Parameters,class_labels):
    clf = OneVsOneClassifier(GridSearchCV(RVC(kernel='rbf', n_iter=1), Parameters))
    clf.fit(XEstimate, class_labels)
    Yvalidate = clf.predict(XValidate)
    EstParameters = clf.get_params()
    return {"Yvalidate": Yvalidate,
            "EstParameters": EstParameters,
            "clf": clf}


def GPRTraining(XEstimate,XValidate,Parameters,class_labels):
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0))
    #clf = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=1)
    clf = GaussianProcessClassifier(kernel= RBF(length_scale=1.0), optimizer=None,
                                       multi_class='one_vs_one', n_jobs=1)

    clf.fit(XEstimate, class_labels)
    Yvalidate = clf.predict(XValidate)
    EstParameters = clf.get_params()
    
    return {"Yvalidate": Yvalidate,
            "EstParameters": EstParameters,
            "clf": clf}



def MyCrossValidate(XTrain, Nf, ClassLabels):
    XTrain = np.asarray(XTrain)
    ClassLabels = np.asarray(ClassLabels)
    kf = KFold(n_splits=Nf)

    ConfMatrix = np.zeros([5, 5], dtype=int)
    EstConfMatrices = []
    for train_index, test_index in kf.split(XTrain):
        X_train, X_test = XTrain[train_index], XTrain[test_index]
        y_train, y_test = ClassLabels[train_index], ClassLabels[test_index]

        if Parameters == "SVM":
            res = SVMTraining(X_train, X_test, Params[Parameters], y_train)
        elif Parameters == "RVM":
            res = RVMTraining(X_train, X_test, Params[Parameters], y_train)
        elif Parameters == "GPR":
            res = GPRTraining(X_train, X_test, Params[Parameters], y_train)
        
        l = MyConfusionMatrix(res["Yvalidate"], " ",y_test)
        EstConfMatrices.append(l[0])
        ConfMatrix += l[0]
    print(ConfMatrix)


def MyConfusionMatrix(predictedLabels, ClassNames, actualLabels):

    matrix = np.zeros([5,5], dtype=int)
    correctHits = 0
    for i in range(len(actualLabels)):
        matrix[actualLabels[i]-1][predictedLabels[i]-1] += 1
        if actualLabels[i] == predictedLabels[i]:
            correctHits += 1
    
    confusion_matrix = matrix[:][:]

    confusion_matrix2 = matrix / matrix.sum(axis=1)[:, None]

    print(confusion_matrix2)
    
    return (confusion_matrix, correctHits/len(actualLabels))


def TestMyClassifier(XTest, Parameters, EstParameters):
    Ytest = EstParameters.predict(XTest)
    return Ytest



def TrainMyClassifier(XEstimate,XValidate,Parameters,class_labels):
    if Parameters == "SVM":
        return SVMTraining(XEstimate,XValidate,Params[Parameters],class_labels)
    elif Parameters == "RVM":
        return RVMTraining(XEstimate,XValidate,Params[Parameters],class_labels)
    elif Parameters == "GPR":
        return GPRTraining(XEstimate,XValidate,Params[Parameters],class_labels)
    else:
        print("invalid input")
        sys.exit()

    

if __name__ == '__main__':
    x=loadmat("Proj2FeatVecsSet1.mat")["Proj2FeatVecsSet1"]
    y=PreProcessing("Proj2TargetOutputsSet1.mat")
    c = list(zip(x, y))
    shuffle(c)
    x, y = zip(*c)
    y = np.asarray(y)
    pca = PCA(n_components=10)
    reduced_XEstimate = pca.fit_transform(x)
    l=[]
    t= []
    for x in range(1,25000,10):
        l.append(x)
    xe=[]
    yy=[]
    testData = []
    testLabels = []

    for i in l:
        xe.append(reduced_XEstimate[i])
        yy.append(y[i])
        testData.append(reduced_XEstimate[i-1])
        testLabels.append(y[i-1])

    print("Input the classifier you want to train (RVM / SVM / GPR) :")
    Parameters =  input().upper()
    res = TrainMyClassifier(xe, testData, Parameters ,yy)
    MyCrossValidate(xe, 5, yy)
    a = [[10]*60]
    print(TestMyClassifier(pca.transform(a),Parameters,res["clf"]))
