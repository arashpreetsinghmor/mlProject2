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
from math import exp

global Parameters
global numClasses
global clf
global tuningSize
import sys

doPrint = 1
Params = {
    'SVM': {
        'gamma': [1]
    },
    'RVM': {
        'alpha': [1e-06],
        'beta': [1e-06]
    },
    'GPR': {
        'length_scale': [10]
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


def SVMTraining(XEstimate, XValidate, Parameters, class_labels):
    svcClassifier = SVC(kernel='rbf', probability=True)
    gridSearcher = GridSearchCV(svcClassifier, Parameters)
    clf = OneVsRestClassifier(gridSearcher)

    clf.fit(XEstimate, class_labels)
    Yvalidate = clf.predict(XValidate)

    EstParameters = clf.get_params()

    mini = 1
    for i in clf.predict_proba(XValidate):
        mini = min(max(i), mini)
    print(mini)
    return {"Yvalidate": Yvalidate,
            "EstParameters": EstParameters,
            "clf": clf}


def RVMTraining(XEstimate, XValidate, Parameters, class_labels):
    clf = OneVsOneClassifier(GridSearchCV(RVC(kernel='rbf', n_iter=1), Parameters))
    clf.fit(XEstimate, class_labels)
    Yvalidate = clf.predict(XValidate)
    EstParameters = clf.get_params()
    return {"Yvalidate": Yvalidate,
            "EstParameters": EstParameters,
            "clf": clf}


def GPRTraining(XEstimate, XValidate, Parameters, class_labels):
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0))
    # clf = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=1)
    clf = GaussianProcessClassifier(kernel=RBF(length_scale=1.0), optimizer=None,
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

    ConfMatrix = np.zeros([5, 5], dtype=float)
    EstConfMatrices = []
    for train_index, test_index in kf.split(XTrain):

        X_train, X_test = XTrain[train_index], XTrain[test_index]
        y_train, y_test = ClassLabels[train_index], ClassLabels[test_index]

        TuneMyClassifier(Parameters, X_train, X_test, y_train, y_test)

        if Parameters == "SVM":
            res = SVMTraining(X_train, X_test, Params[Parameters], y_train)
        elif Parameters == "RVM":
            res = RVMTraining(X_train, X_test, Params[Parameters], y_train)
        elif Parameters == "GPR":
            res = GPRTraining(X_train, X_test, Params[Parameters], y_train)

        l = MyConfusionMatrix(res["Yvalidate"], " ", y_test)
        EstConfMatrices.append(l[0])
        ConfMatrix += l[0]
    print(ConfMatrix/Nf)

#Tunes the parameters of the classifier for a small data set and sets them to be used by cross validation
def TuneMyClassifier(Parameters, X_train, X_test, y_train, y_test):

    global doPrint
    X_train_subset, X_test_subset = X_train[0:50], X_test[0:50]
    y_train_subset, y_test_subset = y_train[0:50], y_test[0:50]
    doPrint = 0

    if Parameters == "SVM":
        best_gamma, max_accuracy = 1, 0
        for i in range(-10, 10):
            gamma = 2 ** i
            Params[Parameters]['gamma'] = [gamma]
            temp = SVMTraining(X_train_subset, X_test_subset, Params[Parameters], y_train_subset)
            temp_confusion_matrix = MyConfusionMatrix(temp["Yvalidate"], " ", y_test_subset)
            if max_accuracy < np.mean(np.diagonal(temp_confusion_matrix[0])):
                best_gamma = gamma
                max_accuracy = np.mean(np.diagonal(temp_confusion_matrix[0]))
            #print('gamma: ', gamma)
        #print('best gamma: ', best_gamma)
        Params[Parameters]['gamma'] = [best_gamma]

    elif Parameters == "RVM":
        best_alpha, best_beta, max_accuracy = 0, 0, 0
        for i in range(-7, 3):
            for j in range(-7,3):
                alpha, beta = exp(i), exp(j)
                Params[Parameters]['alpha'] = [alpha]
                Params[Parameters]['beta'] = [beta]
                temp = RVMTraining(X_train_subset, X_test_subset, Params[Parameters], y_train_subset)
                temp_confusion_matrix = MyConfusionMatrix(temp["Yvalidate"], " ", y_test_subset)
                if max_accuracy < np.mean(np.diagonal(temp_confusion_matrix[0])):
                    best_alpha = alpha
                    best_beta = beta
                    max_accuracy = np.mean(np.diagonal(temp_confusion_matrix[0]))

        #print('best aplha: ', best_alpha)
        #print('best beta: ', best_beta)
        Params[Parameters]['alpha'] = [best_alpha]
        Params[Parameters]['beta'] = [best_beta]

    elif Parameters == "GPR":
        best_length_scale = 0
        max_accuracy = 0
        for i in range(-2, 12):
            Params[Parameters]['length_scale'] = [i]
            temp = GPRTraining(X_train_subset, X_test_subset, Params[Parameters], y_train_subset)
            temp_confusion_matrix = MyConfusionMatrix(temp["Yvalidate"], " ", y_test_subset)
            print(np.mean(np.diagonal(temp_confusion_matrix[0])))
            if max_accuracy < np.mean(np.diagonal(temp_confusion_matrix[0])):
                best_length_scale = i
                max_accuracy = np.mean(np.diagonal(temp_confusion_matrix[0]))

        Params[Parameters]['length_scale'] = [best_length_scale]
        #print('best length scale: ', best_length_scale)
    doPrint = 1

def MyConfusionMatrix(predictedLabels, ClassNames, actualLabels):
    matrix = np.zeros([5, 5], dtype=int)
    correctHits = 0
    for i in range(len(actualLabels)):
        matrix[actualLabels[i] - 1][predictedLabels[i] - 1] += 1
        if actualLabels[i] == predictedLabels[i]:
            correctHits += 1

    confusion_matrix = matrix[:][:]

    confusion_matrix2 = matrix / matrix.sum(axis=1)[:, None]

    if doPrint==1:
        print(confusion_matrix2)

    return (confusion_matrix2, correctHits / len(actualLabels))


def TestMyClassifier(XTest, Parameters, EstParameters):
    Ytest = EstParameters.predict(XTest)
    return Ytest


def TrainMyClassifier(XEstimate, XValidate, Parameters, class_labels):
    if Parameters == "SVM":
        return SVMTraining(XEstimate, XValidate, Params[Parameters], class_labels)
    elif Parameters == "RVM":
        return RVMTraining(XEstimate, XValidate, Params[Parameters], class_labels)
    elif Parameters == "GPR":
        return GPRTraining(XEstimate, XValidate, Params[Parameters], class_labels)
    else:
        print("invalid input")
        sys.exit()


if __name__ == '__main__':
    x = loadmat("Proj2FeatVecsSet1.mat")["Proj2FeatVecsSet1"]
    y = PreProcessing("Proj2TargetOutputsSet1.mat")
    #print(x[5001])
    c = list(zip(x, y))
    shuffle(c)
    x, y = zip(*c)
    y = np.asarray(y)
    pca = PCA(n_components=10)
    reduced_XEstimate = pca.fit_transform(x)
    l = []
    t = []
    for x in range(1, 25000, 10):
        l.append(x)
    xe = []
    yy = []
    testData = []
    testLabels = []

    for i in l:
        xe.append(reduced_XEstimate[i])
        yy.append(y[i])
        testData.append(reduced_XEstimate[i - 1])
        testLabels.append(y[i - 1])

    print("Input the classifier you want to train (RVM / SVM / GPR) :")
    Parameters = input().upper()
    res = TrainMyClassifier(xe, testData, Parameters, yy)
    MyCrossValidate(xe, 5, yy)
    a = [[0.05989583, 0.03385417, 0.05208333, 0.06770833, 0.10677083 ,0.06770833,
 0.078125, 0.109375, 0.05729167, 0.04427083, 0.03385417, 0.046875,
 0.10677083, 0.0625,   0.0546875,0.0625  , 0.09895833, 0.05989583,
 0.0546875,0.06770833, 0.1484375,0.09635417 ,0.3046875,0.1484375,
 0.30208333, 0.23177083, 0.60416667, 0.29427083, 0.2265625,0.20052083,
 0.5390625,0.23177083, 0.22395833, 0.17447917, 0.42708333, 0.16666667,
 0.15364583, 0.1171875,0.1953125,0.09114583, 0.08854167 ,0.0625,
 0.25260417, 0.08072917, 0.1953125,0.1640625,0.52604167 ,0.18489583,
 0.16927083, 0.15625,  0.50520833, 0.18489583, 0.1171875,0.11197917,
 0.37239583, 0.10416667, 0.0546875,0.05729167, 0.140625, 0.0234375 ]];
print(TestMyClassifier(pca.transform(a), Parameters, res["clf"]))
