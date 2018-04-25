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
from random import shuffle, random, randint

from sklearn.svm import OneClassSVM
global Parameters
global numClasses
global clf
global tuningSize
import sys
from collections import Counter

#parameter used to print confusion matrix only during cross validation, not parameter tuning
doPrint = 1

#dictionary for hyper parameters
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

# inferring class labels from the target outputs
def PreProcessing(file_name):
    target = loadmat(file_name)["Proj2TargetOutputsSet1"]
    m = len(target)
    n = len(target[0])
    y = []
    for i in range(m):
        for j in range(n):
            if target[i][j] == 1:
                y.append(j + 1)

    y = np.asarray(y)

    return y

# Classification via SVM
def SVMTraining(XEstimate, XValidate, Parameters, class_labels):
    svcClassifier = SVC(kernel='rbf', probability=True)
    gridSearcher = GridSearchCV(svcClassifier, Parameters)
    clf = OneVsRestClassifier(gridSearcher)

    clf.fit(XEstimate, class_labels)
    Yvalidate = clf.predict(XValidate)

    EstParameters = clf.get_params()

    return {"Yvalidate": Yvalidate,
            "EstParameters": EstParameters,
            "clf": clf}

# Classification via RVM
def RVMTraining(XEstimate, XValidate, Parameters, class_labels):
    clf = OneVsOneClassifier(GridSearchCV(RVC(kernel='rbf', n_iter=1), Parameters))
    clf.fit(XEstimate, class_labels)
    Yvalidate = clf.predict(XValidate)
    EstParameters = clf.get_params()
    return {"Yvalidate": Yvalidate,
            "EstParameters": EstParameters,
            "clf": clf}

# Classification via GPR
def GPRTraining(XEstimate, XValidate, Parameters, class_labels):
    kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0))
    # clf = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=1)
    clf = GaussianProcessClassifier(kernel=RBF(length_scale=Parameters['length_scale']), optimizer=None,
                                    multi_class='one_vs_one', n_jobs=1)

    clf.fit(XEstimate, class_labels)
    Yvalidate = clf.predict(XValidate)
    EstParameters = clf.get_params()

    return {"Yvalidate": Yvalidate,
            "EstParameters": EstParameters,
            "clf": clf}

# cross validation with provided training set
def MyCrossValidate(XTrain, Nf, ClassLabels):
    XTrain = np.asarray(XTrain)
    ClassLabels = np.asarray(ClassLabels)
    
    kf = KFold(n_splits=Nf)

    ConfMatrix = np.zeros([5, 5], dtype=float)
    EstConfMatrices = []
    for train_index, test_index in kf.split(XTrain):

        X_train, X_test = XTrain[train_index], XTrain[test_index]
        y_train, y_test = ClassLabels[train_index], ClassLabels[test_index]
        
        #tune hyper parameters on small subset of estimation set
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
        print('accuracy: ' + str(l[1]))
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
        for i in range(-10, 2):
            for j in range(-10,2):
                alpha, beta = 10**i, 10**j
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
        i = 0.1
        while i <= 2:
            Params[Parameters]['length_scale'] = [i]
            temp = GPRTraining(X_train_subset, X_test_subset, Params[Parameters], y_train_subset)

            temp_confusion_matrix = MyConfusionMatrix(temp["Yvalidate"], " ", y_test_subset)
            #print(np.mean(np.diagonal(temp_confusion_matrix[0])))
            if max_accuracy < np.mean(np.diagonal(temp_confusion_matrix[0])):
                best_length_scale = i
                max_accuracy = np.mean(np.diagonal(temp_confusion_matrix[0]))
            i += 0.1

        Params[Parameters]['length_scale'] = [best_length_scale]
        #print('best length scale: ', best_length_scale)
    doPrint = 1

# Compute confusion matrix using actual labels and predicted labels
def MyConfusionMatrix(predictedLabels, ClassNames, actualLabels):
    matrix = np.zeros([np.unique(actualLabels).size, np.unique(actualLabels).size], dtype=int)
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
    Ytest = EstParameters["clf"].predict(XTest)
    outliers = EstParameters["outlierDetector"].predict(XTest)
    for i in range(len(outliers)):
        if outliers[i] < 0:
            Ytest[i] = 6
    return Ytest

# compute outliers if any
def outlierDetection(Xtrain, Xtest):
    outlierDetector = OneClassSVM(kernel='rbf', gamma = 0.1, nu = 0.001)
    outlierDetector.fit(Xtrain)
    return outlierDetector.predict(Xtest)


def TrainMyClassifier(XEstimate, XValidate, Parameters, class_labels):
    if Parameters == "SVM":
        res = SVMTraining(XEstimate, XValidate, Params[Parameters], class_labels)
    elif Parameters == "RVM":
        res = RVMTraining(XEstimate, XValidate, Params[Parameters], class_labels)
    elif Parameters == "GPR":
        res = GPRTraining(XEstimate, XValidate, Params[Parameters], class_labels)
    else:
        print("invalid input")
        sys.exit()
    outlierDetector = OneClassSVM(kernel='rbf', gamma = 0.1, nu = 0.001)
    outlierDetector.fit(XEstimate)
    res["outlierDetector"] = outlierDetector
    return res


if __name__ == '__main__':
    x = loadmat("Proj2FeatVecsSet1.mat")["Proj2FeatVecsSet1"]
    y = PreProcessing("Proj2TargetOutputsSet1.mat")
    c = list(zip(x, y))
    shuffle(c)
    x, y = zip(*c)
    y = np.asarray(y)
    pca = PCA(n_components=30)
    reduced_XEstimate = pca.fit_transform(x)
    l = []
    t = []

    for x in range(4, 25000, 10):
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
        testData.append(reduced_XEstimate[i - 2])
        testLabels.append(y[i - 2])
        testData.append(reduced_XEstimate[i - 3])
        testLabels.append(y[i - 3])
        testData.append(reduced_XEstimate[i - 4])
        testLabels.append(y[i - 4])

    print("Input the classifier you want to train (RVM / SVM / GPR) :")
    Parameters = input().upper()
    res = TrainMyClassifier(xe, testData, Parameters, yy)
    MyCrossValidate(xe, 5, yy)
    z= [[random() for _ in range(30)] for _ in range(200)]
    for c in z:
        testData.append(c)
        testLabels.append(6)

    l = MyConfusionMatrix(TestMyClassifier(testData, Parameters, res)," ",testLabels)
    np.savetxt(Parameters + "ConfusionMatrix.csv", l[0],delimiter = ",")
    np.savetxt(Parameters + "Accuracy.csv", (l[1],),delimiter = ",")
