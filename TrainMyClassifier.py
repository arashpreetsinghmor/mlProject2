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
    m = len(target)
    n = len(target[0])
    y = []
    for i in xrange(m):
        for j in xrange(n):
            if target[i][j] == 1:
                y.append(j+1)

    return y

def SVMTraining(XEstimate,XValidate,Parameters,class_labels):
    #clf = svm.SVC(decision_function_shape='ovo',Parameters)
    svcClassifier = SVC(kernel='rbf',  probability=True)
    gridSearcher = GridSearchCV(svcClassifier, Parameters)
    clf = OneVsRestClassifier(gridSearcher)
    #clf = OneVsRestClassifier(GridSearchCV(SVC(kernel='rbf',  probability=True), Parameters))
    print(clf.get_params)
    clf.fit(XEstimate, class_labels)
    Yvalidate=clf.predict(XValidate)
    EstParameters=clf.get_params()
    #print(clf.predict_proba(XValidate))
    mini = 1
    for i in clf.predict_proba(XValidate):
        mini = min(max(i), mini)
    #print(mini)
    print(clf.get_params)
    #print(svcClassifier.__dict__)
    #print(clf.d(XValidate))
    return {"Yvalidate": Yvalidate,
            "EstParameters": EstParameters,
            "clf": clf}


def RVMTraining(XEstimate,XValidate,Parameters,class_labels):
    clf = OneVsOneClassifier(GridSearchCV(RVC(kernel='rbf', n_iter=1), Parameters))
    print(clf.get_params)
    print("Haha")
    clf.fit(XEstimate, class_labels)
    print(clf.get_params)
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

    print(clf.fit(XEstimate, class_labels))
    Yvalidate = clf.predict(XValidate)
    EstParameters = clf.get_params()
    print(clf.predict_proba(XValidate))
    return {"Yvalidate": Yvalidate,
            "EstParameters": EstParameters,
            "clf": clf}




if __name__ == '__main__':
    x=loadmat("Proj2FeatVecsSet1.mat")["Proj2FeatVecsSet1"]
    y=PreProcessing("Proj2TargetOutputsSet1.mat")
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
    #for i in range(10):
    #    testData.append(100)
    for i in l:
        xe.append(reduced_XEstimate[i])
        yy.append(y[i])
        testData.append(reduced_XEstimate[i-1])
        testLabels.append(y[i-1])

    #should load the test data here
    #RVMTraining(reduced_XEstimate[10000:19000], reduced_XEstimate[100:200], Params['RVM'],y[10000:19000])
    #GPRTraining(reduced_XEstimate, reduced_XEstimate, Params['GPR'],y)
    #print(xe[0])
    res=RVMTraining(xe, testData, Params['RVM'],yy)
    print(res["Yvalidate"])
    right = 0
    wrong = 0
    for i in range(len(res["Yvalidate"])):
        if res["Yvalidate"][i] == testLabels[i]:
            right+=1.0

    print(right/len(res["Yvalidate"]))
    testData = []
    for i in range(20):
        testData.append(i)

    kf = KFold(n_splits=5)



    for train, test in kf.split(testData):
        print("%s %s" % (train, test))

    l = [[1,3],2,3]
    y = [[4,3],4,5]

    for c in zip(l,y):
        print(c)


    #SVMTraining(reduced_XEstimate, reduced_XEstimate, Params['SVM'],y)

