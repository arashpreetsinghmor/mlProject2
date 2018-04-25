from sklearn.model_selection import KFold



def MyCrossValidate(XTrain, Nf = 5, ClassLabels):
	appended = list(zip(XTrain, ClassLabels))


	kf = KFold(n_splits=Nf)

	ConfMatrix = np.zeros([len(unique_number_of_class_labels), len(unique_number_of_class_labels)], dtype=int)
	EstConfMatrices = []
    for train, test in kf.split(appended):
        res = TrainMyClassifier(train[:][0:len(train[0])-1], train[:][-1], 'SVM', test[:][0:len(train[0])-1])
        l = ComputeConfusionMatrix(res["Yvalidate"], " ",test[:][-1])
        EstConfMatrices.append(l[0])
        ConfMatrix += l[0]
