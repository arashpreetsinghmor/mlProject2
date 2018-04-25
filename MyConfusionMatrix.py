import numpy as np

def ComputeConfusionMatrix(predictedLabels, ClassNames, actualLabels):
    unique_number_of_class_labels = sorted(set(actualLabels))
    matrix = np.zeros([len(unique_number_of_class_labels), len(unique_number_of_class_labels)], dtype=int)
    correctHits = 0
    for i in range(len(actualLabels)):
        matrix[actualLabels[i]][predictedLabels[i]] += 1
        if actualLabels[i] == predictedLabels[i]:
        	correctHits += 1
    
    confusion_matrix = matrix[:][:]

    confusion_matrix2 = matrix / matrix.sum(axis=1)[:, None]


    print(confusion_matrix2)
    
    return (confusion_matrix, correctHits/len(actualLabels))