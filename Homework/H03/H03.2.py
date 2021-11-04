from scipy.io import arff
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.validation import column_or_1d


def loadDataFrame(file_name, flag):
    #read data from given file
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    if flag:
        #correct reading of class values
        df['Class'] = df['Class'].str.decode('utf-8')
    return df.values.tolist()


def splitFeatureLabel(df):
    # Create feature and label lists
    df_features = [x[:-1] for x in df]
    df_labels = [x[-1] for x in df]

    return df_features, df_labels


def predict(alpha, early_stopping, df_features, df_labels, cv, MLP):
    clf = MLP(
        hidden_layer_sizes = (3,2),
        activation = 'relu',
        alpha = alpha,
        early_stopping = early_stopping,
        max_iter=2500)
             
    return cross_val_predict(clf, df_features, df_labels, cv = cv)


def computeConfusionMatrix():

    def print_confusion_matrix(true,pred):
        tn, fp, fn, tp = confusion_matrix(true,pred).ravel()
        print("TN: " + str(tn))
        print("FN: " + str(fn))
        print("TP: " + str(tp))
        print("FP: " + str(fp))
        print("------------------------")


    cv1 = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    # Load DF
    df1 = loadDataFrame("breast.w.arff", True)
    df1_features, df1_labels = splitFeatureLabel(df1)
    # No Early Stop
    pred_no_early_stopping = predict(0.5, False, df1_features, df1_labels, cv1, MLPClassifier)
    print_confusion_matrix(df1_labels, pred_no_early_stopping)
    # Early Stop
    pred_early_stopping = predict(0.5, True, df1_features, df1_labels, cv1, MLPClassifier)
    print_confusion_matrix(df1_labels, pred_early_stopping)


def computeBoxPlot():

    cv2 = KFold(n_splits = 5, shuffle = True, random_state = 0)
    # Load DF
    df2 = loadDataFrame("kin8nm.arff", False)
    df2_features, df2_labels = splitFeatureLabel(df2)
    # No regularization
    pred_no_reg = predict(0,False, df2_features, df2_labels, cv2, MLPRegressor)
    residuals_no_reg = [df2_labels[i] - pred_no_reg[i] for i in range(len(pred_no_reg))]
    # Regularization
    pred_reg = predict(0.0001, False, df2_features, df2_labels, cv2, MLPRegressor)
    residuals_reg = [df2_labels[i] - pred_reg[i] for i in range(len(pred_reg))]                      
    # Save Boxplot as image
    plt.boxplot([residuals_no_reg,residuals_reg], positions=[1, 2], labels=["no_reg","reg"])
    plt.savefig("megaBox.png")


if __name__ == "__main__":

    while(computeConfusionMatrix()):
        pass

    computeBoxPlot()