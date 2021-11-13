import pandas as pd
from scipy.io import arff
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict

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
        random_state = 0)
             
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
    pred_no_early_stopping = predict(3, False, df1_features, df1_labels, cv1, MLPClassifier)
    print_confusion_matrix(df1_labels, pred_no_early_stopping)
    # Early Stop
    pred_early_stopping = predict(3, True, df1_features, df1_labels, cv1, MLPClassifier)
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
    pred_reg1 = predict(0.1, False, df2_features, df2_labels, cv2, MLPRegressor)
    residuals_reg1 = [df2_labels[i] - pred_reg1[i] for i in range(len(pred_reg1))]
    pred_reg5 = predict(1, False, df2_features, df2_labels, cv2, MLPRegressor)
    residuals_reg5 = [df2_labels[i] - pred_reg5[i] for i in range(len(pred_reg5))]                            
    # Save Boxplot as image
    plt.xlabel("alpha")
    plt.ylabel("Residual")
    plt.boxplot([residuals_no_reg,residuals_reg1,residuals_reg5], positions=[1, 2, 3], labels = ["0","1","5"])
    plt.savefig("megaBox.png")


if __name__ == "__main__":

    computeConfusionMatrix()
    computeBoxPlot()