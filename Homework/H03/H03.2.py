from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import tree
import pandas as pd


def loadDataFrame(file_name):
    #read data from given file
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    #correct reading of class values
    df['Class'] = df['Class'].str.decode('utf-8')

    return df

def splitFeatureLabel(df, x):
    x = pd.DataFrame([df.iloc[i] for i in x])
    x_features = x.iloc[:,:-1]
    x_label = x.iloc[:,[-1]]
    x_label = [1 if elem == "malignant" else 0 for elem in x_label["Class"]]

    return x_features, x_label

def print_confusion_matrix(true,pred):
    tn, fp, fn, tp = confusion_matrix(true,pred).ravel()
    print("TN: " + str(tn))
    print("FN: " + str(fn))
    print("TP: " + str(tp))
    print("FP: " + str(fp))
    return tn, fp, fn, tp

def mlp_classifier(kf, df):
    real = []
    predicted = []
    for train, test in kf.split(df):
        train_features, train_labels = splitFeatureLabel(df, train)
        test_features, test_labels = splitFeatureLabel(df, test)
        #como fazer o l2 com o alpha? i'm tired lol
        #usar early_stopping = True/False
        #diz que a otimização ainda n convergiu... hmmmm
        clf = MLPClassifier(random_state = 1, hidden_layer_sizes = [3,2]).fit(train_features, train_labels)
        real += clf.predict(test_features).tolist()
        predicted += test_labels
    print_confusion_matrix(real,predicted)


if __name__ == "__main__":
    dfb = loadDataFrame('breast.w.arff')
    kf = KFold(n_splits = 5, shuffle = True, random_state = 1)
    mlp_classifier(kf, dfb)
    