from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from scipy.io import arff
import pandas as pd

def loadDataFrame(file_name):
    #read data from given file
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    #correct reading of class values
    df['Class'] = df['Class'].str.decode('utf-8')
    return df.values

def splitFeatureLabel(df):
    df_features = [x.tolist()[:-1] for x in df]
    df_labels = [x.tolist()[-1] for x in df]
    return df_features, df_labels

def print_confusion_matrix(true,pred):
    tn, fp, fn, tp = confusion_matrix(true,pred).ravel()
    print("TN: " + str(tn))
    print("FN: " + str(fn))
    print("TP: " + str(tp))
    print("FP: " + str(fp))

def mlp_classifier(kf, x, y, clf):
    real = []
    predicted = []
    for train_index, test_index in kf.split(x, y):
        x_train, x_test = [x[i] for i in train_index], [x[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
        clf.fit(x_train, y_train)
        predicted += clf.predict(x_test).tolist()
        real += y_test
    print_confusion_matrix(real,predicted)


if __name__ == "__main__":
    dfb = loadDataFrame('breast.w.arff')
    kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)
    features, labels = splitFeatureLabel(dfb)
    clf_no_early_stop = MLPClassifier(
            hidden_layer_sizes = (3,2),
            activation = 'relu',
            early_stopping = False)

    clf_early_stop = MLPClassifier(
            hidden_layer_sizes = (3,2),
            activation = 'relu',
            early_stopping = True)

    #mlp_classifier(kf, features, labels, clf_no_early_stop)
    mlp_classifier(kf, features, labels, clf_early_stop)