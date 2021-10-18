from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from scipy import stats
from scipy.io import arff
import pandas as pd

###############################################
#                  READ DATA                  #
###############################################

#read data from breast.w.arff
data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])
#correct reading of class values
df['Class'] = df['Class'].str.decode('utf-8')

###############################################
#                  HISTOGRAM                  #
###############################################

def getHistograms():

    #split data based on class
    mask = df["Class"] == "benign"
    df_ben = df[mask]
    df_mal = df[~mask]

    features = list(df.columns)[:-1]

    fig, axs = plt.subplots(3, 3)
    fig.tight_layout()

    #fill each plot with corresponding histogram
    for i in range(3):
        for j in range(3):
            axs[i,j].hist([df_ben[features[i+j]],df_mal[features[i+j]]],bins=10,color=["green","red"],label=["Benign","Malignant"])
            axs[i,j].set_title(features[i+j].replace("_"," "))
    
    #create legend
    handles, labels = axs[i,j].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()


###############################################
#                 CLASSIFIERS                 #
###############################################


kf = KFold(n_splits=10,shuffle=True,random_state=6)
knn = KNeighborsClassifier(n_neighbors = 3)
naive_bayes = MultinomialNB()

def classifier_accuracies (clf):
    
    acc = []
    for train,test in kf.split(df):
        
        #split data in train and test
        df_train = pd.DataFrame([df.iloc[i] for i in train])
        df_train_features = df_train.iloc[:,:-1]
        df_train_label = df_train.iloc[:,[-1]]
        df_train_label_01 = [1 if elem == "malignant" else 0 for elem in df_train_label["Class"]]

        df_test = pd.DataFrame([df.iloc[i] for i in test])
        df_test_features = df_test.iloc[:,:-1]
        df_test_label = df_test.iloc[:,[-1]]
        df_test_label_01 = [1 if elem == "malignant" else 0 for elem in df_test_label["Class"]]

        #train and score the classifier
        clf.fit(df_train_features.values,df_train_label_01)
        acc.append(clf.score(df_test_features.values,df_test_label_01))
    
    return acc

def knnVSnaive():

    knn_acc = classifier_accuracies(knn)
    naive_acc = classifier_accuracies(naive_bayes)

    print(knn_acc)
    print(naive_acc)

    _, pvalue = stats.ttest_rel(knn_acc, naive_acc, alternative="greater")
    print(pvalue)


if __name__ == "__main__":
    getHistograms()
    knnVSnaive()