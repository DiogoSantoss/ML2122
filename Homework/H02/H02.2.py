from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import tree
import pandas as pd


def loadDataFrame():
    #read data from breast.w.arff
    data = arff.loadarff('breast.w.arff')
    df = pd.DataFrame(data[0])
    #correct reading of class values
    df['Class'] = df['Class'].str.decode('utf-8')

    return df


def splitFeatureLabel(x):
        x_features = x.iloc[:,:-1]
        x_label = x.iloc[:,[-1]]
        x_label_01 = [1 if elem == "malignant" else 0 for elem in x_label["Class"]]

        return x_features, x_label, x_label_01


def runDecisionTree(df_features,df_labels,n,cv):

    clf = tree.DecisionTreeClassifier(max_depth=n)
    score = cross_val_score(clf,df_features,df_labels,cv=cv,scoring="accuracy").mean()

    return score


def runMutualInfoDecisionTree(cv,df_features, df_label):

    scores = []
    for i in [1,3,5,9]:
        sel_cols = SelectKBest(mutual_info_classif,k=i)
        sel_cols.fit(df_features,df_label_01)

        score = runDecisionTree(df_features[sel_cols.get_feature_names_out()],df_label,i,cv)
        scores.append(score)
    return scores


def runMaxDepthDecisionTree(cv,df_features, df_label):

    scores = []
    for i in [1,3,5,9]:
        score = runDecisionTree(df_features,df_label,i,cv)
        scores.append(score)
    return scores
    

def plotAndSaveGraph(scores1,scores2):
    plt.plot([1,3,5,9],scores1,label="Mutual info")
    plt.plot([1,3,5,9],scores2,label="Max depth")
    plt.legend(loc='upper right')
    plt.savefig("megagrafico.png")


if __name__ == "__main__":

    df = loadDataFrame()

    cv = KFold(n_splits=10)
    df_features, df_label, df_label_01 = splitFeatureLabel(df)

    scores_mutual = runMutualInfoDecisionTree(cv,df_features, df_label)  
    scores_depth = runMaxDepthDecisionTree(cv,df_features, df_label)

    print(f"Scores with Mutual Info:\n{scores_mutual}\nScores with Max Depth:\n{scores_depth}")
    plotAndSaveGraph(scores_mutual,scores_depth)