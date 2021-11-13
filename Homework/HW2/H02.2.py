from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import StratifiedKFold, cross_validate
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn import tree
import pandas as pd


def loadDataFrame():
    
    data = arff.loadarff('breast.w.arff')
    df = pd.DataFrame(data[0])
    
    df['Class'] = df['Class'].str.decode('utf-8')

    return df


def splitFeatureLabel(x):

    x_features = x.iloc[:,:-1]
    x_label = x.iloc[:,[-1]]
    x_label_01 = [1 if elem == "malignant" else 0 for elem in x_label["Class"]]

    return x_features, x_label, x_label_01


def runDecisionTree(df_features,df_labels,n,cv):

    clf = tree.DecisionTreeClassifier(max_depth=n)
    scores = cross_validate(clf,df_features,df_labels,cv=cv,scoring="accuracy",return_train_score=True)
    test_score = scores["test_score"].mean()
    train_score = scores["train_score"].mean()
    
    return test_score,train_score


def runMutualInfoDecisionTree(cv,df_features, df_label):

    test_scores = []
    train_scores = []
    for i in [1,3,5,9]:
        sel_cols = SelectKBest(mutual_info_classif,k=i)
        sel_cols.fit(df_features,df_label_01)

        test_score,train_score = runDecisionTree(df_features[sel_cols.get_feature_names_out()],df_label,i,cv)
        test_scores.append(test_score)
        train_scores.append(train_score)
    return test_scores,train_scores


def runMaxDepthDecisionTree(cv,df_features, df_label):

    test_scores = []
    train_scores = []
    for i in [1,3,5,9]:
        test_score,train_score = runDecisionTree(df_features,df_label,i,cv)
        test_scores.append(test_score)
        train_scores.append(train_score)
    return test_scores,train_scores
    

def plotAndSaveGraph(test_scores_mutual,train_scores_mutual,test_scores_depth,train_scores_depth):

    plt.plot([1,3,5,9],test_scores_mutual,label="Test Mutual info")
    plt.plot([1,3,5,9],train_scores_mutual,label="Train Mutual info")
    plt.plot([1,3,5,9],test_scores_depth,label="Test Max depth")
    plt.plot([1,3,5,9],train_scores_depth,label="Train Max depth")

    plt.xlabel('Number of selected features / Tree depth'); plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig("plot.png")


if __name__ == "__main__":

    df = loadDataFrame()

    df_features, df_label, df_label_01 = splitFeatureLabel(df)
    cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 6)

    test_scores_mutual,train_scores_mutual = runMutualInfoDecisionTree(cv,df_features, df_label)  
    test_scores_depth,train_scores_depth = runMaxDepthDecisionTree(cv,df_features, df_label)

    plotAndSaveGraph(test_scores_mutual,train_scores_mutual,test_scores_depth,train_scores_depth)