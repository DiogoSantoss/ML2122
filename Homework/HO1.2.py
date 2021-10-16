from scipy.io import arff
import pandas as pd

#read data from breast.w.arff
data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])


#############################################################################################

# separate dataframe in two
mask = df["Class"].astype('string') == "b'benign'"
df_ben = df[mask]
df_mal = df[~mask]

# only works on jupyter
df_ben.hist(grid=False,xlabelsize=10,layout=(3,3),figsize=(9,9))
df_mal.hist(grid=False,xlabelsize=10,layout=(3,3),figsize=(9,9))  


#############################################################################################
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

features = [ 
    "Clump_Thickness", "Cell_Size_Uniformity", 
    "Cell_Shape_Uniformity", "Marginal_Adhesion", 
    "Single_Epi_Cell_Size", "Bare_Nuclei", "Bland_Chromatin",  
    "Normal_Nucleoli",  "Mitoses"]

predict = ["Class"]

randomSeed = 6
k_test = [3,5,7]
fold_num = 10

# creates folds
kf = KFold(n_splits=fold_num,shuffle=True,random_state=randomSeed)

# train knn for k={3,5,7}
for k in k_test:
    knn = KNeighborsClassifier(n_neighbors = k)
    # each k will get an average score
    score_arr = []
    #train and test contain the index
    for train,test in kf.split(df):

        # create train dataframe
        # from the train index creates df_train
        df_train = pd.DataFrame([df.iloc[i] for i in train])
        # separate features from labels
        df_train_features = df_train.iloc[:,:-1]
        df_train_label = df_train.iloc[:,[-1]]
        # transform label from words to binary values
        df_train_label_01 = [1 if (df_train_label.iloc[i].astype('string')=="b'benign'").bool() else 0 for i in range(len(df_train_label))]


        # create test dataframe
        df_test = pd.DataFrame([df.iloc[i] for i in test])
        df_test_features = df_train.iloc[:,:-1]
        df_test_label = df_train.iloc[:,[-1]]
        df_test_label_01 = [1 if (df_test_label.iloc[i].astype('string')=="b'benign'").bool() else 0 for i in range(len(df_test_label))] 

        knn.fit(df_train_features.values,df_train_label_01)
        knn.predict(df_test_features.values)
        score = knn.score(df_test_features.values,df_test_label_01)
        score_arr.append(score)

    # get average score across all folds tested
    average_score = 0
    for i in score_arr:
        average_score += i
    average_score = average_score/len(score_arr)

    print(f"k={k}, with score={average_score}")

# tried using transform to turn words into binary values, got errors 
#df_test_label_01 = df_test_label.transform(lambda x : 1 if (x.astype('string')=="b'benign'").bool() else 0)

# this transforms the array of binary values into a dataframe
#df_test_label_01_d = {"Class":df_test_label_01}
#df_test_label_01 = pd.DataFrame(df_test_label_01_d)
