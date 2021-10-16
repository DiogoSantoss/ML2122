from scipy.io import arff
import pandas as pd

#read data from breast.w.arff
data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])


#############################################################################################
import plotly.express as px
from plotly.subplots import make_subplots

#df.hist()
#fig = make_subplots(rows=3,cols=3)
#fig = px.histogram(df, x="Clump_Thickness")
#fig.show()
mask = df["Class"].astype('string') == "b'benign'"
df_ben = df[mask]
df_mal = df[~mask]
df_ben.hist(grid=False,layout=(3,3))
df_mal.hist()



#############################################################################################
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

# to search by index use df.iloc[0] -> returns first line
# to search by feature use df.loc

features = [ 
    "Clump_Thickness", "Cell_Size_Uniformity", 
    "Cell_Shape_Uniformity", "Marginal_Adhesion", 
    "Single_Epi_Cell_Size", "Bare_Nuclei", "Bland_Chromatin",  
    "Normal_Nucleoli",  "Mitoses"]

predict = ["Class"]

randomSeed = 6
k_test = [3,5,7]
fold_num = 10

kf = KFold(n_splits=fold_num,shuffle=True,random_state=randomSeed)

for k in range(len(k_test)):
    knn = KNeighborsClassifier(n_neighbors = k_test[k])
    score_arr = []
    #train and test contain the index
    for train,test in kf.split(df):

        # create train dataframe
        df_train = pd.DataFrame([df.iloc[i] for i in train])
        df_train_features = df_train.iloc[:,:-1]
        df_train_label = df_train.iloc[:,[-1]]
        df_train_label_01 = [1 if (df_train_label.iloc[i].astype('string')=="b'benign'").bool() else 0 for i in range(len(df_train_label))]


        # create test dataframe
        df_test = pd.DataFrame([df.iloc[i] for i in test])
        df_test_features = df_train.iloc[:,:-1]
        df_test_label = df_train.iloc[:,[-1]]
        df_test_label_01 = [1 if (df_test_label.iloc[i].astype('string')=="b'benign'").bool() else 0 for i in range(len(df_test_label))] 

        knn.fit(df_train_features.values,df_train_label_01)
        result = knn.predict(df_test_features.values)
        #print(result)
        score = knn.score(df_test_features.values,df_test_label_01)
        #print(score)
        score_arr.append(score)

    median_score = 0
    for i in range(len(score_arr)):
        median_score += score_arr[i]

    print(k_test[k])
    print(median_score/len(score_arr))


# tried using transform to turn words into binary values, got errors 
#df_test_label_01 = df_test_label.transform(lambda x : 1 if (x.astype('string')=="b'benign'").bool() else 0)

# this transforms the array of binary values into a dataframe
#df_test_label_01_d = {"Class":df_test_label_01}
#df_test_label_01 = pd.DataFrame(df_test_label_01_d)
