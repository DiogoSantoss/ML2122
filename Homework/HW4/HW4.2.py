from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
from scipy.io import arff

def loadDataFrame(file_name):
    #read data from given file
    data = arff.loadarff(file_name)
    df = pd.DataFrame(data[0])
    df['Class'] = df['Class'].str.decode('utf-8')
    return df.values.tolist()


def splitFeatureLabel(df):
    # Create feature and label lists
    df_features = [x[:-1] for x in df]
    df_labels = [x[-1] for x in df]

    return df_features, df_labels


def findClustersAndECR(k, kmeans_labels, df_labels):
    clusters = []
    counts = []
    for i in range(k):
        clusters.append([])
    for i in range(len(kmeans_labels)):
        clusters[kmeans_labels[i]].append(df_labels[i])
    print(clusters)
    for i in range(k):
        _, freq = np.unique(clusters[i], return_counts = True)
        max_freq = np.amax(freq)
        counts.append(len(clusters[i])-max_freq)
    return np.mean(counts)

def computeKMeans(df_features, k):
    kmeans = KMeans(n_clusters = k)
    kmeans_labels = kmeans.fit_predict(df_features)
    return kmeans_labels

def computeKMeansAndSilhouette(df_features, df_labels):
    k = [2,3]
    for ki in k:
        kmeans_labels = computeKMeans(df_features,ki)
        #print(kmeans_labels)
        print(findClustersAndECR(ki, kmeans_labels, df_labels))
        print(silhouette_score(df_features, labels=kmeans_labels))


def getHigherMutualInfoClusters(df_features, df_labels):

    def plotGraph(labels, colors, file_name):
        # Zip observation with label
        # 1,1 -> (1,1),(1)
        result = zip(new_features,labels)

        # Plot each point with corresponding color
        for value in result:
            plt.scatter(x=value[0][0], y=value[0][1], color=colors[int(value[1])])
        
        # Hard-Coded top-2 features (very shitty tbf)
        plt.xlabel("Cell Size Uniformity")
        plt.ylabel("Cell Shape Uniformity")
        plt.savefig(file_name)
        plt.show()

    # Compute KMeans (Before selecting features)
    kmeans_labels = computeKMeans(df_features,3)

    # Select top-2 features
    sel_cols = SelectKBest(mutual_info_classif, k = 2)
    sel_cols.fit(df_features,df_labels)

    # Create new features array with only top-2 features
    # 5,1,1,1,2,1,3,1,1 -> 1,1
    indexes = [int(col[1]) for col in sel_cols.get_feature_names_out()]
    new_features = ([[x[i] for i in indexes] for x in df_features])

    # Plot clusters
    plotGraph(kmeans_labels,["red","blue","green"],"MEGACLUSTER360NOSCOPEOMG.png")
    plt.clf()
    plotGraph([1 if label=="benign" else 0 for label in df_labels],["red","blue"],"notsomega.png")
    

if __name__ == "__main__":
    df = loadDataFrame("breast.w.arff")
    df_features, df_labels = splitFeatureLabel(df)
    computeKMeansAndSilhouette(df_features, df_labels)
    getHigherMutualInfoClusters(df_features, df_labels)