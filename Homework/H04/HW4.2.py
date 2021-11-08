from sklearn.cluster import KMeans
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


def computeKMeansAndSilhouette(df_features):
    k = [2,3]
    for ki in k:
        kmeans = KMeans(n_clusters=ki)
        kmeans_labels = kmeans.fit_predict(df_features)
        #print(kmeans_labels)
        #print(silhouette_score(df_features, labels=kmeans_labels))
        print(findClustersAndECR(ki, kmeans_labels, df_labels))


if __name__ == "__main__":
    df = loadDataFrame("breast.w.arff")
    df_features, df_labels = splitFeatureLabel(df)
    computeKMeansAndSilhouette(df_features)

