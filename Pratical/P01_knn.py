import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import accuracy_score

# homework dataset 
y1=[0.2,0.1,0.2,0.9,-0.3,-0.1,-0.9,0.2,0.7,-0.3]
y2=[0.5,-0.4,-0.1,0.8,0.3,-0.2,-0.1,0.5,-0.7,0.4]
y3=["A","A","A","B","B","B","C","C","C","C"]

# Assign the data and target to separate variables
features = np.array([[x,y] for x,y in zip(y1,y2)])
labels = np.array([{"A":0, "B":1, "C":2}[x] for x in y3])

# split our dataset
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.4)
print("----TRAIN----")
print("Input train")
print(features_train)
print("Output train")
print(labels_train)

# will create the empty model
# In order to provide the operations to the model we should train them
# have to train the model with the Features and the Labels
classifier=neighbors.KNeighborsClassifier()

# train the model with fit function
classifier.fit(features_train,labels_train)

# Predictions can be done with predict function
predictions=classifier.predict(features_test)

# these predictions can be matched with the expected output 
# to measure the accuracy value
score=accuracy_score(labels_test,predictions)

# Print Results
print("----TESTS----")
print("Input test")
print(features_test)
print("Output test")
print(labels_test)
print("Predictions")
print(predictions)
print("Score:",score)