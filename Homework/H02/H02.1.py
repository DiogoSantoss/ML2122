import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy, relfreq
from math import sqrt

y1_train = np.array([1,1,0,1,2,1,2,0])
y2_train = np.array([1,1,2,2,0,1,0,2])
y3_train = np.array([0,5,4,3,7,1,2,9])
output_train = np.array([1,3,2,0,6,4,5,7])
train = [y1_train,y2_train,y3_train]

x_train = [[1,1,0],[1,1,5],[0,2,4],[1,2,3],[2,0,7],[1,1,1],[2,0,2],[0,2,9]]

y1_test = np.array([2,1])
y2_test = np.array([0,2])
y3_test = np.array([0,1])
output_test = np.array([2,4])

x_test = [[2,0,0],[1,2,1]]


def applyPhi():
    for i in range(len(x_train)):
        x_train[i] = np.linalg.norm(x_train[i]) 

def getDesignMatrix():
    x = []
    for i in range(len(y1_train)):
        x.append([1,x_train[i],x_train[i]**2,x_train[i]**3])

    return np.array(x)

def getWeights(designMatrix: np.array):

    x = designMatrix
    x_T = x.transpose()
    # Compute (X^T * X)^-1
    x_I = np.linalg.inv(np.dot(x_T,x))
    # Compute X_I * X^T * Z
    w = np.dot(np.dot(x_I,x_T),output_train)

    return w

def computePolynomialRegression(w, x):
    norm = np.linalg.norm(x)
    return sum([w[i]*(norm**i) for i in range(len(w))])


def testPolynomialRegression(w: np.array):
    #RMSE
    return sqrt(mean_squared_error(
        output_test,
        [computePolynomialRegression(w,x) for x in x_test],
    ))

def binarizationY3():
    result = []
    for i in output_train:
        if(i>=4):
            result.append(1)
        else:
            result.append(0)
    return result


def IG(y: np.array):
    #def conditional_entropy(z, y):
        #return sum(relfreq())
    output_entropy = entropy(relfreq(output_train, numbins=8).frequency, base=2)
    # IG(z|y1)=3/2
    # IG(z|y2)=
    # IG(z|y3)=
    print(output_entropy)


    

if __name__ == "__main__":

    applyPhi()
    x = getDesignMatrix()
    print(x)
    w = getWeights(x)
    print(w)

    f = testPolynomialRegression(w)
    print(f) #1.2567231583983314

    y3_bin = binarizationY3()
    print(y3_bin)

    IG(y1_train)
    IG(y2_train)
    IG(y3_bin)




    


    



