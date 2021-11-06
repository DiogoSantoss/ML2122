import numpy as np

cov1 = cov2 = np.array([np.array([1,0]),np.array([0,1])])

x1 = mean1 = np.array([2,2])
x2 = np.array([0,2])
x3 = mean2 =np.array([0,0])
X = [x1,x2,x3]
prior1 = 0.6
prior2 = 0.4

def mvn(x: np.array, d, mean: np.array, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

likelihoods = []
joint_prob = []
norm_post = []

for i in range(len(X)):
    mvn1 = mvn(X[i], 2, mean1, cov1)
    mvn2 = mvn(X[i], 2, mean2, cov2)

    print("P(x" + str(i+1) + "| c=1) = " + str(mvn1))
    print("P(x" + str(i+1) + "| c=2) = " + str(mvn2))

    likelihoods.append([mvn1,mvn2])
    joints = [mvn1*prior1,mvn2*prior2]
    joint_prob.append(joints)
    norm_post.append([joints[0]/np.sum(joints),joints[1]/np.sum(joints)])

clusters = np.transpose(norm_post)
w1, w2 = np.sum(clusters[0]), np.sum(clusters[1])
assert (w1+w2 == 3)
new_prior1, new_prior2 = w1/3, w2/3


new_mean1 = np.sum([clusters[0][i]*X[i] for i in range(len(X))],axis=0)/w1
new_mean2 = np.sum([clusters[1][i]*X[i] for i in range(len(X))],axis=0)/w2

new_mean = [new_mean1, new_mean2]
new_sigma = []
for i in range(len(new_mean)):
    new_sigma11 = new_sigma12 = new_sigma22 = 0
    for j in range(len(X)):
        #print((X[j][0]-new_mean[i][0])**2)
        new_sigma11 += (clusters[i][j]/w1)*(X[j][0]-new_mean[i][0])**2
        new_sigma22 += (clusters[i][j]/w1)*(X[j][1]-new_mean[i][1])**2
        new_sigma12 += (clusters[i][j]/w1)*(X[j][0]-new_mean[i][0])*(X[j][1]-new_mean[i][1])
    new_sigma.append(np.array([[new_sigma11, new_sigma12], [new_sigma12, new_sigma22]]))
    print(new_sigma[i])