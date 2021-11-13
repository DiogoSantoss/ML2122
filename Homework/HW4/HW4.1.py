import numpy as np
import matplotlib.pyplot as plt

cov1 = np.array([np.array([1,0]),np.array([0,1])])
cov2 = np.array([np.array([2,0]),np.array([0,2])])

x1 = mean1 = np.array([2,4])
x2 = mean2 = np.array([-1,-4])
x3 = np.array([-1,2])
x4 = np.array([4,0])
X = [x1,x2,x3,x4]
prior1 = 0.7
prior2 = 0.3

def mvn(x: np.array, d, mean: np.array, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))

norm_post = []

for i in range(len(X)):
    mvn1 = mvn(X[i], 2, mean1, cov1)
    mvn2 = mvn(X[i], 2, mean2, cov2)

    print("P(x" + str(i+1) + "| c=1) = " + str(mvn1))
    print("P(x" + str(i+1) + "| c=2) = " + str(mvn2))

    print("P(x" + str(i+1) + ", c=1) = " + str(mvn1*prior1))
    print("P(x" + str(i+1) + ", c=2) = " + str(mvn2*prior2))

    den = mvn1*prior1 + mvn2*prior2
    norm_post.append([mvn1*prior1/den,mvn2*prior2/den])


print(norm_post)

posteriors = np.transpose(norm_post)
print(posteriors)
w1, w2 = np.sum(posteriors[0]), np.sum(posteriors[1])
assert (w1+w2 == 4)
weights = [w1, w2]
new_prior1, new_prior2 = w1/4, w2/4
print(weights, new_prior1, new_prior2)

new_mean1 = np.sum([posteriors[0][i]*X[i] for i in range(len(X))],axis=0)/w1
new_mean2 = np.sum([posteriors[1][i]*X[i] for i in range(len(X))],axis=0)/w2
new_mean = [new_mean1, new_mean2]
print("New means: " + str(new_mean))
new_sigma = []

for i in range(len(new_mean)):
    new_sigma11 = new_sigma12 = new_sigma22 = 0
    for j in range(len(X)):
        new_sigma11 += (posteriors[i][j]/weights[i])*(X[j][0]-new_mean[i][0])**2
        new_sigma22 += (posteriors[i][j]/weights[i])*(X[j][1]-new_mean[i][1])**2
        new_sigma12 += (posteriors[i][j]/weights[i])*(X[j][0]-new_mean[i][0])*(X[j][1]-new_mean[i][1])
    new_sigma.append(np.array([[new_sigma11, new_sigma12], [new_sigma12, new_sigma22]]))
    print("New Sigma" + str(i+1) + ": " + str(new_sigma[i][0]) + "\n\t    " + str(new_sigma[i][1]))

clusters = [[],[]]

for i in range(len(X)):
    clusters[0].append(X[i]) if posteriors[0][i] > posteriors[1][i] else clusters[1].append(X[i])
    
print(clusters)


# (2)
# Cluster1:  x1,x3,x4
# Cluster2:  x2

# S(x1) = 1 - a(x1)/b(x1) = 1 - ((3.605551275463989+4.47213595499958)/2)/8.54400374531753
#       = 0.52728910992
# S(x3) = 1 - a(x3)/b(x3) = 1 - ((3.605551275463989+5.385164807134504)/2)/6.0
#       = 0.25077365978
# S(x4) = 1 - a(x4)/b(x4) = 1 - ((4.47213595499958+5.385164807134504)/2)/6.4031242374328485
#       = 0.23027412895

# S(c1) = (S(x1)+S(x3)+S(x4))/3 = (0.52728910992+0.25077365978+0.23027412895)/3 
#       = 0.33611229955

# S(c2) = S(x2) = 1 - a(x2)/b(x2) = 1

# S(c) = (S(c1)+S(c2))/2 = (0.33611229955+1)/2 = 0.66805614977

print(np.linalg.norm(X[0]-X[2])) # x1,x3 dist 3.605551275463989
print(np.linalg.norm(X[0]-X[3])) # x1,x4 dist 4.47213595499958
print(np.linalg.norm(X[2]-X[3])) # x3,x4 dist 5.385164807134504
print(np.linalg.norm(X[0]-X[1])) # x1,x2 dist 8.54400374531753
print(np.linalg.norm(X[2]-X[1])) # x3,x2 dist 6.0
print(np.linalg.norm(X[3]-X[1])) # x4,x2 dist 6.4031242374328485

# (3)
# (a)
# (i)   MLP with three hidden layers with as much nodes as the number of input variables
# (ii)  Decision tree assuming input variables are discretized using three bins
# (iii) Bayesian classifier with a multivariate Gaussian likelihood

# (i) 
# dVC(MLP) = numero de parametros
# 5    ->    5     ->    5    ->     5     ->    1
# ( 5*5 + 1*5 )*3 + 5*1 + 1*1  = 96
#   W     b          W     b
#                   
# (ii)
# dVC(tree) = 2^m = 2^5 = 32
#
# (iii)
# dVC(Bay) = numero de parametros = [1 ou 0 (se puder ser obtido por excl. partes)] + [#param miu + #param distintos sigma] = 
#          = [1 ou 0] + [m + m(m+1)/2]*2
# P(c=1|x) = P(c=1)N(x|miu,sigma) => 1 + 5 + 5*(5+1)/2 = 21
# P(c=0|x) = P(c=0)N(x|miu,sigma) => 0 + 5 + 5*(5+1)/2 = 20
# Total = 41

# (b) e (c)

def calculate_vc_dimensions(m, flag, file_name):
    i = (m**2 + m)*3 + m + 1
    ii = 2**m
    iii = 1 + (m+m*(m+1)/2)*2
    plt.plot(m,i,label="MLP")
    if flag:
        plt.plot(m,ii,label="Decision Tree")
    plt.plot(m,iii,label="Bayesian")
    plt.legend(loc="upper left")
    plt.savefig(file_name)
    plt.show()
    
calculate_vc_dimensions(np.array([2,5,10,12,13]), True, "3b)")
plt.clf()
calculate_vc_dimensions(np.array([2,5,10,30,100,300,1000]), False, "3c)")


def sketchCluster():
    
    plt.scatter(x=2,y=4,color="blue")
    plt.scatter(x=-1,y=-4,color="green")
    plt.scatter(x=-1,y=2,color="blue")
    plt.scatter(x=4,y=0,color="blue")
    plt.xlabel("y1")
    plt.ylabel("y2")
    plt.savefig("sketch.png")
    plt.show()

sketchCluster()