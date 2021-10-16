import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score

y1_neg = np.array([0.6, 0.1, 0.2, 0.1])
y1_pos = np.array([0.3, -0.1, -0.3, 0.2, 0.4, -0.2])
y2_neg = np.array(["A", "B", "A", "C"])
y2_pos = np.array(["B", "C", "C", "B", "A", "C"])
y3_neg = np.array([0.2, -0.1, -0.1, 0.8])
y3_pos = np.array([0.1, 0.2, -0.1, 0.5, -0.4, 0.4])
y4_neg = np.array([0.4, -0.4, 0.2, 0.8])
y4_pos = np.array([0.3, -0.2, 0.2, 0.6, -0.7, 0.3])

p_neg = 0.4
p_pos = 0.6

miu_y1_neg = np.mean(y1_neg) #0.24999999999999997
miu_y1_pos = np.mean(y1_pos) #0.049999999999999996

cov_y1_neg = np.cov(y1_neg) #0.05666666666666665
cov_y1_pos = np.cov(y1_pos) #0.08300000000000002

mode_y2_neg = stats.mode(y2_neg)[0][0] #A
mode_y2_pos = stats.mode(y2_pos)[0][0] #C

miu_y34_neg = np.array([np.mean(y3_neg), np.mean(y4_neg)]) #[0.2  0.25]
miu_y34_pos = np.array([np.mean(y3_pos), np.mean(y4_pos)]) #[0.11666667 0.08333333]

cov_y34_neg = np.cov([y3_neg, y4_neg])  #[[0.18 0.18] [0.18 0.25]]
cov_y34_pos = np.cov([y3_pos, y4_pos])  #[[0.10966667 0.12233333] [0.12233333 0.21366667]]

inv_cov_y34_neg = np.linalg.inv(np.cov([y3_neg, y4_neg])) #[[ 19.84126984 -14.28571429] [-14.28571429  14.28571429]]
inv_cov_y34_pos = np.linalg.inv(np.cov([y3_pos, y4_pos])) #[[ 25.23622047 -14.4488189 ] [-14.4488189   12.95275591]]

det_cov_y34_neg = np.linalg.det(np.cov([y3_neg, y4_neg])) #0.0126
det_cov_y34_pos = np.linalg.det(np.cov([y3_pos, y4_pos])) #0.008466666666666664


# P(c|y1,y2,y3,y4) = P(c)P(y1,y2,y3,y4|c)/P(y1,y2,y3,y4) = P(c)P(y1|c)P(y2|c)P(y3,y4|c)/P(y1,y2,y3,y4)

# P(y1,y2,y3,y4) = P(y1,y2,y3,y4 e c=0) + P(y1,y2,y3,y4 e c=1)
# P(y1,y2,y3,y4 e c) = P(c)P(y1,y2,y3,y4|c) = P(c)P(y1|c)P(y2|c)P(y3,y4|c)
def pred_c(
    x: np.array, 
    p_pos, p_neg, 
    y2_pos, y2_neg, 
    miu_uni_pos, miu_uni_neg, 
    cov_uni_pos, cov_uni_neg,
    miu_bi_pos, miu_bi_neg, 
    cov_bi_pos, cov_bi_neg
    ):

    print(f"x: {x}")

    # P(c)P(y1|c)P(y2|c)P(y3,y4|c)
    prob_num_pos = prob(x, p_pos, y2_pos, miu_uni_pos, cov_uni_pos, miu_bi_pos, cov_bi_pos)
    print(f"x|c=1: {prob_num_pos}")
    prob_num_neg = prob(x, p_neg, y2_neg, miu_uni_neg, cov_uni_neg, miu_bi_neg, cov_bi_neg)
    print(f"x|c=0: {prob_num_neg}")

    # P(y1,y2,y3,y4) = P(y1,y2,y3,y4 e c=0) + P(y1,y2,y3,y4 e c=1)
    prob_den = prob_num_pos + prob_num_neg
    print(f"denominador: {prob_den}")

    result = 1 if prob_num_pos > prob_num_neg else 0
    return prob_num_pos/prob_den, result

# P(c)P(y1|c)P(y2|c)P(y3,y4|c)
def prob(x: np.array, p_c, y2_c, miu_uni, cov_uni, miu_bi, cov_bi):

    p_y1 = prob_y1(x[0],miu_uni,cov_uni)
    print(f"p_y1: {p_y1}")
    p_y2 = prob_y2(x[1],y2_c)  
    print(f"p_y2: {p_y2}")
    p_y3y4 = prob_y3y4(x,2,miu_bi,cov_bi)
    print(f"p_y3y4: {p_y3y4}")

    return p_c*p_y1*p_y2*p_y3y4


def prob_y1(x, miu, cov):
    print(f"miu: {miu}")
    print(f"cov: {cov}")
    return univariate_normal(x,miu,cov)


def prob_y2(x,y):
    """probability of x in y set"""
    total = 0
    for i in range(len(y)):
        if x == y[i]:
            total += 1
    return total/len(y)


def prob_y3y4(x,d,miu,cov):
    print(f"miu: {miu}")
    print(f"cov: {cov}")
    return multivariate_normal(np.array([x[2],x[3]]), d, miu, cov)


def univariate_normal(x, mean, variance):
    """pdf of the univariate normal distribution."""
    return ((1. / np.sqrt(2 * np.pi * variance)) * 
            np.exp(-(x - mean)**2 / (2 * variance)))


def multivariate_normal(x: np.array, d, mean: np.array, covariance):
    """pdf of the multivariate normal distribution."""
    x_m = x - mean
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) * 
            np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))


def print_confusion_matrix(true,pred):
    tn, fp, fn, tp = confusion_matrix(true,pred).ravel()
    print("TN: " + str(tn))
    print("FN: " + str(fn))
    print("TP: " + str(tp))
    print("FP: " + str(fp))
    return tn, fp, fn, tp


def find_accuracies(probs_pos, x_c):

    def predict(prob_pos, threshold):
        return 1 if prob_pos > threshold else 0 #Should we use > or >=? Two different optimal thresholds will be returned that way 

    size = len(probs_pos)

    # accuracy for each threshold
    accuracies = []
    best_accuracy = 0

    for i in range(size):
        x_pred = []
        for j in range(size):
            x_pred.append(predict(probs_pos[j], probs_pos[i])) #Find prediction of probs_pos[j] using probs_pos[i] as threshold
        tn, fp, fn, tp = confusion_matrix(x_c,x_pred).ravel()
        accuracy = (tn+tp)/size
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = probs_pos[i]
    print(accuracies)
    print(f"Best threshold: {best_threshold}")


if __name__ == "__main__":

    y1 = [0.6, 0.1, 0.2, 0.1,0.3, -0.1, -0.3, 0.2, 0.4, -0.2]
    y2 = ["A", "B", "A", "C","B", "C", "C", "B", "A", "C"]
    y3 = [0.2, -0.1, -0.1, 0.8,0.1, 0.2, -0.1, 0.5, -0.4, 0.4]
    y4 = [0.4, -0.4, 0.2, 0.8,0.3, -0.2, 0.2, 0.6, -0.7, 0.3]

    x_c = [0,0,0,0,1,1,1,1,1,1]
    x_pred = []

    # probability of P(c=1|x)
    probs_pos = []

    for i in range(10):
        x = [y1[i],y2[i],y3[i],y4[i]]
        prob_pos, result = pred_c(
                    x,
                    p_pos,p_neg,
                    y2_pos, y2_neg,
                    miu_y1_pos, miu_y1_neg, 
                    cov_y1_pos, cov_y1_neg,
                    miu_y34_pos, miu_y34_neg, 
                    cov_y34_pos, cov_y34_neg
                )
        x_pred.append(result)
        probs_pos.append(prob_pos)
    
    #tn, fp, fn, tp = print_confusion_matrix(x_c,x_pred)
    #print("F1 Score: " + str(f1_score(x_c, x_pred)))
    #print("P(c=1|x) for each x")
    #print(probs_pos)
    #find_accuracies(probs_pos, x_c)