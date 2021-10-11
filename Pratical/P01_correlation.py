from scipy.stats import pearsonr,spearmanr

# prepare data
y1=[0.2,0.1,0.2,0.9,-0.3,-0.1,-0.9,0.2,0.7,-0.3]
y2=[0.5,-0.4,-0.1,0.8,0.3,-0.2,-0.1,0.5,-0.7,0.4]
y3=["A","A","A","B","B","B","C","C","C","C"]
# yi ranks
r1=[7,5,7,10,2.5,4,1,7,9,2.5]
r2=[8.5,2,4.5,10,6,3,4.5,8.5,1,7]

# calculate Pearson's correlation
corr, _ = pearsonr(y1, y2)
print('Pearsons correlation: %.3f' % corr)

# calculate spearman's correlation
corr, _ = spearmanr(y1, y2)
print('Spearmans correlation: %.3f' % corr)








