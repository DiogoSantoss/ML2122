from scipy.io import arff
import pandas as pd

data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])

print(df.head(700))