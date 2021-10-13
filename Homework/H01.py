from scipy.io import arff
import pandas as pd
import plotly.express as px

data = arff.loadarff('breast.w.arff')
df = pd.DataFrame(data[0])

fig = px.histogram(df, x="Clump_Thickness")
fig.add_subplot()
fig.show()

