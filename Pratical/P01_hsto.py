import plotly.express as px

# homework dataset 
y1=[0.2,0.1,0.2,0.9,-0.3,-0.1,-0.9,0.2,0.7,-0.3]
y2=[0.5,-0.4,-0.1,0.8,0.3,-0.2,-0.1,0.5,-0.7,0.4]
y3=["A","A","A","B","B","B","C","C","C","C"]

print(px.data.stocks())
df = px.data.tips()
print(df)
fig = px.histogram(df, x="total_bill", nbins=20)
fig.show()