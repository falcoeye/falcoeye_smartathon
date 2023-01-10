import pandas as pd
import numpy as np
import plotly.express as px
import sys

plottype = "scatter"
column = "n_cracks"
data = "data.csv"
if len(sys.argv) >= 2:
    data = sys.argv[1]
if len(sys.argv) >= 3:
    plottype = sys.argv[2]
if len(sys.argv) >= 4:
    column = sys.argv[3]


df = pd.read_csv(data)
print(df)
#df = df.drop_duplicates(subset=["latitude","longitude"])

if plottype == "scatter":
    fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color=column,
                            mapbox_style="open-street-map", width=1920, height=900)
elif plottype == "density":
    fig = px.density_mapbox(df, lat='latitude', lon='longitude', z=column,
                        mapbox_style="open-street-map", width=1920, height=900)
    
fig.show()
