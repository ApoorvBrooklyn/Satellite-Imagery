from glob import glob

import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

import rasterio as rio
from rasterio.plot import plotting_extent
from rasterio.plot import show
from rasterio.plot import reshape_as_raster, reshape_as_image

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

import plotly.graph_objects as go

np.seterr(divide='ignore', invalid='ignore')

S_sentinel_bands = glob("/content/drive/MyDrive/Satellite_data/Aquileia_Italy/*B?*.tiff")
S_sentinel_bands.sort()
S_sentinel_bands


l = []
for i in S_sentinel_bands:
  with rio.open(i, 'r') as f:
    l.append(f.read(1))

arr_st = np.stack(l)

print(f'Height: {arr_st.shape[1]}\nWidth: {arr_st.shape[2]}\nBands: {arr_st.shape[0]}')

ep.plot_bands(arr_st, cmap = 'gist_earth', figsize = (14, 12), cols = 3, cbar = False)
plt.show()


rgb = ep.plot_rgb(arr_st, 
                  rgb=(3,2,1), 
                  figsize=(12, 16), 
                  # title='RGB Composite Image'
                  )

plt.show()


ep.plot_rgb(
    arr_st,
    rgb=(3, 2, 1),
    stretch=True,
    str_clip=0.02,
    figsize=(12, 16),
    # title="RGB Composite Image with Stretch Applied",
)

plt.show()


colors = ['tomato', 'navy', 'MediumSpringGreen', 'lightblue', 'orange', 'blue',
          'maroon', 'purple', 'yellow', 'olive', 'brown', 'cyan']

ep.hist(arr_st, 
         colors = colors,
        title=[f'Band-{i}' for i in range(1, 13)], 
        cols=3, 
        alpha=0.5, 
        figsize = (12, 10)
        )

plt.show()

x = np.moveaxis(arr_st, 0, -1)
x.shape
     

x.reshape(-1, 12).shape, 954*298

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_data = x.reshape(-1, 12)

scaler = StandardScaler().fit(X_data)

X_scaled = scaler.transform(X_data)

X_scaled.shape


pca = PCA(n_components = 4)

pca.fit(X_scaled)

data = pca.transform(X_scaled)

data.shape

pca.explained_variance_ratio_


np.sum(pca.explained_variance_ratio_)


ep.plot_bands(np.moveaxis(data.reshape((689, 1200, data.shape[1])), -1, 0),
              cmap = 'gist_earth',
              cols = 2,
              figsize = (12, 6),
              title = [f'PC-{i}' for i in range(1,5)])

plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 10, random_state = 11, n_jobs= 6)

kmeans.fit(data)


labels = kmeans.predict(data)
np.unique(labels)


ep.plot_bands(labels.reshape(689, 1200), 
              figsize = (12, 16),
              cmap='Spectral_r')
plt.show()



import plotly.express as px

fig = px.imshow(labels.reshape(689, 1200), color_continuous_scale ='Spectral_r')

fig.update_xaxes(showticklabels=False)

fig.update_yaxes(showticklabels=False)

fig.update_layout(
    autosize=False,
    width=1200,
    height=698,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    # paper_bgcolor="LightSteelBlue",
)