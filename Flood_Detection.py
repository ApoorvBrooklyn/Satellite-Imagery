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

np.seterr(divide='ignore', invalid='ignore')


before_floods_data_path = r"C:\Users\HP\Desktop\Final EL\Satellite_Imagery_Analysis-main\Data\Madagascar_data\Madagascar_18_01_2017"
during_floods_data_path = r"C:\Users\HP\Desktop\Final EL\Satellite_Imagery_Analysis-main\Data\Madagascar_data\Madagascar_27_01_2020"

before_floods = glob(r"C:\Users\HP\Desktop\Final EL\Satellite_Imagery_Analysis-main\Data\Madagascar_data\Madagascar_18_01_2017/*B?*.tiff")
before_floods.sort()
before_floods


l = []
for i in before_floods:
  with rio.open(i, 'r') as f:
    l.append(f.read(1))

arr_bef = np.stack(l)

arr_bef.shape
#print(arr_bef.shape)

ep.plot_bands(arr_bef, cmap='Spectral_r', cols=3, figsize=(10, 10), cbar=False)
plt.show()


ep.plot_rgb(arr_bef, rgb=(3, 2, 1), figsize=(10, 10))
plt.show()

ndwi_bef = es.normalized_diff(arr_bef[5, :, :], arr_bef[7, :, :])

ep.plot_bands(ndwi_bef, cmap='coolwarm_r', vmin=0, vmax=1)
plt.show()

during_floods = glob(r"C:\Users\HP\Desktop\Final EL\Satellite_Imagery_Analysis-main\Data\Madagascar_data\Madagascar_27_01_2020/*B?*.tiff")
during_floods.sort()
during_floods

dl = []
for i in during_floods:
  with rio.open(i, 'r') as f:
    dl.append(f.read(1))

arr_dur = np.stack(dl)

arr_dur.shape


ep.plot_bands(arr_dur, cmap='Spectral_r', cols=3, figsize=(10, 10), cbar=False)
plt.show()

ep.plot_rgb(arr_dur, rgb=(3, 2, 1), figsize=(10, 10))

plt.show()
ndwi_dur = es.normalized_diff(arr_dur[5, :, :], arr_dur[7, :, :])


mask_bef = (ndwi_bef > 0.6).astype(int)

ep.plot_bands(mask_bef, cmap='Greys_r', figsize=(12, 12))
plt.show()

# Mask the data into water and non water pixels based on a threshold value(0.6)

mask_dur = (ndwi_dur > 0.6).astype(int)

ep.plot_bands(mask_dur, cmap='Greys_r', figsize=(12, 12))
plt.show()

# Mask the data into water and non water pixels based on a threshold value(0.6)

mask_dur = (ndwi_dur > 0.6).astype(int)

ep.plot_bands(mask_dur, cmap='Greys_r', figsize=(12, 12))
plt.show()

from matplotlib.colors import colorConverter
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# create dummy data
zvals = ndwi_dur
zvals2 = mask_bef

# generate the colors for your colormap
color1 = colorConverter.to_rgba('white')
color2 = colorConverter.to_rgba('black')

# make the colormaps
cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['green','blue'],256)
cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)

cmap2._init() # create the _lut array, with rgba values

# create your alpha array and fill the colormap with them.
# here it is progressive, but you can create whathever you want
alphas = np.linspace(0, 0.8, cmap2.N+3)
cmap2._lut[:,-1] = alphas

fig = plt.figure(figsize=(12, 12)) 
plt.imshow(zvals, interpolation='nearest', cmap= cmap1, )
plt.imshow(zvals2, interpolation='nearest', cmap=cmap2, label='flood')
plt.colorbar()
plt.axis('off')
plt.show()

fig = plt.figure(figsize=(10, 10))
plt.imshow(ndwi_bef, vmin=0, vmax=1, cmap='gray_r', interpolation='none')

plt.axis('off')
plt.show()

fig = plt.figure(figsize=(12, 12))

plt.imshow(mask_bef, vmin=0, vmax=1, cmap='Reds', interpolation='none')
plt.imshow(ndwi_bef, vmin=0, vmax=1, cmap='gist_earth_r', interpolation='none', alpha=0.5)

plt.axis('off')
plt.show()


from matplotlib.colors import ListedColormap
fig = plt.figure(figsize=(12, 12))

rgb = np.moveaxis(np.stack([dl[3], dl[2], dl[1]]), 0, -1)
Image = rgb/np.amax(rgb)
Image = np.clip(Image, 0, 1)


plt.imshow(Image, interpolation='none')
plt.imshow(mask_bef, vmin=0, vmax=1, 
           cmap=ListedColormap(['#ffffff00', '#00FF33']), 
           interpolation='none' , alpha=0.5)

plt.axis('off')
# plt.savefig('flood_result.png', dpi=400)

plt.show()