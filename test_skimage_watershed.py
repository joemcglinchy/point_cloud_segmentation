import rasterio
import os,sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filters import gaussian
from skimage.measure import regionprops

demo = False

if demo:
    # Generate an initial image with two overlapping circles
    x, y = np.indices((80, 80))
    x1, y1, x2, y2 = 28, 28, 44, 52
    r1, r2 = 16, 20
    mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
    mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
    image = np.logical_or(mask_circle1, mask_circle2)

    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(image)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=image)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers, mask=image)

    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()


dtm_name = '2016_ORNL_2_736000_3981000_DTM.tif'
dsm_name = '2016_ORNL_2_736000_3981000_DSM.tif'
dsm_dir = r'C:\Projects\RD\objectid\elev'

fi_name = os.path.join(dsm_dir, dsm_name)

with rasterio.open(fi_name) as fi:
    dsm = fi.read()
 
fi_name = os.path.join(dsm_dir, dtm_name) 
with rasterio.open(fi_name) as fi:
    dtm = fi.read()
    
print(dsm.squeeze().shape)
ndsm = dsm.squeeze() -  dtm.squeeze()

# crop the middle
width = 200
r,c = ndsm.shape
r_c = int(np.floor(r/2))
c_c = int(np.floor(c/2))
ndsm_chip = ndsm[r_c - width/2 : r_c + width/2, c_c - width/2 : c_c + width/2]
ndsm_chip_sm = gaussian(ndsm_chip/ndsm_chip.max(), sigma=0.4)

max_search = 3
# get local maxima as image coordinates [rows, cols]
l_max = peak_local_max(ndsm_chip_sm, min_distance = max_search)

# get local maxima as image
l_max_im = peak_local_max(ndsm_chip_sm, min_distance = max_search, indices=False)

# threshold image
ndsm_chip_bin = ndsm_chip_sm*ndsm.max() > 6

 
distance = ndi.distance_transform_edt(ndsm_chip_bin)

print(distance.shape)
# plt.imshow(distance)
# plt.show()

# get local max as image masked by the binary image
#local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=ndsm_chip_bin)
local_maxi = peak_local_max(ndsm_chip_sm, indices=False, footprint=np.ones((3, 3)), labels=ndsm_chip_bin)

# label the maximum as markers 
markers = ndi.label(local_maxi)[0]

# get the segments using watershed algorithm. (input in example is -distance, but the distance image looks like garbage in a heavily forested environment)
# labels = watershed(-distance, markers, mask=image)
labels = watershed(-ndsm_chip, markers, mask=ndsm_chip_bin)

plot_flag = False
if plot_flag:
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True,
                             subplot_kw={'adjustable': 'box-forced'})
    ax = axes.ravel()

    ax[0].imshow(ndsm_chip, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(ndsm_chip_sm, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
    plt.scatter(l_max[:,1], l_max[:,0])
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()


# sample the dsm with the segments
label_props = regionprops(labels, intensity_image=ndsm_chip)


# create local maxima images iteratively by varying max_search
cands = []
for max_search in range(1,20):
    cands.append(peak_local_max(ndsm_chip_sm, min_distance = max_search, indices=False, labels = ndsm_chip_bin).astype('float32'))
    
# convert list into numpy array and sum
cands_arr = np.array(cands)
cands_sum = np.sum(cands_arr, axis=0)
del cands

# run watershed workflow
thresh=5
sum_max = cands_sum > thresh
markers = ndi.label(sum_max)[0]
labels = watershed(-ndsm_chip_sm, markers, mask=ndsm_chip_bin)
