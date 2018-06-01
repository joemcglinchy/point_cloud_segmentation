import os, sys
from skimage.segmentation import slic
from skimage.segmentation import felzenszwalb
from skimage.filters import gaussian
import rasterio
import numpy as np
from matplotlib import pyplot as plt

def roll_first_axis_to_back(arr):
    temp = np.rollaxis(arr,0,2)
    return np.rollaxis(temp,2,1)

def plot2(im1, im2):
    plt.subplot(1,2,1)
    plt.imshow(im1)
    plt.subplot(1,2,2)
    # plt.imshow(im2,cmap='Accent_r')
    plt.imshow(im2)
    plt.show()
    return



# test
from skimage.data import astronaut
img = astronaut()
segments = slic(img, n_segments=100, compactness=10)

# try a NEON image
neon_fi = r"C:\Projects\RD\objectid\NEON_ORNL_rgb\16052313_EH021656(20160523135722)-0001_ort.tif"
with rasterio.open(neon_fi, 'r+') as fi:
    neon_im = fi.read()
    
print(neon_im.shape)
print(roll_first_axis_to_back(neon_im).shape)

# use the arrays for plotting
temp = roll_first_axis_to_back(neon_im)
# neon_segments = slic(temp, n_segments=100, compactness=10)

# work on a chip
chip = temp[3000:5000,2000:4000,:]
# chip_segs = slic(chip, convert2lab=True, compactness=.5, max_iter=100)
# plt.subplot(1,2,1)
# plt.imshow(chip)
# plt.subplot(1,2,2)
# plt.imshow(chip_segs)
# plt.show()

img = gaussian(chip, 2)
fz = felzenszwalb(img, scale=100, sigma=0.8, min_size=20, multichannel=True)
plot2(chip, fz)

# plt.subplot(1,2,1)
# plt.imshow(chip)
# plt.subplot(1,2,2)
# plt.imshow(fz,cmap='Accent')
# plt.show()

