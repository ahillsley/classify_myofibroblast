import skimage
import scipy
from math import isnan
from skimage.measure import label, regionprops
from skimage import io, filters
import numpy as np
from numpy import linalg
from scipy import ndimage as ndi


def fThresMin(image=None, nbins=256, max_iter=10000, *, hist=None):
    # Returns the threshold of a grayscale image based on the minimum method.
    # Modified from skimage.filters.threshold_minimum
    def _validate_image_histogram(image, hist, nbins=None):
        if image is None and hist is None:
            raise Exception("Either image or hist must be provided.")
        if hist is not None:
            if isinstance(hist, (tuple, list)):
                counts, bin_centers = hist
            else:
                counts = hist
                bin_centers = np.arange(counts.size)
        else:
            counts, bin_centers = skimage.exposure.histogram(
                image.ravel(), nbins, source_range='image')
        return counts.astype(float), bin_centers

    def find_local_maxima_idx(hist):
        # We can't use scipy.signal.argrelmax
        # as it fails on plateaus
        maximum_idxs = list()
        direction = 1
        for i in range(hist.shape[0] - 1):
            if direction > 0:
                if hist[i + 1] < hist[i]:
                    direction = -1
                    maximum_idxs.append(i)
            else:
                if hist[i + 1] > hist[i]:
                    direction = 1
        return maximum_idxs
    counts, bin_centers = _validate_image_histogram(image, hist, nbins)
    smooth_hist = counts.astype(np.float64, copy=False)
    for counter in range(max_iter):
        smooth_hist = ndi.uniform_filter1d(smooth_hist, 3)
        maximum_idxs = find_local_maxima_idx(smooth_hist)
        if len(maximum_idxs) < 3:
            break
    if len(maximum_idxs) != 2:
        return skimage.filters.threshold_otsu(image)
    elif counter == max_iter - 1:
        raise RuntimeError('Maximum iteration reached for histogram'
                           'smoothing')
    # Find lowest point between the maxima
    threshold_idx = np.argmin(smooth_hist[maximum_idxs[0]:maximum_idxs[1] + 1])
    return bin_centers[maximum_idxs[0] + threshold_idx]


def fSplit(img, tileSize):
    # Splits image into square tiles with side length of tileSize
    tiles = []
    x, y = np.shape(img)
    if x % tileSize != 0:
        return "error"
    i = 0
    while i < y:
        j = 0
        while j < x:
            tile = img[i:i+tileSize, j:j+tileSize]
            tiles.append([(int(i/tileSize), int(j/tileSize)), tile])
            j += tileSize
        i += tileSize
    return tiles


def pad_img(img, maxSize):
    # Pads the image to make it into a square with side length of maxSize
    length = maxSize[0]
    an_array = np.array(img)
    shape = np.shape(an_array)
    padded_array = np.zeros((length, length, 4))
    padded_array[:shape[0], :shape[1]] = an_array
    return padded_array


def fPearsonCoeff3(img1, img2, Fred):
    # Calculates Pearson's correlation coefficient between img1 and img2,
    # provided the pixels are also in Fred.
    z = np.array(Fred*1)
    x = np.array(img1)
    y = np.array(img2)

    # Determine if the individual pixels in img1 and img2 are in Fred.
    x = np.where(x == 0, -1, x)
    y = np.where(y == 0, -1, y)
    x = x*z
    y = y*z
    x = x[x != 0]
    y = y[y != 0]
    z = z[z != 0]
    x = np.where(x == -1, 0, x)
    y = np.where(y == -1, 0, y)

    if len(y) <= 2:
        return 0.
    else:
        x = np.asarray(x)
        y = np.asarray(y)
        dtype = type(1.0 + x[0] + y[0])
        xmean = x.mean(dtype=dtype)
        ymean = y.mean(dtype=dtype)
        xm = x.astype(dtype) - xmean
        ym = y.astype(dtype) - ymean
        normxm = linalg.norm(xm)
        normym = linalg.norm(ym)
        r = np.dot(xm/normxm, ym/normym)
        r = max(min(r, 1.0), -1.0)
    return r


def fPearsonCellBlock(img1, img2, Fred, tileSize):
    # Returns a matrix of Pearson's correlation for each tile in img1 and img2
    if img1.shape[0] % tileSize != 0:
        return "error"
    tiles1 = fSplit(img1, tileSize)
    tiles2 = fSplit(img2, tileSize)
    tilesF = fSplit(Fred, tileSize)
    tilesA = np.concatenate((tiles1, tiles2, tilesF), axis=1)
    tilesA = tilesA.tolist()
    bPearson = np.zeros((int(tiles1[-1][0][0])+1, int(tiles1[-1][0][1])+1))
    for cRow in tilesA:
        num = fPearsonCoeff3(cRow[1], cRow[3], cRow[5])
        if isnan(num):
            bPearson[cRow[0][0]][cRow[0][1]] = 0.
        else:
            bPearson[cRow[0][0]][cRow[0][1]] = num
    return bPearson


def fFractalDim(img):
    # Calculates the Minkowski–Bouligand dimension for image img
    # takes binary image ONLY
    # https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])
    Z = img
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


def fPearsonLst(img, thres, maxSize, tileSizes):
    # Calculates the mean Pearson's Correlation Coefficient for the tile size.
    # tileSizes must be a list
    # Put a 0 in the tileSizes to call Pearsons on the whole cell as one tile
    imgPad = pad_img(img, maxSize)
    redPad = imgPad[:, :, 0]
    greenPad = imgPad[:, :, 1]
    BredPad = redPad > thres
    FredPad = scipy.ndimage.binary_fill_holes(BredPad)
    PCB = []
    for cTile in tileSizes:
        if cTile == 0:
            tile = np.shape(imgPad)[0]
            Mpear = float(fPearsonCellBlock(redPad, greenPad, FredPad, tile))
        else:
            bPearson = fPearsonCellBlock(redPad, greenPad, FredPad, cTile)
            bPearson = bPearson[bPearson != 0]
            Mpear = np.mean(bPearson)
        PCB.append(Mpear)
    return PCB


def fProcessRed(img):
    # Filter and calculate properties of the cell cytoskeleton.
    red = img[:, :, 0]
    Tred = skimage.filters.threshold_li(red)
    Bred = red > Tred
    Fred = scipy.ndimage.binary_fill_holes(Bred)
    lb = label(Fred)
    Fred = (lb == np.bincount(lb.ravel())[1:].argmax()+1).astype(int)
    RedPropi = regionprops(Fred*1)
    RedCircularity = (4*RedPropi[0]['area']*np.pi) / \
        (RedPropi[0]['perimeter']**2)
    RedProp = [RedPropi[0]['area'], RedPropi[0]['Perimeter'],
               RedPropi[0]['MajorAxisLength'], RedPropi[0]['MinorAxisLength'],
               RedCircularity, RedPropi[0]['Eccentricity'],
               RedPropi[0]['extent'], fFractalDim(Fred)]
    return Tred, RedProp


def fProcessBlue(img, speed=2000):
    # Filter and calculate properties of the cell nucleus.
    blue = img[:, :, 2]
    Tblue = fThresMin(blue, max_iter=speed)
    Bblue = blue > Tblue
    Fblue = scipy.ndimage.binary_fill_holes(Bblue)
    lb = label(Fblue)
    Fblue = (lb == np.bincount(lb.ravel())[1:].argmax()+1).astype(int)
    BluePropi = regionprops(Fblue*1)
    BlueCircularity = (4*BluePropi[0]['area']*np.pi) / \
        (BluePropi[0]['perimeter']**2)
    BlueProp = [BluePropi[0]['area'], BluePropi[0]['perimeter'],
                BluePropi[0]['MajorAxisLength'],
                BluePropi[0]['MinorAxisLength'],
                BlueCircularity,
                BluePropi[0]['Eccentricity']]
    return BlueProp
