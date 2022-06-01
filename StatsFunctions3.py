import cv2, skimage, scipy, csv, glob, os, time, multiprocessing, itertools
from math import sqrt, ceil, floor, isnan
from skimage import filters, io
from skimage.measure import label, regionprops
from PIL import Image
import numpy as np
from numpy import linalg, cov
from scipy.stats import pearsonr
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from multiprocessing import Pool, RawArray#, shared_memory

def fGenOtsu(img):
    csum = lambda z: np . cumsum (z )[: -1]
    dsum = lambda z: np . cumsum (z [:: -1])[ -2:: -1]
    argmax = lambda x , f: np . mean (x [: -1][ f == np . max (f )]) # Use the mean for ties .
    clip = lambda z:np.maximum(1e-30, z)
    def preliminaries (n , x ):
        """ Some math that is shared across each algorithm ."""
        assert np . all (n >= 0)
        x = np . arange ( len (n ), dtype =n. dtype ) if x is None else x
        assert np . all (x [1:] >= x [: -1])
        w0 = clip ( csum (n ))
        w1 = clip ( dsum (n ))
        p0 = w0 / ( w0 + w1 )
        p1 = w1 / ( w0 + w1 )
        mu0 = csum (n * x) / w0
        mu1 = dsum (n * x) / w1
        d0 = csum (n * x **2) - w0 * mu0 **2
        d1 = dsum (n * x **2) - w1 * mu1 **2
        return x , w0 , w1 , p0 , p1 , mu0 , mu1 , d0 , d1
    def GHT (n , x = None , nu =0 , tau =0 , kappa =0 , omega =0.5):
        """ Our generalization of the above algorithms ."""
        assert nu >= 0
        assert tau >= 0
        assert kappa >= 0
        assert omega >= 0 and omega <= 1
        x , w0 , w1 , p0 , p1 , _ , _ , d0 , d1 = preliminaries (n , x)
        v0 = clip (( p0 * nu * tau **2 + d0 ) / ( p0 * nu + w0 ))
        v1 = clip (( p1 * nu * tau **2 + d1 ) / ( p1 * nu + w1 ))
        f0 = -d0 / v0 - w0 * np . log ( v0 ) + 2 * ( w0 + kappa * omega ) * np . log ( w0 )
        f1 = -d1 / v1 - w1 * np . log ( v1 ) + 2 * ( w1 + kappa * (1 - omega )) * np . log ( w1 )
        return argmax (x , f0 + f1 ), f0 + f1
    hist, junk = np.histogram(img, bins = 256)
    return GHT(hist)

def fThresMin(image=None, nbins=256, max_iter=10000, *, hist=None):
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
            counts, bin_centers = skimage.exposure.histogram(image.ravel(), nbins, source_range='image')
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
    #make sure tileSize won't error out
    tiles =  []
    x,y = np.shape(img)
    if x%tileSize != 0:
        return "error"
    i = 0
    while i < y:
        j = 0
        while j < x:
            tile = img[i:i+tileSize, j:j+tileSize]
            tiles.append([(int(i/tileSize),int(j/tileSize)), tile])
            j+= tileSize
        i+=tileSize
    return tiles

def pad_img(img, maxSize):
    l = maxSize[0]
    an_array = np.array(img)
    shape = np.shape(an_array)
    padded_array = np.zeros((l, l, 4))
    padded_array[:shape[0],:shape[1]] = an_array
    
    return padded_array

def fPearsonCoeff(img1, img2):
    r,p = scipy.stats.pearsonr(img1.ravel(), img2.ravel())
    return r

def fPearsonCoeff3(img1, img2, Fred):
    #trying to remove values from initial array by multiplying by mask and 
    #removing all zero values before chunking into formula
    z = np.array(Fred*1)
    x = np.array(img1)
    y = np.array(img2)    
    # print("out")
    # print(len(x), len(y), len(z))
    # print(np.shape(x), np.shape(y), np.shape(z))
    x = np.where(x == 0, -1, x)
    y = np.where(y == 0, -1, y)
    x = x*z
    y = y*z 
    # print("cell only")
    # print(len(x), len(y), len(z))
    # print(np.shape(x), np.shape(y), np.shape(z))
    x = x[x != 0]
    y = y[y != 0]
    z = z[z != 0]
    x = np.where(x == -1, 0, x)
    y = np.where(y == -1, 0, y)
    # print("boolean")
    # print(len(x), len(y), len(z))
    # print(np.shape(x), np.shape(y), np.shape(z))
    if len(y) <= 2:
        # print("if")
        # print(len(x), len(y), len(z))
        # print("`````````````````")
        return 0.
    else:
        # print("else")
        # print(len(x), len(y), len(z))
        # print("`````````````````")
        r,p = scipy.stats.pearsonr(x,y)
    return r
    
def fPearsonWholeBlock(img1, img2, tileCount):
    tiles1 = fSplit(img1, tileCount)
    tiles2 = fSplit(img2, tileCount)
    tilesA = np.hstack((tiles1, tiles2))
    #tilesA = np.delete(tilesA, obj = 2, axis = 1)
    tilesA = tilesA.tolist()
    #print(tiles1[-1])
    bPearson = np.zeros((int(tiles1[-1][0][0])+1, int(tiles1[-1][0][1])+1))
    for cRow in tilesA:
       bPearson[cRow[0][0]][cRow[0][1]] = fPearsonCoeff(cRow[1], cRow[3])
    
    return bPearson

def fPearsonCellBlock(img1, img2, Fred, tileCount):
    if img1.shape[0]%tileCount != 0:
        return "error"
    tiles1 = fSplit(img1, tileCount)
    tiles2 = fSplit(img2, tileCount)
    tilesF = fSplit(Fred, tileCount)
    # print("1:", np.shape(tiles1))
    # print("2:", np.shape(tiles2))
    # print("F:", np.shape(tilesF))
   # print(np.shape(img1), np.shape(img2), np.shape(Fred))
    tilesA = np.concatenate((tiles1, tiles2, tilesF), axis = 1)
   # tilesA = np.delete(tilesA, obj = 2, axis = 1)
   # tilesA = np.delete(tilesA, obj = 3, axis = 1)
    tilesA = tilesA.tolist()
    bPearson = np.zeros((int(tiles1[-1][0][0])+1, int(tiles1[-1][0][1])+1))
    for cRow in tilesA:
      #print(np.shape(cRow[1]), np.shape(cRow[3]), np.shape(cRow[5]))
      #print(len(cRow[1]), len(cRow[3]), len(cRow[5]))
      #print(cRow[0])
      num = fPearsonCoeff3(cRow[1], cRow[3], cRow[5])
      if isnan(num):
          #print(cRow[0])
          bPearson[cRow[0][0]][cRow[0][1]] = 0
      else:
          bPearson[cRow[0][0]][cRow[0][1]] = num         
    return bPearson

def fFractalDim(img):
    #takes binary image ONLY
    #https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1
    Z = img
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])
    p = min(Z.shape)
    n = 2**np.floor(np.log(p)/np.log(2))
    n = int(np.log(n)/np.log(2))
    sizes = 2**np.arange(n, 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def fPearsonLst(img,thres, maxSize, tileCounts):
    '''Use 0 in the tileCounts to call Pearsons on the whole cell as one tile '''
    #ONLY USE Padded FOR PEARSONS STUFF                    
    imgPad = pad_img(img, maxSize)
    redPad = imgPad[:,:,0]
    greenPad = imgPad[:,:,1]
    # print(redPad)
    # print(greenPad)
    # bluePad = imgPad[:,:,2]
    # print("maxSize:", maxSize[0])
    #print("Red",np.shape(redPad), "Blue", np.shape(greenPad))
    #BgreenPad = greenPad > thres[1]
    BredPad = redPad > thres
    FredPad = scipy.ndimage.binary_fill_holes(BredPad)
    PCB = []
    for cTile in tileCounts:
        if cTile == 0:
            tile = np.shape(imgPad)[0]
            Mpear = float(fPearsonCellBlock(redPad, greenPad, FredPad, tile))
        else:  
            bPearson = fPearsonCellBlock(redPad, greenPad, FredPad, cTile)
            bPearson = bPearson[bPearson != 0]
            #bPearson = np.array(bPearson).astype(np.float)
            Mpear = np.mean(bPearson)
        PCB.append(Mpear)
    return PCB

def fPearsonLstMulti(img, thres, maxSize, tileCounts):
    '''Use 0 in the tileCounts to call Pearsons on the whole cell as one tile '''
    def fPearsonInternal(XredP_np, XgreenP_np, XFredP_np, cTile):
        redPad = np.frombuffer(XredP_np).reshape(XredP_np.shape)
        greenPad = np.frombuffer(XgreenP_np).reshape(XredP_np.shape)
        FredPad = np.frombuffer(XFredP_np).reshape(XredP_np.shape)
        
        if cTile == 0:
            tile = np.shape(imgPad)[0]
            Mpear = float(fPearsonCellBlock(redPad, greenPad, FredPad, tile))
        else:  
            bPearson = fPearsonCellBlock(redPad, greenPad, FredPad, cTile)
            bPearson = bPearson[bPearson != 0]
            Mpear = np.mean(bPearson)
        return Mpear
    #ONLY USE Padded FOR PEARSONS STUFF                    
    imgPad = pad_img(img, maxSize)
    redPad = imgPad[:,:,0]
    greenPad = imgPad[:,:,1]
    BredPad = redPad > thres
    FredPad = scipy.ndimage.binary_fill_holes(BredPad)*1
    XredP = RawArray('d', maxSize[0]*maxSize[1])
    XgreenP = RawArray('d', maxSize[0]*maxSize[1])
    XFredP = RawArray('i', maxSize[0]*maxSize[1])
    
    XredP_np = np.frombuffer(XredP).reshape(maxSize)
    XgreenP_np = np.frombuffer(XgreenP).reshape(maxSize)
    XFredP_np = np.frombuffer(XFredP).reshape(maxSize)
    
    np.copyto(XredP_np, redPad)
    np.copyto(XgreenP_np, greenPad)
    np.copyto(XFredP_np, FredPad)
    #print(i) #pass the arguments to fPearsonInternal with list of tuples here

    with Pool(processes = os.cpu_count()*2) as pool:
        PCB = pool.starmap(fPearsonInternal,
                           [(XredP_np, XgreenP_np, XFredP_np, j) 
                            for j in tileCounts])
    return PCB
def fPearsonLstMulti2(img, thres, maxSize, tileCounts):
    '''Use 0 in the tileCounts to call Pearsons on the whole cell as one tile '''

    imgPad = pad_img(img, maxSize)
    redPad = imgPad[:,:,0]
    greenPad = imgPad[:,:,1]
    BredPad = redPad > thres
    FredPad = scipy.ndimage.binary_fill_holes(BredPad)
    
    imgPad = imgPad.tolist()
    redPad = redPad.tolist()
    greenPad = greenPad.tolist()
    FredPad = FredPad.tolist()
    def fPearsonInternal(cTile):
        # redPad = np.frombuffer(XredP_np).reshape(XredP_np.shape)
        # greenPad = np.frombuffer(XgreenP_np).reshape(XredP_np.shape)
        # FredPad = np.frombuffer(XFredP_np).reshape(XredP_np.shape)
        
        # if cTile == 0:
        #     tile = np.shape(imgPad)[0]
        #     Mpear = float(fPearsonCellBlock(redPad, greenPad, FredPad, tile))
        # else:  
        #     bPearson = fPearsonCellBlock(redPad, greenPad, FredPad, cTile)
        #     bPearson = bPearson[bPearson != 0]
        #     Mpear = np.mean(bPearson)
        Mpear = 5*cTile
        return Mpear
    #ONLY USE Padded FOR PEARSONS STUFF                    

    # XredP = RawArray('d', maxSize[0]*maxSize[1])
    # XgreenP = RawArray('d', maxSize[0]*maxSize[1])
    # XFredP = RawArray('i', maxSize[0]*maxSize[1])
    
    # XredP_np = np.frombuffer(XredP).reshape(maxSize)
    # XgreenP_np = np.frombuffer(XgreenP).reshape(maxSize)
    # XFredP_np = np.frombuffer(XFredP).reshape(maxSize)
    
    # np.copyto(XredP_np, redPad)
    # np.copyto(XgreenP_np, greenPad)
    # np.copyto(XFredP_np, FredPad)
    #print(i) #pass the arguments to fPearsonInternal with list of tuples here
    pool = multiprocessing.Pool(processes=os.cpu_count())
    PCB = pool.map(fPearsonInternal, tileCounts)                        
    return PCB

def fProcessRed(img, TredP, RedPropP):
    red = img[:, :, 0]
    Tred = skimage.filters.threshold_li(red)
    Bred = red > Tred
    Fred = scipy.ndimage.binary_fill_holes(Bred)
    RedPropi = regionprops(Fred*1)
    RedCircularity = (4*RedPropi[0]['area']*np.pi)/(RedPropi[0]['perimeter']**2)
    TredP.value = float(Tred)
    RedPropP[:] = [RedPropi[0]['area'], RedPropi[0]['Perimeter'], 
                RedPropi[0]['MajorAxisLength'], RedPropi[0]['MinorAxisLength'],
                RedCircularity, RedPropi[0]['Eccentricity'], RedPropi[0]['euler_number'],
                RedPropi[0]['extent'], fFractalDim(Fred)]
    #print("Result(in process pRed): {}".format(RedPropt[:]))
    return
def fProcessGreen(img, TgreenP):
    green = img[:, :, 1]
    TgreenP.value = float(skimage.filters.threshold_triangle(green))
    #Bgreen = green > Tgreen
    return
    
def fProcessBlue(img, BluePropP):
        blue = img[:, :, 2]
        Tblue = fThresMin(blue)
        Bblue = blue > Tblue
        Fblue = scipy.ndimage.binary_fill_holes(Bblue)
        l = label(Fblue)
        Fblue = (l==np.bincount(l.ravel())[1:].argmax()+1).astype(int)
        BluePropi = regionprops(Fblue*1)
        BlueCircularity = (4*BluePropi[0]['area']*np.pi)/(BluePropi[0]['perimeter']**2)
        BluePropP[:] = [BluePropi[0]['area'], BluePropi[0]['perimeter'],
                                        BluePropi[0]['MajorAxisLength'],
                                        BluePropi[0]['MinorAxisLength'],
                                        BlueCircularity,
                                        BluePropi[0]['Eccentricity']]
        return