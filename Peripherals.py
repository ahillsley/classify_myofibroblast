import skimage, scipy
import numpy as np
from matplotlib import pyplot as plt
from StatsFunctions3 import fSplit, fPearsonCellBlock
from StatsFunctions3 import fPearsonWholeBlock

def fPearsonHeatmap(fileName, tileCounts, save = False):
    #makes heatmap of pearsons coeff for up to 3 tileCounts
    img = skimage.io.imread(fileName)
    red = img[:,:,0]
    green = img[:,:,1] 
    Tred = skimage.filters.threshold_li(red)
    Bred = red > Tred
    Fred = scipy.ndimage.binary_fill_holes(Bred) 
    wholeR = []
    cellR = []
    for tileCount in tileCounts:    
        wholeR.append(fPearsonWholeBlock(red, green, tileCount))
        cellR.append(fPearsonCellBlock(red, green, Fred, tileCount))
    fig, axs = plt.subplots(2, 4)
    ax1 = axs[0,0]
    ax1.pcolormesh(red, cmap = "Reds", vmin = 600, vmax = 2000)
    ax1.axis("off")
    ax1.set_title("Red")
    ax5 = axs[1,0]
    ax5.pcolormesh(green, cmap = "Greens", vmin = 600, vmax = 4200)
    ax5.axis("off")
    ax5.set_title("Green")
    i = 1
    while i < 4:
        cellRt = cellR[i-1]
        cellRt = cellRt[cellRt != 0]
        cellAvg = str(round(np.mean(cellRt), 4))
        titleC = "Cell " + str(tileCounts[i-1]) + "\n Avg: " + cellAvg
        titleW = "Whole " + str(tileCounts[i-1])
        axs[0,i].imshow(wholeR[i-1])
        axs[0,i].set_title(titleW)
        axs[0,i].axis("off")
        axs[1,i].imshow(cellR[i-1])
        axs[1,i].set_title(titleC)
        axs[1,i].axis("off")
        i+=1
    if save == True:
        sFileName = fileName.split(".")
        nFileName = f"{sFileName[0]}_Pearson2_{tileCounts[0]},{tileCounts[1]},{tileCounts[2]}.png"
        fig.savefig(nFileName, box_inches = "tight", pad_inches = 0.1)
        print("File saved to:", nFileName)
                                
    plt.show()
    #return cellR
def fShowTiles(fileName, tileSize, save = False):
    img = skimage.io.imread(fileName)
    gray = img[:,:,3]
    red = img[:,:,0]
    green = img[:,:,1]
    Tred = skimage.filters.threshold_li(red)
    Bred = red > Tred
    Fred = scipy.ndimage.binary_fill_holes(Bred)
    tiles = fSplit(gray, tileSize)
    bPearson = fPearsonCellBlock(red, green, Fred, tileSize)
    fig, axs = plt.subplots(tiles[-1][0][0]+1, tiles[-1][0][1]+1)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    for cRow in tiles:
        r = cRow[0][0]
        c = cRow[0][1]
        title = str(round(bPearson[c][r], 4))
        axs[r,c].imshow(cRow[1], cmap = "gray")
        axs[r,c].set_title(title, fontsize = 24)
        axs[r,c].axis("off")
    plt.show()
    if save:
        sFileName = fileName.split(".")
        nFileName = f"{sFileName[0]}_tiled{tileSize}.png"
        fig.savefig(nFileName, bbox_inches = "tight", pad_inches = 0.1)
        print("File saved to:", nFileName)
    return bPearson