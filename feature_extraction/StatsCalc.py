import skimage
import glob
import time
import numpy as np
from StatsCalc_HelperFunctions import fProcessRed, fProcessBlue, fPearsonLst
from joblib import load
import Predict_Functions as PF
import pandas as pd
from skimage import io
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
np.seterr(divide='ignore', invalid='ignore')


newFileName = '__' # output file path
folderName = "__" # path to image folder
pKNNpath = "__" # location of pKNN.joblib

def fProcessMain(cFileName, knn):
    # Calculates properties of the image, designated by cFileName,
    # Outputs: cLine (image properties, dTree and kNN_1 prediction, padding),
    # blueFail (name of image, in case the image isn't thresholded properly)
    img = skimage.io.imread(cFileName)
    Tred, RedProp = fProcessRed(img)
    BlueProp = fProcessBlue(img, speed=10000)
    blueFail = 0.
    if (BlueProp[0] > RedProp[0]) or (BlueProp[0] < 10):
        BlueProp = [0]*6
        print(f"{cFileName[-8:]} -- blue fail")
        blueFail = str(cFileName[-8:])
    else:
        print(cFileName[-8:])
    PCB = fPearsonLst(img, thres=Tred, maxSize=(1024, 1024),
                      tileSizes=[0, 64, 32, 16, 8, 4])
    cLine = [cFileName[-8:], RedProp[0], RedProp[1],
             RedProp[2], RedProp[3], RedProp[4], RedProp[5], RedProp[6],
             BlueProp[0], BlueProp[1], BlueProp[2], BlueProp[3],
             BlueProp[4], BlueProp[5],
             BlueProp[0]/RedProp[0], RedProp[7],
             PCB[0], PCB[1], PCB[2], PCB[3], PCB[4], PCB[5],
             PF.run_dTree(RedProp[0], RedProp[1], RedProp[3], RedProp[4]),
             PF.run_kNN(knn, RedProp[0], RedProp[1], RedProp[3], RedProp[4]),
             0]
    return cLine, blueFail


def main(newFileName, folderName, knn):
    # newFileNameU is the final saving CSV
    # folderName is the path to the folder where all the images are saved
    lstFileNames = np.sort(glob.glob(folderName, recursive=True))
    failed = []
    print("Cell Statistics")
    print(f"Number of images: {len(lstFileNames)}")
    print("-"*66)
    print("Stats printing to:", newFileName)
    header = ["Name", "Cell Area", "Cell Perimeter",
              "Cell Major Axis Length", "Cell Minor Axis Length",
              "Cell Circularity", "Cell Eccentricity", "Cell Extent",
              "Nucleus Area", "Nucleus Perimeter", "Nucleus Major Axis Length",
              "Nucleus Minor Axis Length", "Nucleus Circularity",
              "Nucleus Eccentricity", "Nuc Cyt Ratio",
              "Minkowski–Bouligand dimension",
              "Mean Pearson's r whole", "Mean Pearson's r 64x64",
              "Mean Pearson's r 32x32", "Mean Pearson's r 16x16",
              "Mean Pearson's r 8x8", "Mean Pearson's r 4x4",
              'PredDTree', 'PredPKNN', 'PredMSVM4']
    tCSV = []
    cRow = []

    # Calculate properties and dTree/kNN predictions for all images
    for cFileName in lstFileNames:
        cRow, blueFail = fProcessMain(cFileName, knn)
        tCSV.append(cRow)
        if type(blueFail) == str:
            failed.append(blueFail)

    # save properties and dTree/kNN predictions to CSV
    res = pd.DataFrame(tCSV, columns=header)
    res = res.replace([np.inf], 0)
    res.to_csv(newFileName, index=False, header=True)
    print("-"*66)
    print("Total Failed:", len(failed))
    return res, failed


if __name__ == '__main__':
    # newFileName is the final saving CSV
    # folderName is the path to the folder where all the images are saved
    # pKNNpath is the path to pKNN.joblib
    startTime = time.time()
    newFileName = 'CellProperties18_TestImg.csv'
    folderName = "G:\\Shared drives\\000_College\\Rosales Lab\\Images\\Test\*.tif"
    pKNNpath = "G:\\Shared drives\\000_College\\Rosales Lab\\Cleaned up Code\\pKNN.joblib"
    
    knn = load(pKNNpath)
    res, failed = main(newFileName, folderName, knn)
    print("Stats printed to:", newFileName)
    print(f"Run Time: {round((time.time()-startTime)/3600,5)} hours")
