''' CLEAN current version w/out ray'''
import skimage, glob, time, os
import numpy as np
from StatsFunctions4 import fProcessRed, fProcessBlue, fPearsonLst
import Predict_Functions as BP
import UMAP_v2 as u_me2
import pandas as pd
from skimage import io
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  
np.seterr(divide='ignore', invalid='ignore')

def fProcessMain(cFileName):
    img = skimage.io.imread(cFileName)
    ''' this bit rearranges the images if needed'''
    # blue = img[:,:,0]
    # red = img[:,:,1]
    # green = img[:,:,2]
    # gray = img[:,:,3]
    # img = np.stack([red, green, blue, gray], axis = 2)
    '''-----------------------------------------------'''
    global svm, knn, maxSize
    Tred, RedProp = fProcessRed(img)
    ''' RemoveBlueProp for fast testing'''
    BlueProp = fProcessBlue(img, speed = 10000)
    blueFail = 0.
    if (BlueProp[0] > RedProp[0]) or (BlueProp[0] < 10):
        BlueProp = [0]*6
        print(f"{cFileName[-8:]} -- blue fail")
        blueFail = str(cFileName[-8:])
    else:
        print(cFileName[-8:])
    PCB = fPearsonLst(img, thres = Tred, maxSize = maxSize,
          tileCounts = [0, 50, 32, 20, 10, 5])
    
    cLine = [cFileName[-8:], RedProp[0], RedProp[1], 
             RedProp[2], RedProp[3],
             RedProp[4], RedProp[5], RedProp[6],
             BlueProp[0], BlueProp[1],
             BlueProp[2], BlueProp[3],
             BlueProp[4], BlueProp[5],
             BlueProp[0]/RedProp[0], RedProp[7],
             PCB[0],
             PCB[1], PCB[2], PCB[3],
             PCB[4], PCB[5], 0, 0, 0,
             BP.run_dTree_3(RedProp[0], RedProp[1], RedProp[3], RedProp[4]),
             BP.run_kNN_1(knn, RedProp[0], RedProp[1], RedProp[3], RedProp[4]),
             0, 0, 0, 0, 0, 0]
    return cLine, blueFail

def fAddUMAP(res, newFileNameU, newFileName):
    trans1 = u_me2.setup_UMAP_a(dim = 1)
    trans2 = u_me2.setup_UMAP_a(dim = 2)
    cData = res[['Cell Area', 'Cell Perimeter', 'Cell Major Axis Length',
                   'Cell Minor Axis Length', 'Cell Circularity',
                   'Cell Eccentricity','Cell Extent', 'Nucleus Area',
                   'Nucleus Perimeter','Nucleus Major Axis Length',
                   'Nucleus Minor Axis Length','Nucleus Circularity',
                   'Nucleus Eccentricity', 'Nuc Cyt Ratio',
                   'Minkowski–Bouligand dimension', "Mean Pearson's r whole",
                   "Mean Pearson's r 50x50","Mean Pearson's r 32x32",
                   "Mean Persons's r 20x20", "Mean Person's r 10x10",
                   "Mean Person's r 5x5"]].values
    cData1 = u_me2.run_linComb(cData)
    umap2 = u_me2.run_UMAP_1(trans2, cData1)
    res["UMAP 1D"] = np.array(u_me2.run_UMAP_1(trans1, cData1))
    res["UMAP 2D X"] = np.array(umap2[:,0])
    res["UMAP 2D y"] = np.array(umap2[:,1])
    res.to_csv(newFileNameU, index = False, header = True)
    if os.path.exists(newFileName):
        os.remove(newFileName)    
    print("Basic and UMAP Stats printed to:", newFileNameU)
    return res

def main(newFileName, newFileNameU, folderName):
    lstFileNames = np.sort(glob.glob(folderName, recursive = True))
    maxSize = (0,0)
    maxFile = ""
    failed = []
    for cFileName in lstFileNames:
        img = skimage.io.imread(cFileName)
        if np.shape(img[:,:,0]) > maxSize:
            maxSize = np.shape(img[:,:,0])
            maxFile = cFileName
    globals()['maxSize'] = maxSize
    print("Stats v6")
    print("Largest image size:", maxSize)
    print("Largest image:", maxFile)
    print(f"Number of images: {len(lstFileNames)}")
    print("------------------------------------------------------------------")
    print("Basic Stats printing to:", newFileName)
    header = ["Name", "Cell Area", "Cell Perimeter", 
              "Cell Major Axis Length", 
              "Cell Minor Axis Length",
              "Cell Circularity", "Cell Eccentricity", 
              "Cell Extent", 
              "Nucleus Area", "Nucleus Perimeter",
              "Nucleus Major Axis Length", 
              "Nucleus Minor Axis Length",
              "Nucleus Circularity", 
              "Nucleus Eccentricity",
              "Nuc Cyt Ratio",
              "Minkowski–Bouligand dimension",
              "Mean Pearson's r whole",
              "Mean Pearson's r 50x50",
              "Mean Pearson's r 32x32",
              "Mean Persons's r 20x20",
              "Mean Person's r 10x10",
              "Mean Person's r 5x5",
              "UMAP 1D","UMAP 2D X", "UMAP 2D y",
              "Pred dTree",
              "Pred pKNN", "Pred mKNN3",
              "Pred mSVM3", "Pred mSVM4",
              "Pred mLR4",
              "Pred Ens10", "Acti"]
    tCSV = []
    cRow = []
    cFileName = "CellProp_9pM.csv"
    globals()['knn'] = BP.setup_kNN_1(cFileName)
    globals()['svm'] = BP.setup_SVM_1(cFileName)
    
    for cFileName in lstFileNames:
        cRow, blueFail = fProcessMain(cFileName)
        tCSV.append(cRow)
        if type(blueFail) == str: failed.append(blueFail)
        
    res = pd.DataFrame(tCSV, columns = header)
    res = res.replace([np.inf], 0)
    res.to_csv(newFileName, index = False, header = True)
    print("-"*50)
    print("Basic Stats printed to:", newFileName)
    print("Total Failed:", len(failed))
    return res, failed, newFileNameU
if __name__ == '__main__':
    startTime = time.time()
    newFileNameU = 'CellProperties15_UMAP.csv'
    newFileName = 'CellProperties15.csv'
    folderName = 'ProcessedPython5T\*.tif'
    res, failed, newFileNameU = main(newFileName, newFileNameU, folderName)
    resU = fAddUMAP(res, newFileNameU, newFileName)
    print(f"Run Time: {(time.time()-startTime)/60} sec")
    print("Stats v6 clean - no ray")
