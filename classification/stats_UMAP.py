import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap as umap
import random
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from itertools import combinations
random.seed(4)
np.random.seed(4)

stats = pd.read_csv('Cellproperties16_UMAP.csv')
#stats = pd.read_csv('../Final_things/Cell_properties_um.csv')
#stats = stats.iloc[:1104,:]
key = pd.read_csv('labels_new.csv', header=None)
key = key.iloc[:1104,:]
cEmb = stats.iloc[:,1:22]
unsuper = pd.read_csv('../../Code/BYOL/coordinates.csv')
byol_37 = pd.read_csv('../../Final_things/old_37_byol_coords.csv', header=None)
byol_37_emb = np.load('../../Final_things/37_byol_emb.npy')
uns_coords = unsuper.iloc[:,2:]


a80stats = stats[['CellArea', 'CellPerimeter', "CellMinorAxisLength", "CellMajorAxisLength",
                "NucCytRatio", "MeanPearson_sR64x64", "NucleusArea",
               "NucleusMajorAxisLength","NucleusMinorAxisLength", "NucleusPerimeter", "CellCircularity"]]

cshape_stats = stats[['CellArea', 'CellPerimeter', "CellMinorAxisLength", 
                  "CellCircularity"]]

def emb_to_tUMAP(cEmb, nn, mind, rs, metric, ncomp):
    ''' creates trained UMAP from embedding'''
    cEmb = np.asarray(cEmb)
    random.seed(4)
    np.random.seed(4)
    reducer = umap.UMAP(n_neighbors = nn, min_dist = mind,
                        metric = metric,
                        random_state = rs, n_components = ncomp)
    trans = reducer.fit(cEmb)
    coords = trans.transform(cEmb)
    return coords, trans

def plot_umap_2d(coords, key):
    coords = np.asarray(coords)
    x1 = np.arange(coords[:,0].min()-0.1,coords[:,0].max()+0.1,0.01)
    slope, intercept, direction = fit_line(coords, key)
    fig, ax1 = plt.subplots()
    ax1.scatter(
         coords[:, 0],
         coords[:,1],
         #c=[sns.color_palette()[x] for x in key], 
         c = key, cmap = 'OrRd',
         s=20, alpha = 0.75,
         vmax = 0.5)
    #ax1.plot(x1, slope*x1 +intercept, 'k-', lw=3)
    ax1.set_xbound(coords[:,0].min()-1, coords[:,0].max()+1)
    ax1.set_ybound(coords[:,1].min()-1, coords[:,1].max()+1)
    plt.xlabel('UMAP 1', labelpad=5)
    plt.ylabel('UMAP 2', labelpad=5)
    plt.tick_params(bottom = False, labelbottom = False, left=False, labelleft=False)
    plt.gca().set_aspect(1.25)
    del coords
    #fig.savefig(f"../Final_things/Figures/UMAPs/UMAP_cshape_cont.png", dpi = 600, bbox_inches = 'tight')
    return direction
    
def fit_line(coordinates, key):
    '''
    coordinates needs to be in form: X , Y
    '''
    key = np.asarray(key)
    coordinates = np.asarray(coordinates)
    coords = np.zeros((coordinates.shape[0], coordinates.shape[1]+1))
    coords[:,:2] = coordinates[:,:]
    coords[:,2] = key
    
    acti = coords[coords[:,2]==1]
    non  = coords[coords[:,2]==0]
    avg_x_acti = np.mean(acti[:,0])
    avg_y_acti = np.mean(acti[:,1])
    avg_x_non = np.mean(non[:,0])
    avg_y_non = np.mean(non[:,1])
    
    slope = (avg_y_acti - avg_y_non) / (avg_x_acti - avg_x_non)
    intercept = avg_y_acti - (slope*avg_x_acti)
    if np.greater(avg_x_acti, avg_x_non)==True:
        direction = '100 = Activated'
    else:
        direction = '100 = Non-activated'
    
    return slope, intercept, direction

def project2line(coords, key):
    coords = np.asarray(coords)
    slope, intercept, direction = fit_line(coords, key)
    projection = np.zeros(coords.shape)
    m_orth = -1 / slope
    for i in range(len(coords)):
        x = coords[i,0]
        y = coords[i,1]
        b_orth = y - (m_orth*x)
        x_inter = (b_orth - intercept) / (slope - m_orth)
        y_inter = slope*x_inter + intercept
        projection[i,0], projection[i,1] = x_inter, y_inter 
    
    return projection

def cont_labels(projection, size=1000):
    labels = projection[:,0] # only need X values
    scale_labels = (labels - np.min(labels))/(np.max(labels) - np.min(labels))
    
    return np.rint(scale_labels*size)
    
    

def run_linComb(cell_data):
    cell_data = StandardScaler().fit_transform(cell_data)
    i = 0
    lList = []
    while i < np.shape(cell_data)[1]:
        lList.append(cell_data[:,i])
        i+=1
    lcomb = list(combinations(lList, 2))
    flcomb = []
    for cRow in lcomb: flcomb.append(np.multiply(cRow[0], cRow[1]))
    flcomb = np.transpose(flcomb)
    return np.concatenate((flcomb, cell_data), axis = 1)
    
    

def from_stats():
    '''
    to generate labels from cell stats
    '''
    cEmb = cshape_stats
    lin_comb = run_linComb(cEmb)
    
    ''' Below are good UMAPs for all 22 props'''
    #coords, trans = emb_to_tUMAP(lin_comb, nn=50, mind=1, rs=7, metric='correlation', ncomp=2)
    #coords, trans = emb_to_tUMAP(cEmb, nn=50, mind=0.75, rs=6, metric='correlation', ncomp=2)
    ''' Below are good UMAPs for only props with AUC above 80%  a80stats'''
    #coords, trans = emb_to_tUMAP(lin_comb, nn=100, mind=0.25, rs=2, metric='correlation', ncomp=2)
    #coords, trans = emb_to_tUMAP(lin_comb, nn=100, mind=0.75, rs=6, metric='correlation', ncomp=2)
    
    ''' Below are good UMAPs for only cell shape props cshape_stats '''
    #coords, trans = emb_to_tUMAP(lin_comb, nn=200, mind=1, rs=6, metric='manhattan', ncomp=2)
    coords, trans = emb_to_tUMAP(lin_comb, nn=200, mind=1, rs=9, metric='manhattan', ncomp=2)
    
    direction = plot_umap_2d(coords, key.iloc[:,2])
    #direction = plot_umap_2d(coords, mcontlabels[:1104]) #need to adjust vmin / vmax
    
    print(direction)
    projection = project2line(coords, key.iloc[:,2])
    labels = cont_labels(projection, size=1000)
    
    return labels
def plot_highlight_cells():
    cell_1 = np.expand_dims(coords[750,:],0) # low acti cell# 751
    cell_2 = np.expand_dims(coords[863,:],0) # mid acti cell# 864
    cell_3 = np.expand_dims(coords[595,:],0) # high acti cell# 596
    high_cells = np.concatenate((cell_1, cell_2, cell_3), axis=0)
    
    fig, ax1 = plt.subplots()
    ax1.scatter(
         coords[:, 0],
         coords[:,1],
         c=[sns.color_palette()[x] for x in key.iloc[:,2]], 
         #c = key, cmap = 'OrRd',
         s=20, alpha = 0.35)
    ax1.scatter(
        high_cells[:,0],
        high_cells[:,1],
        c='k',
        s=20, alpha=1)
    ax1.set_xbound(coords[:,0].min()-1, coords[:,0].max()+1)
    ax1.set_ybound(coords[:,1].min()-1, coords[:,1].max()+1)
    plt.xlabel('UMAP 1', labelpad=3)
    plt.ylabel('UMAP 2', labelpad=0)
    plt.tick_params(bottom = False, labelbottom = False, left=False, labelleft=False)
    plt.gca().set_aspect(1.5)
    fig.savefig(f"../Final_things/Figures/UMAPs/cshape_cell_highlight.png", dpi = 600, bbox_inches = 'tight')
    
    return
    
def grad_plots_cshape():
    ''' built using coords from cshape_stats'''
    props = ['CellArea', 'CellPerimeter', "CellMinorAxisLength", 
                  "CellCircularity"]
    titles = ['Area', 'Perimeter', 'Minor Axis Length', 'Circularity']
    vmins = [0, 0, 0, 0]
    vmaxs  = [100000, 4000, 500, 0.75]
    #vmaxs  = [42165, 2597, 324, 0.75]
    fig = plt.figure()  
    for i in range(4):
        plt.subplot(2,2,i+1)
        A = plt.scatter(coords[:,0], coords[:,1], c = stats[props[i]], cmap = 'OrRd',
                    vmin=vmins[i], vmax=vmaxs[i], s=10, alpha = 0.75)
        plt.colorbar(mappable=A)
        plt.gca().set_aspect(1)
        plt.tick_params(bottom = False, labelbottom = False, left=False, labelleft=False)
        plt.title(titles[i], fontsize=8, fontweight='bold')
        
    fig.subplots_adjust(wspace=-0.1, hspace=0.2)
    
    fig.savefig(f"../Final_things/Figures/UMAPs/cshape_grads_colorbar.png", dpi = 600, bbox_inches = 'tight')

def grad_plots_byol():
    #for i in range(22):
        #plot_umap_2d(coords, stats.iloc[:,i+1])
    props = ['CellArea', 'CellPerimeter', "CellMinorAxisLength", 
                  "CellCircularity"]
    titles = ['Area', 'Perimeter', 'Minor Axis Length', 'Circularity']
    vmins = [0, 0, 0, 0]
    vmaxs  = [100000, 4000, 500, 0.75]
    fig = plt.figure()  
    for i in range(4):
        plt.subplot(2,2,i+1)
        A = plt.scatter(coords[:,0], coords[:,1], c = stats[props[i]], cmap = 'OrRd',
                    vmin=vmins[i], vmax=vmaxs[i], s=10, alpha = 0.75)
        plt.gca().set_aspect(1.25)
        plt.tick_params(bottom = False, labelbottom = False, left=False, labelleft=False)
        plt.title(titles[i], fontsize=8, fontweight='bold')
        
    fig.subplots_adjust(wspace=-0.3, hspace=0.2)
    fig.savefig(f"../Final_things/Figures/UMAPs/BYOL37_grads.png", dpi = 600, bbox_inches = 'tight')

def cont_label_BYOL(coords, key):
    coords = np.asarray(coords)
    x1 = np.arange(coords[:,0].min()-0.1,coords[:,0].max()+0.1,0.01)
    slope, intercept, direction = fit_line(coords, key)
    fig, ax1 = plt.subplots()
    ax1.scatter(
         coords[:, 0],
         coords[:,1],
         #c=[sns.color_palette()[x] for x in key], 
         c = key, cmap = 'OrRd',
         s=40, alpha = 1)
    #ax1.plot(x1, slope*x1 +intercept, 'k-', lw=3)
    ax1.set_xbound(coords[:,0].min()-1, coords[:,0].max()+1)
    ax1.set_ybound(coords[:,1].min()-1, coords[:,1].max()+1)
    #plt.xlabel('UMAP 1', labelpad=0)
    #plt.ylabel('UMAP 2', labelpad=0)
    plt.gca().set_aspect(1.25)
    plt.tick_params(bottom = False, labelbottom = False, left=False, labelleft=False)
    del coords
    fig.savefig(f"../Final_things/Figures/UMAPs/UMAP_byol37_cont_lables.png", dpi = 600, bbox_inches = 'tight')
    return direction


def from_BYOL():
    direction = plot_umap_2d(unsuper.iloc[:,2:], unsuper.iloc[:,1])
    print(direction)
    projection = project2line(unsuper.iloc[:,2:], unsuper.iloc[:,1])
    labels = cont_labels(projection, size=1000)
    
    return labels

def from_BYOL_37():
    coords, trans = emb_to_tUMAP(byol_37_emb, nn=30, mind=0.5, rs=9, metric='manhattan', ncomp=2)
    #coords, trans = emb_to_tUMAP(byol_37_emb, nn=200, mind=1, rs=1, metric='correlation', ncomp=2)
    '''for below rs = 30, '''
    #coords, trans = emb_to_tUMAP(byol_37_emb, nn=225, mind=1, rs=2, metric='correlation', ncomp=2)
    #coords, trans = emb_to_tUMAP(byol_37_emb, nn=10, mind=0.5, rs=2, metric='euclidean', ncomp=2)
    ''' below makes plot with standard +/- labels'''
    direction = plot_umap_2d(coords, key.iloc[:,2])
    ''' below makes plot with continuious labels from continuous labels'''
    #continuous_labels = np.load('../Final_things/Figures/UMAPs/labels_cshape.npy').astype('int')
    #direction = cont_label_BYOL(coords, sscontlabels)
    
    
    #direction = plot_umap_2d(byol_37.iloc[:,:], key.iloc[:,2])
    projection = project2line(coords, key.iloc[:,2])
    labels = cont_labels(projection, size=1000)
    return labels
    
def compare_labels(continuous_labels, b_cont_labels):
    ''' '''
    fig, ax1 = plt.subplots()
    b_cont_labels = 1000 - b_cont_labels # need to invert which end of spectrum is activated
    ax1.scatter(continuous_labels, b_cont_labels)
    plt.xlabel('feature label', labelpad=0)
    plt.ylabel('BYOL_label', labelpad=0)
    r2 = r2_score(continuous_labels, b_cont_labels)
    reg = sklearn.linear_model.LinearRegression().fit(continuous_labels.reshape(-1,1), b_cont_labels.reshape(-1,1))
    slope = reg.coef_
    
    
def comp_plot(labels, prop):
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(labels,
         np.asarray(stats[prop]))
    fig, ax1 = plt.subplots()
    ax1.scatter(
         labels,
         np.asarray(stats[prop]),
         s=20, alpha = 0.75)
    ax1.plot(labels, slope*labels + intercept, 'k-', lw=3)
    ax1.set_xbound(0,1000)
    ax1.set_ybound(np.asarray(stats[prop]).min(), np.asarray(stats[prop]).max())
    plt.xlabel('continuious label', labelpad=0)
    plt.ylabel(f'{prop}', labelpad=0)
    plt.title(f'{prop}')
    print(f'The R^2 value is {r_value**2}')



def comp_subplots(labels):
    prop_list = ['CellArea', "CellMinorAxisLength", "NucCytRatio", "CellCircularity"]
    fig = plt.figure()
    for i in range(4):
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(labels,
         np.asarray(stats[prop_list[i]]))
        plt.subplot(2,2,i+1)
        plt.scatter(labels,np.asarray(stats[prop_list[i]]), s=10, alpha=0.75)
        plt.plot(labels, slope*labels + intercept, 'k-', lw=3)
        # plt.set_xbound(0,1000)
        # plt.set_ybound(np.asarray(stats[prop]).min(), np.asarray(stats[prop]).max())
        #plt.xlabel('continuious label', labelpad=0)
        if (i+1) < 3:
            plt.tick_params(labelbottom=False)    
        plt.tick_params(labelleft=False)
        #plt.ylabel(f'{prop_list[i]}', labelpad=0)
        plt.title(f'{prop_list[i]}', fontsize=10, fontweight='bold')
        
    fig.subplots_adjust(wspace=0.1, hspace=0.3)
    
    return

# A = list(stats)

# for i in range(21):
#     print(f'{A[i]}')
#     comp_plot(labels, A[i+1])
    

# plt.scatter(labels, key.iloc[:,2], s=100, alpha=0.05)







