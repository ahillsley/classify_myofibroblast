'''histograms of everything'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.interpolate import make_interp_spline, BSpline
import scipy
import math

def histPlot(prop):
    fig, ax1 = plt.subplots()
    aRange = (np.min(oCSV[prop]), np.mean(aCSV[prop]) + 2*np.std(aCSV[prop]))
    
    # remove outliers
    aVal = np.array(aCSV[prop])#/1000  # for area
    nVal = np.array(nCSV[prop])#/1000  # for area
    aVal = aVal[aVal >= aRange[0]]
    aVal = aVal[aVal <= aRange[1]]
    nVal = nVal[nVal >= aRange[0]]
    nVal = nVal[nVal <= aRange[1]]
    
    a_avg = np.mean(aVal)
    n_avg = np.mean(nVal)
    a_std = np.std(aVal)
    n_std = np.std(nVal)
    
    aDist = stats.norm(np.mean(aVal), np.std(aVal))
    nDist = stats.norm(np.mean(nVal), np.std(nVal))
    values = np.linspace(aRange[0], aRange[1], 1000)
    aProb = aDist.pdf(values)
    nProb = nDist.pdf(values)
    ax1.plot(values, aProb, color = colors[0], linewidth = 5)#, label = labels[0])
    ax1.plot(values, nProb, color = colors[1], linewidth = 5)#, label = labels[1]) 
    
    
    ax1.hist(x=[aVal, nVal], bins='auto', color = colors, 
             alpha=0.9, rwidth=0.9, density = True,
             range = aRange, label = labels)
    
    ax1.legend()
    ax1.set_yticks([])
    plt.setp(ax1.get_xticklabels(), rotation = 45, fontsize = 15, fontweight='bold')
    plt.setp(ax1.get_yticklabels(), fontsize = 15)
    #plt.grid(axis='y', alpha=0.75)
    #plt.xlabel(prop, fontsize = 15, fontweight = 'bold')
    #plt.ylabel('Frequency', fontsize = 24, fontweight = 'bold')
    plt.title(f"{prop}", fontsize = 30, fontweight = 'bold')
    plt.legend(fontsize = 10, loc=1)
    plt.show()
    #fig.savefig(f"../Final_things/Figures/Hist_{prop}.png", dpi = 600, bbox_inches = 'tight')
    
    return a_avg, a_std, n_avg, n_std

oCSV = pd.read_csv('../Final_things/Cell_properties_um.csv')
aCSV = oCSV[oCSV["Acti"] != 'Not'] #all activated cells
nCSV = oCSV[oCSV["Acti"] == 'Not'] #all not activated cells

allProp = list(oCSV.columns)
allProp = allProp[1:22]
# allProp = ['Cell Perimeter', "Cell Major Axis Length", "Cell Minor Axis Length",
#            "Cell Area", "Nucleus Minor Axis Length", "Nucleus Major Axis Length",
#            "Nucleus Perimeter", "Nucleus Area"]
colors = ['red', 'orange']
labels = ['a-SMA +', 'a-SMA -']
avg_stats = np.zeros((len(allProp), 4))
i = 0
for prop in allProp: 
    a_avg, a_std, n_avg, n_std = histPlot(prop)
    avg_stats[i,:] = [a_avg, a_std, n_avg, n_std]
    i += 1
