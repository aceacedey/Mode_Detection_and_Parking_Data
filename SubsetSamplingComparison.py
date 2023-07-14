import xml.etree.ElementTree as ET  
import gpxpy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime,timedelta
import osmnx as ox
import networkx as nx
import xml.sax
import fiona
import csv
from rtree import index
import math
from math import exp, sqrt
from datetime import datetime
import itertools
from collections import Counter, defaultdict
from pyproj import Proj, transform
from sklearn.neighbors import NearestNeighbors
import scipy
from scipy import spatial
import xml.sax
import statistics
import ast
from os import listdir,path
import time
from numpy import ma
from pykalman import KalmanFilter
import pywt
from sklearn.cluster import KMeans
import re

plt.rcParams.update({'font.size': 20})

def Kmeans(data):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    eL = kmeans.labels_
    
    return eL

def PlotSubset(x,y,ylabel):
    idx = np.argsort(y)
    sx = np.array(x)[idx]
    sy = np.array(y)[idx]
    
    poix = list(range(1,len(sx)+1))
    fig, ax = plt.subplots()
    ax.plot(poix,sx, c='red',marker='$L$', ms=13, linestyle='--',label=lg1)
    #for an in x1: 
        #ax.annotate(an,poi[an])
    ax.plot(poix,sy,c='blue',marker='$S$', ms=13, linestyle='--',label=lg2)
    
    plt.xlabel(xlabel)
    #plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(1, len(x)+1))
    ax.legend(loc='lower right', frameon=False)
    plt.xticks(poix, np.array(xticks)[idx], rotation=90)
    plt.grid(True)
    ax.grid(alpha=.3)
    #plt.ylim(0,1)
    plt.show()

    


FMT = '%H:%M:%S'
def DTtoFloat(dstring):
    x = datetime.strptime(dstring, FMT)
    return x.hour * 3600 + x.minute * 60 + x.second
def PlotTime(x,y,xlabel,ylabel,lg1,lg2,xticks):
    
    poix = list(range(1,len(x)+1))
    fig, ax = plt.subplots()
    ax.plot(poix,x, c='red',marker='*', linestyle='--',label=lg1,ms=12)
    #for an in x1: 
        #ax.annotate(an,poi[an])
    ax.plot(poix,y,c='blue',marker='o', linestyle='--',label=lg2,ms=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(1, len(x)+1))
    ax.legend(loc='lower right', frameon=False)
    plt.xticks(poix, xticks, rotation=90)
    plt.grid(True)
    ax.grid(alpha=.3)
    #plt.ylim(0,1)
    plt.show()   


edfall = pd.read_csv('errdf_CWall.csv',index_col=0)
edfsub = pd.read_csv('errordf_Subset_CWAll.csv',index_col=0)
sdf = pd.read_csv('errordf_Subset_only.csv',index_col=0)

edfsub['MeanStd'] = edfsub.MeanStd.apply(lambda x: ast.literal_eval(x))
edfsub[['mean','std']] = pd.DataFrame(edfsub.MeanStd.to_list(), index= edfsub.index)
edfsub['ratio'] = edfsub['std']/edfsub['mean']
 #edfall['MeanStd'] = edfall.MeanStd.apply(lambda x: ast.literal_eval(x))
##edfall[['mean','std']] = pd.DataFrame(edfall.MeanStd.to_list(), index= edfall.index)
##edfall['std/mean'] = edfall['std']/edfall['mean']
wfile = ['DataSet_13.csv', 'DataSet_14.csv']
##
edfall = edfall[~edfall.Dataset.isin(wfile)]
edfsub = edfsub[~edfsub.Dataset.isin(wfile)]
subfiles = ['DataSet_19.csv', 'DataSet_25.csv', 'DataSet_26.csv', 'DataSet_31.csv', 'DataSet_34.csv', 'DataSet_35.csv', 'DataSet_36.csv', 'DataSet_38.csv', 'DataSet_39.csv', 'DataSet_40.csv', 'DataSet_42.csv', 'DataSet_44.csv', 'DataSet_47.csv', 'DataSet_48.csv',
 'DataSet_54.csv', 'DataSet_60.csv', 'DataSet_61.csv', 'DataSet_67.csv']

allfiles = edfall.Dataset.apply(lambda x: x[0:-4]).to_list()
allfiles = [int(re.findall('\d+', item)[0]) for item in allfiles]
subfiles = sdf.Dataset.apply(lambda x: x[0:-4]).to_list()

locs = edfsub[edfsub.Dataset.isin(sdf.Dataset.to_list())].index.to_list()

edfsub.loc[locs,['acRFf','acSVMf','acGBf']] = sdf[['acRFf','acSVMf','acGBf']].values

### Plotting subset sampling comparison for all 77 test dataset : 
if 1 == 1:
    xlabel = 'Trip dataset number'
    title ='Accuracy of mode prediction compared with subset sampling'
    
    lg1 = 'noERbC with 75 trips (130,973 datapoints)'
    lg2 = 'SERbC with 18 trips (58,276 datapoints)'
    xticks = allfiles

    x = edfall.acSVM.to_list()
    y = edfsub.acSVM.to_list()
    ylabel = 'Accuracy of prediction using SVM classifier'
    PlotSubset(x,y,ylabel)

    x = edfall.acRF.to_list()
    y = edfsub.acRF.to_list()
    ylabel = 'Accuracy of prediction using RF classifier'
    PlotSubset(x,y,ylabel)

    x = edfall.acGB.to_list()
    y = edfsub.acGB.to_list()
    ylabel = 'Accuracy of prediction using GB classifier'
    PlotSubset(x,y,ylabel)
    #en = edfall.Entropy.to_list()
    #dci = edfall.dci.to_list()
    #te = edfall.TE.to_list()

if 1 == 1:    
    print(edfall.acSVM.mean())
    print(edfsub.acSVM.mean())
    print(edfsub.acSVMf.mean())
    print("--------------")
    print(edfall.acRF.mean())
    print(edfsub.acRF.mean())
    print(edfsub.acRFf.mean())
    print("--------------")
    print(edfall.acGB.mean())
    print(edfsub.acGB.mean())
    print(edfsub.acGBf.mean())
    
if 1 == 1:
    
    acSVM = edfall.acSVM.to_list()
    acSVMf = edfsub.acSVMf.to_list()
    acSVMdwt = []

    acRF = edfall.acRF.to_list()
    acRFf = edfsub.acRFf.to_list()
    acRFdwt = []

    acGB = edfall.acGB.to_list()
    acGBf = edfsub.acGBf.to_list()
    
    TRfiles = allfiles
    ylabel = "Accuracy score with RF classifier"
    xlabel = "Trip dataset number"
    lg1 = "noERbC"
    lg2 = "MERbC"
    #Title = "Mode detection accuracy comparison"
    idxs = list(np.argsort(acRFf))
    sorted_ac = np.array(acRF)[idxs]#np.sort(acRF)
    sorted_acSW =  np.array(acRFf)[idxs]
    #sorted_xticks = [int(re.findall('\d+', TRfiles[item])[0])  for item in idxs]
    sorted_xticks = [TRfiles[item] for item in idxs]
    #sorted_xticks = ['Dataset '+ str(item) for item in range(1,len(acGB)+1)]
    PlotTime(sorted_ac,sorted_acSW,xlabel,ylabel,lg1,lg2,sorted_xticks)

    ylabel = "Accuracy score with SVM classifier"
    xlabel = "Trip dataset number"
    lg1 = "noERbC"
    lg2 = "MERbC"
    #Title = "Mode detection accuracy comparison"
    idxs = np.argsort(acSVMf)
    sorted_ac = np.array(acSVM)[idxs]#np.sort(acRF)
    sorted_acSW =  np.array(acSVMf)[idxs]
    #sorted_xticks = [int(re.findall('\d+', TRfiles[item])[0])  for item in idxs]
    sorted_xticks = [TRfiles[item] for item in idxs]
    #sorted_xticks = ['Dataset '+ str(item) for item in range(1,len(acGB)+1)]
    PlotTime(sorted_ac,sorted_acSW,xlabel,ylabel,lg1,lg2,sorted_xticks)

    ylabel = "Accuracy score with GB classifier"
    xlabel = "Trip dataset number"
    lg1 = "noERbC"
    lg2 = "MERbC"
    #Title = "Mode detection accuracy comparison"
    idxs = np.argsort(acGBf)
    sorted_ac = np.array(acGB)[idxs]#np.sort(acRF)
    sorted_acSW =  np.array(acGBf)[idxs]
    #sorted_xticks = [int(re.findall('\d+', TRfiles[item])[0]) for item in idxs]
    sorted_xticks = [TRfiles[item] for item in idxs]
    #sorted_xticks = ['Dataset '+ str(item) for item in range(1,len(acGB)+1)]
    PlotTime(sorted_ac,sorted_acSW,xlabel,ylabel,lg1,lg2,sorted_xticks)
##if 1 == 1:
####    TRpath = r"C:\Users\deys\OneDrive - The University of Melbourne\Ph D\Codes\GeoLife Work\Geolife Mode detection\Geoinformetica_code\3. Prepare Subset of Training\CWSelected"
####    dataframes = ''
####    SubFiles = listdir(TRpath)
##    subdf= pd.DataFrame()
##    subdf = edfsub[['Dataset', 'acRF', 'acRFdwt', 'acSVM', 'acSVMf','acGB', 'acGBf', 'Entropy','mean','std']][edfsub[(edfsub.mean < 5) & (edfsub['std'].divide(edfsub['mean']) < 10)]
##    print(subdf.round(2).to_latex(index=False))

if 1 == 1:
    #edfsub[['Dataset','Entropy','MeanStd']][(edfsub.acSVMdwt - edfsub.acSVM <0 ) | (edfsub.acRFdwt - edfsub.acRF < 0) | (edfsub.acGBdwt - edfsub.acGB < 0)]
    ffdf = pd.DataFrame()
    edfsub['SVM_diff'] = edfsub.acSVMf - edfsub.acSVM
    edfsub['RF_diff'] = edfsub.acRFf - edfsub.acRF
    edfsub['GB_diff'] = edfsub.acGBf - edfsub.acGB
    edfsub['Trip id'] = allfiles 
    ffdf = edfsub[(edfsub.acSVMf - edfsub.acSVM < 0 ) & (edfsub.acRFf - edfsub.acRF < 0)
                                                              & (edfsub.acGBf - edfsub.acGB < 0)]
    
    #ffdf['MeanStd'] = ffdf.MeanStd.apply(lambda x: ast.literal_eval(x))
    #ffdf[['mean','std']] = pd.DataFrame(ffdf.MeanStd.to_list(), index= ffdf.index)
    fdf = ffdf[['Trip id','Entropy','mean','std','TE','SVM_diff','RF_diff','GB_diff','acSVM','acRF','acGB']]
    print(fdf.round(2).to_latex(index=False))


plt.scatter(edfsub.ratio.to_list(),edfsub.acSVM.to_list())
plt.show()


if 1 == 1:    
    print(sdf.acSVM.mean())
    print(sdf.acSVMf.mean())
    
    print("--------------")

    print(sdf.acRF.mean())
    print(sdf.acRFf.mean())

    print("--------------")

    print(sdf.acGB.mean())
    print(sdf.acGBf.mean())
    
if 1 == 1:
    TRfiles = subfiles
    acSVM = sdf.acSVM.to_list()
    acSVMf = sdf.acSVMf.to_list()
    acSVMdwt = []

    acRF = sdf.acRF.to_list()
    acRFf = sdf.acRFf.to_list()
    acRFdwt = []

    acGB = sdf.acGB.to_list()
    acGBf = sdf.acGBf.to_list()
    TRfiles = allfiles
    ylabel = "Accuracy score with RF classifier"
    xlabel = "Trip dataset number"
    lg1 = "noERbC"
    lg2 = "ERbC"
    #Title = "Mode detection accuracy comparison"
    idxs = np.argsort(acRFf)
    sorted_ac = np.array(acRF)[idxs]#np.sort(acRF)
    sorted_acSW =  np.array(acRFf)[idxs]
    #sorted_xticks = [int(re.findall('\d+', TRfiles[item])[0]) for item in idxs]
    sorted_xticks = [TRfiles[item] for item in idxs]
    #sorted_xticks = ['Dataset '+ str(item) for item in range(1,len(acGB)+1)]
    PlotTime(sorted_ac,sorted_acSW,xlabel,ylabel,lg1,lg2,sorted_xticks)

    ylabel = "Accuracy score with SVM classifier"
    xlabel = "Trip dataset number"
    lg1 = "noERbC"
    lg2 = "ERbC"
    #Title = "Mode detection accuracy comparison"
    idxs = np.argsort(acSVMf)
    sorted_ac = np.array(acSVM)[idxs]#np.sort(acRF)
    sorted_acSW =  np.array(acSVMf)[idxs]
    #sorted_xticks = [int(re.findall('\d+', TRfiles[item])[0])  for item in idxs]
    sorted_xticks = [TRfiles[item] for item in idxs]
    #sorted_xticks = ['Dataset '+ str(item) for item in range(1,len(acGB)+1)]
    PlotTime(sorted_ac,sorted_acSW,xlabel,ylabel,lg1,lg2,sorted_xticks)

    ylabel = "Accuracy score with GB classifier"
    xlabel = "Trip dataset number"
    lg1 = "noERbC"
    lg2 = "ERbC"
    #Title = "Mode detection accuracy comparison"
    idxs = np.argsort(acGBf)
    sorted_ac = np.array(acGB)[idxs]#np.sort(acRF)
    sorted_acSW =  np.array(acGBf)[idxs]
    #sorted_xticks = [int(re.findall('\d+', TRfiles[item])[0])  for item in idxs]
    sorted_xticks = [TRfiles[item] for item in idxs]
    #sorted_xticks = ['Dataset '+ str(item) for item in range(1,len(acGB)+1)]
    PlotTime(sorted_ac,sorted_acSW,xlabel,ylabel,lg1,lg2,sorted_xticks)
