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
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy
import hdbscan
from sklearn.cluster import KMeans

plt.rcParams.update({'font.size': 18})

FMT = '%H:%M:%S'
def DTtoFloat(dstring):
    x = datetime.strptime(dstring, FMT)
    return x.hour * 3600 + x.minute * 60 + x.second




def AnglePointwiseProj(pointsp): # # calculate angle in radians
    #pointsp = list(optdf.point.apply(lambda x: ast.literal_eval(x)))
    lon = []
    lat = []
    for item in pointsp:
        lon.append(item[0])
        lat.append(item[1])
    theta = [0] * len(pointsp)
    for i in range(1,len(pointsp)):
        lat1 = lat[i-1]
        lat2 = lat[i]
        lon1 = lon[i-1]
        lon2 = lon[i]
        dy = lat2-lat1
        dx = lon2-lon1
        the = math.atan2(dy,dx)
        if the < 0:
            the = 2*math.pi + the
            
        theta[i] = the
    angles = np.rad2deg(np.array(theta)) # in degree
    angdif = abs(np.ediff1d(angles))
    angdiff = np.insert(angdif,0,0)
    return angdiff #np.rad2deg(np.array(the

def PlotTime(x,y,xlabel,title,ylabel,lg1,lg2,xticks):
    
    poix = list(range(1,len(x)+1))
    fig, ax = plt.subplots()
    ax.plot(poix,x, c='blue',marker='*', linestyle=' ',label=lg1)
    #for an in x1: 
        #ax.annotate(an,poi[an])
    ax.plot(poix,y,c='red',marker='o', linestyle=' ',label=lg2)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(1, len(x)+1))
    ax.legend(loc='lower right', frameon=False)
    plt.xticks(poix, xticks, rotation=90)
    plt.grid(True)
    #plt.ylim(0,1)
    plt.show()
    
def TimeDiff(t):
    #.apply(lambda x: ast.literal_eval(x)[0]).values.tolist() 
    tL = [[]] * len(t) #
    tL[0] = 0
    for i in range(1,len(t)):
        dt =  (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
        tL[i] = dt 
    return tL #* 3.6


#def dwtFilteringdf(XY):
            
def lowpassfilter(signal, thresh, wavelet="db3"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy


def Kmeans(data):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    eL = kmeans.labels_
    
    return eL


def hdbscan(data):
    Csize = round(len(data)[0]/5)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=Csize, min_samples=1,gen_min_span_tree=True)
    clusterer.fit(data)
    #clusterer.probabilities_
    eL = clusterer.labels_
    pL = clusterer.probabilities_
    return eL,pL

import MaskedKalmanFilter as mkf
TRpath = "CWAll"

trainMean = []
trainStd = []
dci = []
qMeanStdTest = []
Entropy = []

dataframes = ''
TRfiles = listdir(TRpath)

output = pd.DataFrame(columns=['Observed','Predicted','error'])
c = 0
dfTrainTest = pd.DataFrame()
TimeofStay = []
for inf,file in enumerate(TRfiles):
    ## Read the relevant raw trajectory
    #rPath = str(MMpath) + "/" + str(file)
    rPath = str(TRpath) + "/" + str(file)
    rawdf = pd.read_csv(rPath)
    rawdf = rawdf.reset_index(drop=True)
    pointsp = list(rawdf.pointsp.apply(lambda x: ast.literal_eval(x)))
    time = rawdf.time.values#apply(lambda x: ast.literal_eval(x)))
    #delete points those have impossible velocities before applying KF

    rawdf['ModeChanges'] = (rawdf.observed.diff(1)!= 0).astype('int').cumsum()
    delTs = TimeDiff(time)
    rawdf["deltime"] = delTs
    kfout = mkf.MaskedKalmanFilter(pointsp,list(rawdf.time))
    if kfout:
        fpoints = list(zip(kfout[0],kfout[1]))
    
    rawdf['pointsf'] = fpoints
    vf = kfout[2]
    accelP = kfout[3]
####    accelDwt = AccPointwise(vfDwt)
    rawdf['velocityFiltered'] = vf
    rawdf['accelFiltered'] = accelP
    rawdf['AngleFiltered'] = kfout[4]# angles in radians,change to degree by np.rad2deg(np.array(theta))
    rawdf['filename'] = [file] * len(rawdf)
    dfTrainTest = dfTrainTest.append(rawdf)
    print(file)
    
    dci.append(len(rawdf)/sum(delTs))
    qMeanStdTest.append([np.mean(delTs),np.std(delTs)])
    Entropy.append(calculate_entropy(delTs))
    

TTName = "TrainTestFile_filtered_CWAll.csv"# + str(file)
dfTrainTest.replace([np.inf, -np.inf], 0,inplace=True)
dfTrainTest.fillna(0,inplace=True)
df = dfTrainTest.reset_index(drop=True)
df.to_csv(TTName)

