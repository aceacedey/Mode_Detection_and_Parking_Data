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
from shapely.geometry import shape, Point, mapping, LineString

import utm

FMT = '%H:%M:%S'
FMTD = '%D/%M/:Y'
def path_cost(G, path):
    return sum([G[path[i]][path[i+1]][0]['length'] for i in range(len(path)-1)])


def osmTodis(o1,o2,og):#,omsN,osmE): # takes two osmids, return two lists, distance time velocity etc.
    #tl = [t1,t2]
    #tdelta = datetime.strptime(t2, FMT) - datetime.strptime(t1, FMT)
    try:
        path = nx.shortest_path(og,o1,o2,weight='length')
    except:
        path = []
    if path:dist = sum([og[path[i]][path[i+1]][0]['length'] for i in range(len(path)-1)])
    else: dist = 1000000 ## almost infinite for no path
    #vel = []
    return path,dist

def Bearing(optdf):
    p = optdf.point.values
    #e = optdf.edge.values
    bearList = [0]
    #bearListM = [0]
    for i in range(1,len(p)):
        b = math.atan2(p[i][1]-p[i-1][1],p[i][0]-p[i-1][0])
        bd = np.rad2deg(b)
        o1 = osmE.u[e[i]]
        o2 = osmE.v[e[i]]
        p1 = (osmN.x[o1],osmN.y[o1])
        p2 = (osmN.x[o2],osmN.y[o2])
        a = math.atan2(p2[1]-p1[1],p2[0]-p1[0])
        ad = np.rad2deg(a)
        bearList.append(abs(ad-bd))
    return bearList

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
    return angdiff #np.rad2deg(np.array(theta)) #np.array(theta)#

def VelocityPointwise(rawdf):
    lat = rawdf.lat.values.tolist()
    lon = rawdf.lon.values.tolist()
    t = rawdf.time.values.tolist() #apply(lambda x: ast.literal_eval(x)[0]).values
    tL = [[]] * len(t) #
    #tL[0] = osmTodis(osmE.u[e[0]], osmE.v[e[0]],ogp)[0]
    vL = [0] * len(t)
             
    for i in range(1,len(t)):
        lat1 = lat[i-1]
        lat2 = lat[i]
        lon1 = lon[i-1]
        lon2 = lon[i]
        dx = DistanceLatLon(lat1,lat2,lon1,lon2)
        dt =  (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
        try:
            vL[i] = dx/dt
        except:
            vL[i] = 0
        
    return np.array(vL) #* 3.6

def VelocityPointwiseProj(pointsp):
    lon = []
    lat = []
    for item in pointsp:
        lon.append(item[0])
        lat.append(item[1])
    
    t = rawdf.time.values.tolist()#.apply(lambda x: ast.literal_eval(x)[0]).values.tolist() 
    tL = [[]] * len(t) #
    tL[0] = 0
    #tL[0] = osmTodis(osmE.u[e[0]], osmE.v[e[0]],ogp)[0]
    vL = [0] * len(t)
    for i in range(1,len(t)):
        lat1 = lat[i-1]
        lat2 = lat[i]
        lon1 = lon[i-1]
        lon2 = lon[i]
        dx = sqrt((lat1-lat2)**2 +(lon1-lon2)**2 )
        dt =  (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
        try:
            vL[i] = dx/dt
        except:
            vL[i] = 0
        tL[i] = dt 
    return np.array(vL),tL #* 3.6

def AccPointwise(velsp):
    t = rawdf.time.values.tolist()#.apply(lambda x: ast.literal_eval(x)[0]).values.tolist() 
    tL = [[]] * len(t) #
    aL = [0] * len(t)
    for i in range(1,len(t)):
        v1 = velsp[i-1]
        v2 = velsp[i]

        dx2 = v2-v1 
        dt2 =  (DTtoFloat(t[i]) - DTtoFloat(t[i-1]))
        try:
            aL[i] = dx2/dt2
        except:
            aL[i] = 0
        
    return np.array(aL) #* 3600

def DistanceLatLon(lat1,lat2,lon1,lon2):
   import math
   lat1 = lat1 * (math.pi/180)
   lat2 = lat2 * (math.pi/180)
   dlon = (lon2-lon1)  * (math.pi/180)
   dlat = (lat2 - lat1)  * (math.pi/180)
   a = (math.sin(dlat/2))**2 + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2))**2 
   c = 2 * math.atan2( np.sqrt(a), np.sqrt(1-a) ) 
   d = 6373 * c 
   Dist = d
   return Dist * 1000

def removeListfromListNUMPY(b): ## remove lists from a list
    #print(b)
    a=[]
    for i,item in enumerate(b):
        if type(item) == list: # or type(item) == list:
            a.append(i)
    return np.delete(b,a)


def OSMmidPoint(u,v):
    return ( (osmN.x[u] + osmN.x[v])/2 , (osmN.y[u] + osmN.y[v])/2 )

def DistPoint(p1,p2):
    return sqrt((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2 )

def AreaTriangle(oe,p):
    #print(oe)
    x1 = osmN.x[osmE.u[oe]]
    y1 = osmN.y[osmE.u[oe]]
    x2 = osmN.x[osmE.v[oe]]
    y2 = osmN.y[osmE.v[oe]]
    a = np.array([[x1,y1,1],[x2,y2,1],[p[0],p[1],1]])
    #print(a)
    return abs(np.linalg.det(a))

def DTtoFloat(dstring):
    x = datetime.strptime(dstring, FMT)
    return x.hour * 3600 + x.minute * 60 + x.second

def OrthoDist(oe,p):
    x1 = osmN.x[osmE.u[oe]]
    y1 = osmN.y[osmE.u[oe]]
    x2 = osmN.x[osmE.v[oe]]
    y2 = osmN.y[osmE.v[oe]]
    A = np.array([x1,y1])
    B = np.array([x2,y2])
    C = np.array(p)
    
    d = np.linalg.norm(np.cross(B - A, C - A))/np.linalg.norm(B - A)
    return d

def Plot(x,y):
    plt.scatter(x,y, color='b')
    plt.show()

def PlotTime(x,y):
    
    poiy = list(range(0,len(y)))
    fig, ax = plt.subplots()
    ax.plot(poiy,x/max(x), c='blue',marker='*', linestyle='-',label='Observed')
    #for an in x1: 
        #ax.annotate(an,poi[an])
    ax.plot(poiy,y/max(y),c='red',marker='o', linestyle='--',label='Predicted')
    plt.xlabel('GPS Samples in with labeles')
    plt.title("Un supervised predction of change of mode: 0 = Walk, 1 = Drive")
    ax.legend(loc='upper left', frameon=False)
    plt.show()


def Kmeans(data):
    kmeans = KMeans(n_clusters=2, random_state=0).fit(data)
    eL = kmeans.labels_
    
    return eL,kmeans.cluster_centers_


def hdbscan(data):
    Csize = 1
    import hdbscan
    #clusterer = hdbscan.HDBSCAN(min_cluster_size=round(len(data)/2), min_samples=round(len(data)/2)
    clusterer = hdbscan.HDBSCAN(metric='euclidean',gen_min_span_tree=True)
    clusterer.fit(data)
    #clusterer.probabilities_
    eL = clusterer.labels_
    
    return eL


startTime = datetime.now()

np.set_printoptions(suppress=True) 
import hdbscan

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import mixture
from os import listdir
from os.path import isfile, join


TRpath = "LabelledTrajectories"
dataframes = ''
#filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
TRfiles = listdir(TRpath)
#filenames = listdir(MMpath)
#drive_filter = ox.core.get_osm_filter('drive')
output = pd.DataFrame(columns=['Observed','Predicted','error'])
drive_filter = ox.core.get_osm_filter('drive')
c = 0
dfTrainTest = pd.DataFrame()
TimeofStay = []
dsnumber = 1
for inf,file in enumerate(TRfiles):
    ## Read the relevant raw trajectory
    #rPath = str(MMpath) + "/" + str(file)
    rPath = str(TRpath) + "/" + str(file)
    rawdf = pd.read_csv(rPath,index_col=0)
    rawdf.drop(['0', 'Alt', 'NoD', 'date'],inplace = True,axis=1)
    uTRmode = set(rawdf.TrMode.values)
    #print(uTRmode)
    lon, lat = list(rawdf.lon.values), list(rawdf.lat.values)
    points = list(zip(lon,lat))
    pointsp = rawdf.apply(lambda x: utm.from_latlon(x.lat,x.lon)[0:2],axis=1)
    rawdf['pointsp'] = pointsp   
    #rawdf['PointVelocity'] = VelocityPointwise(rawdf)
    rawdf['PointVelocityPrj'] = VelocityPointwiseProj(pointsp)[0]
    
    
    rawdf['PointAnglePrj'] = AnglePointwiseProj(pointsp)
    #rawdf['BearingDiff'] = Bearing(rawdf)
    velsp = rawdf.PointVelocityPrj.values
    rawdf['PointAccPrj'] = AccPointwise(velsp)
    #vel = rawdf.PointVelocity.values
    #rawdf['PointAcc'] = AccPointwise(vel)
    #print(rawdf)
    rawdf["deltime"] = VelocityPointwiseProj(pointsp)[1]
    
    if uTRmode == {'car','walk'}:
        #if 'walk' in uTRmode:
        c = c + 1
        print(uTRmode)
        rawdf['observed'] = rawdf.TrMode.apply(lambda x: 0 if x == 'walk' else 1)
        rawdf['ModeChanges'] = (rawdf.observed.diff(1)!= 0).astype('int').cumsum()
        Grpdf = rawdf.groupby('ModeChanges').head(n=1)
        start_idx = Grpdf.index.values.tolist()
        start_idx.append(rawdf.index.values[-1])
        numGr = len(start_idx)-1 
        print(numGr)
        for tr in range(0,numGr):
            numPoints = start_idx[tr+1] - start_idx[tr]
            temp = rawdf.loc[start_idx[tr]:start_idx[tr+1]-1]
            timeOfstay = temp.deltime.values.sum()
            TimeofStay.append(timeOfstay)
            rawdf.loc[start_idx[tr]:start_idx[tr+1]-1,'TimeOfStay'] = timeOfstay
        rawdf.loc[start_idx[tr+1],'TimeOfStay'] = timeOfstay
        
        curName = "DataSet_" + str(dsnumber) + '.csv'
        dsnumber = dsnumber + 1
        #curName = "CW_" + str(file)
        rawdf.to_csv(curName,header=True)
        dfTrainTest = dfTrainTest.append(rawdf)
        
        
    elif uTRmode.intersection({'car','walk'}) == {'car','walk'}:
        print("------intersection------")
        print(uTRmode)
        ##############
        
        cwdf = rawdf[np.logical_or(rawdf.TrMode == 'walk', rawdf.TrMode=='car')] ## Keep only car and walk columns
        
        cwdf.loc[:,'observed'] = cwdf.TrMode.apply(lambda x: 0 if x == 'walk' else 1)
#        #############

        
        discon = np.ediff1d(cwdf.index.values).tolist()### check for discontinuity
        discon.insert(0,1)
        discon_idx = np.where(np.array(discon) != 1)[0]
        d_list = [0] + list(discon_idx) + [cwdf.index.values[-1]]
        
        list_of_dfs = [cwdf.iloc[d_list[n]:d_list[n+1]] for n in range(len(d_list)-1)]

        for idx,temp2 in enumerate(list_of_dfs):
            temp = temp2
            if len(set(temp.TrMode.values)) > 1:
                curName = "DataSet_" + str(dsnumber) + '.csv'
                dsnumber = dsnumber + 1
                temp.reset_index()
                temp.loc[:,'ModeChanges'] = (temp.observed.diff(1)!= 0).astype('int').cumsum()

                Grpdf = temp.groupby('ModeChanges').head(n=1)
                start_idx = Grpdf.index.values.tolist()
                start_idx.append(temp.index.values[-1])
                numGr = len(start_idx)-1 
                #print(numGr)
                for tr in range(0,numGr):
                    numPoints = start_idx[tr+1] - start_idx[tr]
                    temp1 = temp.loc[start_idx[tr]:start_idx[tr+1]-1]
                    timeOfstay = temp1.deltime.values.sum()
                    TimeofStay.append(timeOfstay)
                    temp.loc[start_idx[tr]:start_idx[tr+1]-1,'TimeOfStay'] = timeOfstay
                

                temp.to_csv(curName,header=True)
                print(curName)
            
##        start_idx = 0
##        for tr in range(0,len(discon_idx)):
##            end_idx = discon_idx[tr] ## the index where we have a discontinuity
##            temp = rawdf.loc[start_idx:end_idx]
##        Grpdf = cwdf.groupby('TrMode').head(n=1)
##        start_idx = Grpdf.index.values
##        numGr = len(start_idx) - 1
####        l = Grpdf.TrMode.values
####        a = ['walk','car']
####        b = ['car','walk']
####        checkSequence1 = [a == l[i:i+2] for i in range(len(l1) - 1)]
####        checkSequence2 = [b == l[i:i+2] for i in range(len(l1) - 1)]
####        print(checkSequence)
####        if any(checkSequence):
####        for i in range(1, len(Grpdf)):
####            if (Grpdf.loc[i, 'TrMode'] == 'walk' and Grpdf.loc[i-1, 'TrMode'] == 'car') or 
####            
##        for tr in range(0,numGr):
##            numPoints = start_idx[tr+1] - start_idx[tr]
##            temp = rawdf.loc[start_idx[tr]:start_idx[tr+1]-1]
##            if (temp.TrMode.values[0] == 'car' or temp.TrMode.values[0] == 'walk'):
##                c = c + 1
##                TimeofStay.append(temp.deltime.values.sum())

##            curName = "CW_" + str(file)
            #curName = "CW_" + str(file)
            #rawdf.to_csv(curName,header=True)
            #dfTrainTest = dfTrainTest.append(rawdf)
            
minimum_time = min(TimeofStay)
print(TimeofStay)
TTName = "TrainTestFile.csv"# + str(file)

#dfTrainTest.to_csv(TTName)

