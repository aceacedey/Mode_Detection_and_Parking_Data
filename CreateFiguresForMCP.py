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
from shapely.geometry import shape, Point, mapping
from shapely.ops import linemerge
import fiona
import csv
from rtree import index
import math
from math import exp, sqrt
import itertools
from collections import Counter, defaultdict
from pyproj import Proj, transform
from scipy import spatial
import xml.sax
import statistics
from os import listdir
from os.path import isfile, join
import ast
import uuid
from types import GeneratorType
import requests
import json
from bs4 import BeautifulSoup
import re
from geopy import distance
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
import utm
import googlemaps
import google_streetview.api
import shapely
import geopandas as gpd
plt.rcParams.update({'font.size': 18})

apikey='AIzaSyDWCjv6bX4CiziCjQit_gzET1JZA0uDsJI'

def PlotParking(pointsp,labels,PLp,ogp):
    lon = []
    lat = []
    for item in pointsp:
        lon.append(item[0])
        lat.append(item[1])
        
    x1 = np.arange(0,len(PLp))
    x2 = np.arange(0,len(pointsp))
    osmN,osmE = ox.graph_to_gdfs(ogp,edges=True,nodes=True)
    poix = []
    poiy = []
    #pointsp = rawdf.apply(lambda x: utm.from_latlon(x.lat,x.lon)[0:2],axis=1)
    for poi in PLp:             #poi = [(osmN.x[i],osmN.y[i]) for i in traj]
        poix.append(poi[0])    #[osmN.x[i] for i in traj]
        poiy.append(poi[1])
         
    #fig, ax = ox.plot_graph(ogp, show=False, close=False)
    fig, ax = ox.plot_graph(ogp, show=False, close=False)
    ax.plot(poix,poiy, c='red',marker='o', linestyle=' ')
    for an in x1: 
        ax.annotate('Parking',PLp[an])
    
    ax.plot(lon,lat,c='green',marker='*', linestyle='dashed')
    for an in x2: 
        ax.annotate(labels[an],pointsp[an])
    plt.show()


def APICALL(ListOfPoints): ##(lat,lon)
    LoP = ListOfPoints
    print(LoP)
    ParkingLat = []
    ParkingLon = []
    nearest_list = []
    ArgNameStr = ()
    for point in LoP:
        ArgName = str(point[0]) + "," + str(point[1])
        
        url = "https://maps.googleapis.com/maps/api/place/search/json?location="+ ArgName +"&radius=200&types=parking&key=AIzaSyDWCjv6bX4CiziCjQit_gzET1JZA0uDsJI"
        #print(url)
        try:
            resp = requests.get(url)
            resp_json = resp.json()
            geo = resp_json['results']
            dist_list = []
            
            for g in geo:
                plon = g['geometry']['location']['lng']
                plat = g['geometry']['location']['lat']
                d = distance.distance(point,(plat,plon)).meters
                dist_list.append((plat,plon,d))                
            nearestParking = min(dist_list, key=lambda t: t[2]) ## find the tuple that has minimum distance
           # minDistPark.append(nearestParking)
            ParkingLat.append(nearestParking[0])
            ParkingLon.append(nearestParking[1])
            
            nearest_list.append(nearestParking[2])
 
        except:
            ParkingLat.append(0)
            ParkingLon.append(0)
            #allParking.append(['Null'])
            nearest_list.append(500)

        
    return nearest_list,ParkingLat,ParkingLon

def steet_view(pl_lat_lon):
    # Define parameters for street view api
    params = [{
            'size': '640x640', # max 640x640 pixels
            'location': pl_lat_lon,
            'heading': '0;90;180;270',
            'fov': '90',
            'key': 'AIzaSyDWCjv6bX4CiziCjQit_gzET1JZA0uDsJI'
    }]

    # Create a results object
    results = google_streetview.api.results(params)
    print(results)
    # Download images to directory 'downloads'
    return results


def PlotPSE(df):
    sxyp = df.apply(lambda x: utm.from_latlon(x.starty,x.startx)[0:2],axis=1)
    exyp = df.apply(lambda x: utm.from_latlon(x.endy,x.endx)[0:2],axis=1)
    NPsxyp = df.apply(lambda x: utm.from_latlon(x.NPstarty,x.NPstartx)[0:2],axis=1)
    
    sxp = list(zip(*list(sxyp)))[1]
    syp = list(zip(*list(sxyp)))[0]
    
    exp = list(zip(*list(exyp)))[1]
    eyp = list(zip(*list(exyp)))[0]

    NPsxp = list(zip(*list(NPsxyp)))[1]
    NPsyp = list(zip(*list(NPsxyp)))[0]
    
    NPendx = list(zip(*list(df.NPend.values)))[0]
    NPendy = list(zip(*list(df.NPend.values)))[1]
    lat = df.starty.values.tolist() + df.endy.values.tolist() + df.NPstarty.values.tolist()
    lat = list(filter((0.0).__ne__, lat))
    lon = df.startx.values.tolist() + df.endx.values.tolist() + df.NPstartx.values.tolist()
    lon = list(filter((0.0).__ne__, lon))
    
    bbox = [max(lat) + 0.0025,min(lat)-0.0025, max(lon)+0.0025, min(lon)-0.0025] ##[maxLat,minLat,maxLon,minLon]
    drive_filter = ox.core.get_osm_filter('drive')
    og = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],truncate_by_edge=True,retain_all=True)
    ogp = ox.projection.project_graph(og)
    
    fig ,ax = ox.plot_graph(ogp, show=False, close=False)
    ax.plot(NPsyp,NPsxp,c='green',marker='$P$', linestyle = ' ', ms=15,label='Parking location extracted from Google API')
    ax.plot(syp,sxp,c='blue',marker='$S$', linestyle = ' ', ms=15 ,label='Starting point of car-only part of a trip')
    ax.plot(eyp,exp,c='red',marker='$E$', linestyle = ' ', ms=15,label='Ending points of car-only part of another trip')
    plt.legend()
    x, y, arrow_length = 0, 1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax.transAxes)
    scalebar = ScaleBar(1) # 1 pixel = 0.2 meter
    plt.gca().add_artist(scalebar)
    plt.show()    

FMT = '%H:%M:%S'
FMTD = '%D/%M/:Y'

startTime = datetime.now()

path = "CWall"

filenames = listdir(path)

drive_filter = ox.core.get_osm_filter('drive')
if 1 == 1:
    ListCarStart = []
    ListCarEnd = []
    LCS = []
    LCE = []

    print('now')
    #print(stI)
    same = pd.DataFrame()
    for file in filenames:
        fPath = str(path) + "/" + str(file)
        rawdf = pd.read_csv(fPath)
        

        pointsp = list(rawdf.pointsp.apply(lambda x: ast.literal_eval(x)))
        time = rawdf.time.values
        
        lon, lat = list(rawdf.lon.values), list(rawdf.lat.values)
        pointsLatLon = list(zip(lat,lon)) ## in lat lon required for GOOGLE api

        
        tdf = rawdf.groupby('ModeChanges').head(n=1)
        
        cardf = rawdf[rawdf.TrMode == 'car']
        wdf = rawdf[rawdf.TrMode == 'walk']
        carGrpdf = cardf.groupby('ModeChanges').head(n=1)
        wGrpdf = wdf.groupby('ModeChanges').head(n=1)
        start_c_idx = carGrpdf.index.values.tolist()
        start_w_idx = wGrpdf.index.values.tolist()
        
        if start_w_idx[0] == 0: ## if a trajecotry starts with walking, then there was no car ended prior to it
            start_w_idx = start_w_idx[1::]
            
        end_c_idx = [(item-1) for item in start_w_idx]
        if (rawdf.TrMode.values[-1] == 'car'):    ## if last point of a trajectory is a car type, append that one
            end_c_idx.append(rawdf.index.values[-1])
            
        CarStart = [pointsp[item] for item in start_c_idx]
        CarEnd = [pointsp[item] for item in end_c_idx]
        
        ListCarStart = ListCarStart + CarStart
        ListCarEnd = ListCarEnd + CarEnd

        CarSt = [time[item] for item in start_c_idx]
        CarEn = [time[item] for item in end_c_idx]
        
        LCS = LCS + [pointsLatLon[item] for item in start_c_idx]
        LCE = LCE + [pointsLatLon[item] for item in end_c_idx]

        if len(CarStart) > 1:#type(CarStart) == list: 
            #same = same.append(pd.DataFrame({'Start_same':[CarStart],'End_same':[CarEnd],'st':[CarSt],'en':[CarEn],'file':file}),ignore_index = True)   
            for idx,temp in enumerate(CarStart[1::]):
                print(idx)
                S = temp
                E = CarEnd[idx]
                same = same.append(pd.DataFrame({'Sxp':S[0],'Syp':S[1],'Exp':E[0],'Eyp':E[1],'et':CarEn[idx],'st':CarSt[idx+1],
                                                 'C_CW': [(round(lat[end_c_idx[idx]],4),round(lon[end_c_idx[idx]],4))],
                                                 'C_WC': [(round(lat[end_c_idx[idx+1]],4),round(lon[end_c_idx[idx+1]],4))],
                                                 'lat':lat[end_c_idx[idx]],'lon':lon[end_c_idx[idx]]},index=[0]),ignore_index = True)

if 1 == 1:
    LCE = np.array(LCE)
    LCS = np.array(LCS)
        
    startTree = spatial.KDTree(ListCarStart)
    endTree = spatial.KDTree(ListCarEnd)
    positions = [[idx,endTree.query(item)[1]] for idx,item in enumerate(ListCarStart)]
    values = [endTree.query(item)[0] for idx,item in enumerate(ListCarStart)]

    W = np.zeros((len(ListCarEnd), len(ListCarEnd)),dtype='int')

    for idx,item in enumerate(ListCarStart):
        if endTree.query(item)[0] <= 150:
            W[idx][endTree.query(item,k=22)[1]] = endTree.query(item,k=22)[0]

    idxCarStart = np.where((W>0) & (W<200))[0]
    idxCarEnd = np.where((W>0) & (W<200))[1]

    st = LCS[idxCarStart].tolist()
    end = LCE[idxCarEnd].tolist()
    if 1==1:
        df = pd.DataFrame()
        df['start'] = st
        df['end'] = end
        df['NPstart'] = list(zip(APICALL(st)[1],APICALL(st)[2])) 
        df['NPend'] = list(zip(APICALL(end)[1],APICALL(end)[2])) 

    nozdf = df[(df.NPend != (0,0)) & (df.NPstart != (0,0))]

    eqdf = df[(df.NPstart == df.NPend) & (df.NPstart != (0,0)) & (df.NPend != (0,0))]
    if 1==1:
        df['startx'] = list(zip(*list(df.start.values)))[1]
        df['starty'] = list(zip(*list(df.start.values)))[0]
        df['endx'] = list(zip(*list(df.end.values)))[1]
        df['endy'] = list(zip(*list(df.end.values)))[0]
        df['NPstartx'] = list(zip(*list(df.NPstart.values)))[1]
        df['NPstarty'] = list(zip(*list(df.NPstart.values)))[0]
        df['NPendx'] = list(zip(*list(df.NPend.values)))[1]
        df['NPendy'] = list(zip(*list(df.NPend.values)))[0]
    df.to_csv('Parking_Caronly.csv',header=True)

    allP = df.NPstart.to_list() + df.NPend.to_list()
    allP = list(filter(((0,0)).__ne__, allP)) ## delete all (0,0) from allP
    unP = list(set(allP))
    NPsxyp = df.apply(lambda x: utm.from_latlon(x.NPstarty,x.NPstartx)[0:2],axis=1)
        
    #    sxp = list(zip(*list(sxyp)))[1]
    #    syp = list(zip(*list(sxyp)))[0]
        
    #    exp = list(zip(*list(exyp)))[1]
    #    eyp = list(zip(*list(exyp)))[0]

    NPsxp = list(zip(*list(NPsxyp)))[1]
    NPsyp = list(zip(*list(NPsxyp)))[0]

if 1 == 1:
    import random
    samedf = same.loc[3:25] ## for visualization
    
    number_of_colors = len(samedf)

    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]

    #color = list(np.random.choice(range(256), size=3))

    lat = samedf.lat.to_list()
    lon = samedf.lon.to_list()

    bbox = [max(lat) + 0.0025,min(lat)-0.0025, max(lon)+0.0025, min(lon)-0.0025] ##[maxLat,minLat,maxLon,minLon]
    drive_filter = ox.core.get_osm_filter('drive')
    og = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],custom_filter=drive_filter,simplify=True)
    ogp = ox.projection.project_graph(og)
if 1 == 1:
    #import matplotlib.lines as mlines
    streets = ox.save_load.graph_to_gdfs(ogp, nodes=False, edges=True,node_geometry=False, fill_edge_geometry=True)
    strt = shapely.ops.unary_union(streets.geometry)
    polygons = shapely.ops.polygonize(strt)
    polygonsg = gpd.GeoSeries(polygons)
    #polygonsg.plot()
    #plt.show()
if 1 == 1:
    NP = list(zip(NPsyp,NPsxp)) # + list(zip(NPsxp,NPsyp))

    plt.rcParams.update({'font.size': 18})
    same['\delta_t'] = same.apply(lambda x: (datetime.strptime(x.st, FMT) - datetime.strptime(x.et, FMT)).seconds/60, axis=1)
    samedf = same.drop(index=[19,22])
    samedf['zeta'] = np.linalg.norm(samedf[['Sxp','Syp']].values - samedf[['Exp','Eyp']].values,axis=1)
    samedf = samedf[samedf.zeta < 130]
    sxp = samedf.Sxp.to_list()
    syp = samedf.Syp.to_list()
    exp = samedf.Exp.to_list()
    eyp = samedf.Eyp.to_list()
    fig ,ax = ox.plot_graph(ogp, show=False, close=False)
    for p in NP:
        P = shapely.ops.Point(p)
        target = polygonsg.geometry[polygonsg.contains(P)].to_list()
        if len(target) >0:
            PL = target[0]
            PLx,PLy = PL.exterior.xy
            ax.fill(PLx,PLy, fc='green',alpha=0.3,ec='green')
    ax.plot(NPsyp,NPsxp,c='black',marker='$P$', linestyle = ' ', ms=15,label='Valid parking space')
    ax.plot(sxp,syp,'o', ms=20, markerfacecolor="None",markeredgecolor='blue', markeredgewidth=2,label='$C_{CW}$ of a MCP trip')
    ax.plot(exp,eyp,'o', ms=30, markerfacecolor="None",markeredgecolor='red', markeredgewidth=2,label='$C_{WC}$ of the same trip')

    plt.legend(loc='lower left')#['$S$','$E$'],['Starting point','Ending point'])
    x, y, arrow_length = 0, 1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax.transAxes)
    scalebar = ScaleBar(1) # 1 pixel = 0.2 meter
    plt.gca().add_artist(scalebar)
    plt.show()

    print(samedf[['C_CW', 'C_WC', 'zeta','et','st','\delta_t']].round(2).to_latex())
#np.linalg.norm(same[['Sxp','Syp']].values - same[['Exp','Eyp']].values,axis=1)
#from scipy.spatial.distance import pdist, squareform

#dist = pdist(same[['x1', 'x2']], 'euclidean')
#df_dist = pd.DataFrame(squareform(dist))

##    for i in range(len(sxp)):
##        ax.plot(sxp[i],syp[i],c=color[i],marker='$S$', linestyle = ' ', ms=15)
##        ax.plot(exp[i],eyp[i],c=color[i],marker='$E$', linestyle = ' ', ms=15)
##
##    sl = mlines.Line2D([], [],, marker='$S$',
##                          markersize=15, label='Starting point')
##    
##    el = mlines.Line2D([], [], color='blue', marker='$E$',
##                          markersize=15, label='Ending point')
##    plt.legend(handles=[sl,el])

    
##if 1 == 1: ## for mapping all parking spaces
##    
##    pdf = df.copy()
##    park = pd.DataFrame()
##
##    for item in unP:
##        tempdf = pdf[(pdf.NPstart == item) | (pdf.NPend == item)]
##        ustart = np.unique(np.array(tempdf.start.to_list()),axis=0) #list(tempdf.start.unique())
##        uend = np.unique(np.array(tempdf.end.to_list()),axis=0) #list(tempdf.end.unique())
##        park = park.append({'Parking spot':item,'Car_start':int(len(ustart)),'Car_end':int(len(uend))},ignore_index=True)
##        if (len(ustart) > 2 and len(uend) >2):
##            PlotPSE(tempdf)
##
##    park['Car_start'] = park.Car_start.astype(int)
##    park['Car_end'] = park.Car_end.astype(int)
##    print(park.to_latex())
##
##    park_trim = park.drop(index=[24,25])
##    ## plot all parking spaces
##    ### remove 
##    if 1 == 1:
##        PLp = park_trim['Parking spot'].to_list()#park.apply(lambda x: utm.from_latlon(x.lat,x.lon)[0:2],axis=1)
##        x1 = np.arange(0,len(PLp))
##        poix = []
##        poiy = []
##        lat = []
##        lon = []
##        for poi in PLp:
##            lat.append(poi[0])
##            lon.append(poi[1])
##            poip = utm.from_latlon(poi[0],poi[1])[0:2]  
##            poix.append(poip[0])    
##            poiy.append(poip[1])
##             
##        lat = list(filter((0.0).__ne__, lat))
##        lon = list(filter((0.0).__ne__, lon))
##        bbox = [max(lat) + 0.0025,min(lat)-0.0025, max(lon)+0.0025, min(lon)-0.0025] ##[maxLat,minLat,maxLon,minLon]
##        drive_filter = ox.core.get_osm_filter('drive')
##        og = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],custom_filter=drive_filter,simplify=True)
##        
##        ogp = ox.projection.project_graph(og)
##    if 1 == 1:
##        fig, ax = ox.plot_graph(ogp, show=False, close=False)
##        
##        ax.plot(poix,poiy, c='green',marker='$P$', linestyle = ' ', ms=15,label='Mapped parking spaces')
##        plt.legend(loc='lower left')
##        x, y, arrow_length = 0, 1, 0.1
##        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
##                arrowprops=dict(facecolor='black', width=5, headwidth=15),
##                ha='center', va='center', fontsize=20,
##                xycoords=ax.transAxes)
##        scalebar = ScaleBar(1) # 1 pixel = 0.2 meter
##        plt.gca().add_artist(scalebar)
##        plt.show()



    
