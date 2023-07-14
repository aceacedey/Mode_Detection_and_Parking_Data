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
from shapely.geometry import shape, Point, mapping
from shapely.ops import linemerge
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
from PIL import Image,ImageTk
from io import BytesIO
from tempfile import TemporaryFile
#matplotlib.use("Qt5Agg")
from bs4 import BeautifulSoup
import re
from geopy import distance

import googlemaps
import google_streetview.api

plt.rcParams.update({'font.size': 15})
#gmaps = googlemaps.Client(key='AIzaSyDWCjv6bX4CiziCjQit_gzET1JZA0uDsJI')
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
            #print(dist_list)
            #print(minDistPark[-1])
            #print(allParking[-1])
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

def plotPoints(df):
#if 1 == 1:
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
    lat = list(syp + eyp +NPsyp)
    lon = list(sxp + exp +NPsxp)
    bbox = [max(lat),min(lat), max(lon), min(lon)] ##[maxLat,minLat,maxLon,minLon]
    drive_filter = ox.core.get_osm_filter('drive')
    og = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],truncate_by_edge=True,custom_filter=drive_filter)
    ogp = ox.project_graph(og)
    fig ,ax = ox.plot_graph(ogp, show=False, close=False)
    ax.plot(NPsxp,NPsyp,c='green',marker='*', linestyle=' ')
    ax.plot(sxp,syp,c='blue',marker='+', linestyle=' ')
    ax.plot(exp,eyp,c='red',marker='o', linestyle=' ')
    plt.show()
    

FMT = '%H:%M:%S'
FMTD = '%D/%M/:Y'

startTime = datetime.now()
#import mmHmmOrthogonalDistanceFunction as mm
import utm


dfAll = pd.read_csv('Parking_Caronly.csv',index_col=0)
#dfxl = pd.read_excel('Parking_Caronly.xlsx',index_col=0)
#dfAll[['start','end','NPstart','NPend']] = pd.DataFrame(dfAll.apply(lambda x: [ast.literal_eval(item) for item in [x.start,x.end,x.NPstart,x.NPend]],axis=1).to_list())
uNPs = list(dfAll.NPstart.unique())
for item in uNPs:
    tempdf = dfAll[dfAll.start == item]
    if len(tempdf) > 2:
        break
#df = eqdf
df = dfAll.loc[[7,8,9,10],:]
if 1 == 1:
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
    lon = df.startx.values.tolist() + df.endx.values.tolist() + df.NPstartx.values.tolist()
    bbox = [max(lat) + 0.0025,min(lat)-0.0025, max(lon)+0.0025, min(lon)-0.0025] ##[maxLat,minLat,maxLon,minLon]
    
    drive_filter = ox.core.get_osm_filter('drive')
    og = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],truncate_by_edge=True,retain_all=True)
    ogp = ox.projection.project_graph(og)
##if 1 == 1:
##    fig ,ax = ox.plot_graph(ogp, show=False, close=False)
##    ax.plot(NPsxp,NPsyp,c='green',marker='*', linestyle=' ')
##    ax.plot(sxp,syp,c='blue',marker='+', linestyle=' ')
##    ax.plot(exp,eyp,c='red',marker='o', linestyle=' ')
##    plt.show()
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib_scalebar.scalebar import ScaleBar
import ast

if 1 == 1:
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

###########find out all unique parking spots
pdf = pd.DataFrame()
pdf[['start','end','NPstart','NPend']] = pd.DataFrame(dfAll.apply(lambda x: [ast.literal_eval(item) for item in [x.start,x.end,x.NPstart,x.NPend]],axis=1).to_list())

allP = pdf.NPstart.to_list() + pdf.NPend.to_list()
allP = list(filter(((0,0)).__ne__, allP)) ## delete all (0,0) from allP
unP = list(set(allP))

## find all unique start end locations of a trajectory

allSE = pdf.start.to_list() + pdf.end.to_list()
unSE = np.unique(np.array(allSE),axis=0)

### for each parking lot, idenfiy hhow many change points are there: find nearest endand Start of trajectories, find locations and tag
for tp in unP:
    
