import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime,timedelta
import osmnx as ox
import networkx as nx
from shapely.geometry import shape, Point, mapping
import fiona
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
import geopandas as gpd
import shapely
plt.rcParams.update({'font.size': 18})

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
    #drive_filter = ox.core.get_osm_filter('drive')
    og = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],truncate_by_edge=True,retain_all=True)
    ogp = ox.projection.project_graph(og)
    
    fig ,ax = ox.plot_graph(ogp, show=False, close=False,node_alpha=.2,edge_alpha=.3, bgcolor='w',node_color='b', node_size=1)
    ax.plot(NPsyp,NPsxp,c='green',marker='$P$', linestyle = ' ', ms=20,label='Parking location extracted from Google API')
    ax.plot(syp,sxp,c='blue',marker='$C_S$', linestyle = ' ', ms=20,label='$C_S$ a SCP trip')
    ax.plot(eyp,exp,c='red',marker='$C_E$', linestyle = ' ', ms=20,label='$C_E$ another SCP trip')
    plt.legend()
    x, y, arrow_length = 0, 1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax.transAxes)
    scalebar = ScaleBar(1) # 1 pixel = 0.2 meter
    plt.gca().add_artist(scalebar)
    plt.show()    


from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

point = Point(0.5, 0.5)
polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
print(polygon.contains(point))


##from scipy.spatial import ConvexHull, convex_hull_plot_2d
##points = np.random.rand(30, 2)   # 30 random points in 2-D
##hull = ConvexHull(points)
##
##import matplotlib.pyplot as plt
##plt.plot(points[:,0], points[:,1], 'o')
##for simplex in hull.simplices:
##    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

##unP = [(39.93297499999999, 116.384513), (39.9681392, 116.4193557),(39.981826, 116.327011), (39.98853769999999, 116.4667842),
##       (39.976081, 116.333105), (39.97683300000001, 116.326615),(39.971469, 116.418101), (39.9328317, 116.385243),
##       (39.9671123, 116.34134), (39.97503630000001, 116.3366916),(39.934721, 116.431141), (39.9824921, 116.4082543),
##       (39.9792806, 116.3278011), (39.935574, 116.431606),(39.982245, 116.494569), (39.986353, 116.381505), (39.9549196, 116.3981333),
##       (39.989693, 116.467846),(39.9338618, 116.4311106), (39.9709313, 116.4182334), (39.934801, 116.432149), (40.056968, 116.409963),
##       (40.0767333, 116.3272783), (40.3588584, 116.0039173), (26.88797, 100.224959), (26.8887682, 100.2280684), (39.9769593, 116.3321997)]

unP = [(39.93297499999999, 116.384513), (39.9681392, 116.4193557), (39.981826, 116.327011), (39.98853769999999, 116.4667842),
 (39.976081, 116.333105), (39.97683300000001, 116.326615), (39.971469, 116.418101), (39.9328317, 116.385243),
 (39.9671123, 116.34134), (39.97503630000001, 116.3366916), (39.934721, 116.431141), (39.9824921, 116.4082543),
 (39.9792806, 116.3278011), (39.935574, 116.431606), (39.982245, 116.494569), (39.986353, 116.381505),
 (39.9549196, 116.3981333), (39.989693, 116.467846), (39.9338618, 116.4311106), (39.9709313, 116.4182334),
 (39.934801, 116.432149), (40.056968, 116.409963), (40.0767333, 116.3272783),
 (40.3588584, 116.0039173), (26.88797, 100.224959), (26.8887682, 100.2280684), (39.9769593, 116.3321997)]

pdf = pd.read_csv('Parking_Caronly.csv',index_col = 0)

#allP = df.NPstart.to_list() + df.NPend.to_list()
#allP = list(filter(((0,0)).__ne__, allP)) ## delete all (0,0) from allP
#unP = list(set(allP))

park = pd.DataFrame()
count = 0
for item in unP:
    tempdf = pdf[(pdf.NPstart == str(item)) | (pdf.NPend == str(item))]
    ustart = np.unique(np.array(tempdf.start.to_list()),axis=0) #list(tempdf.start.unique())
    uend = np.unique(np.array(tempdf.end.to_list()),axis=0) #list(tempdf.end.unique())
    #break 
    if (len(ustart) > 0 and len(uend) > 0):
        #PlotPSE(tempdf)
#if 1 == 1:
        df = tempdf
        sxyp = df.apply(lambda x: utm.from_latlon(x.starty,x.startx)[0:2],axis=1)
        exyp = df.apply(lambda x: utm.from_latlon(x.endy,x.endx)[0:2],axis=1)
        NPsxyp = df.apply(lambda x: utm.from_latlon(x.NPstarty,x.NPstartx)[0:2],axis=1)
        NPexyp = df.apply(lambda x: utm.from_latlon(x.NPendy,x.NPendx)[0:2],axis=1)

        sxp = list(zip(*list(sxyp)))[0]
        syp = list(zip(*list(sxyp)))[1]

        exp = list(zip(*list(exyp)))[0]
        eyp = list(zip(*list(exyp)))[1]

        NPsxp = np.unique(list(zip(*list(NPsxyp)))[0])
        NPsyp = np.unique(list(zip(*list(NPsxyp)))[1])

        NPexp = np.unique(list(zip(*list(NPexyp)))[0])
        NPeyp = np.unique(list(zip(*list(NPexyp)))[1])

        lat = df.starty.values.tolist() + df.endy.values.tolist() + df.NPstarty.values.tolist()
        lat = list(filter((0.0).__ne__, lat))
        lon = df.startx.values.tolist() + df.endx.values.tolist() + df.NPstartx.values.tolist()
        lon = list(filter((0.0).__ne__, lon))

        bbox = [max(lat) + 0.0024,min(lat)-0.0024, max(lon)+0.0024, min(lon)-0.0024] ##[maxLat,minLat,maxLon,minLon]
        #drive_filter = ox.core.get_osm_filter('drive')
        og = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],retain_all=True,truncate_by_edge=False)
        ogp = ox.projection.project_graph(og)
        osmN,osmE = ox.graph_to_gdfs(ogp,edges=True,nodes=True)
        streets = ox.graph_to_gdfs(ogp, nodes=False, edges=True,node_geometry=False, fill_edge_geometry=True)
        strt = shapely.ops.unary_union(streets.geometry)
        polygons = shapely.ops.polygonize(strt)
        polygonsg = gpd.GeoSeries(polygons)
        #polygonsg.plot()
        #plt.show()
        #if 1 == 1:
        NP = list(zip(NPsxp,NPsyp)) # + list(zip(NPsxp,NPsyp))

        SCPx = len(sxyp.drop_duplicates().to_list())
        SCPy = len(exyp.drop_duplicates().to_list())
        SCP = sxyp.drop_duplicates().to_list()+ exyp.drop_duplicates().to_list() ## it contains all SCP chagepoints
        scpCir = [shapely.ops.Point(item).buffer(35) for item in SCP] ## convert all SCP points to a circle polygon 

        sxy = []
        exy = []
        #ax.set_aspect('equal', 'datalim')
        for p in NP:
            P = shapely.ops.Point(p)
            target = polygonsg.geometry[polygonsg.contains(P)].to_list()
            if len(target) >0:
                PL = target[0]
                fig ,ax = ox.plot_graph(ogp, show=False, close=False,node_alpha=.2,edge_alpha=.3, bgcolor='w',node_color='b', node_size=1)
                PLx,PLy = PL.exterior.xy
                ax.fill(PLx,PLy, fc='green',alpha=0.3,ec='green')
                scpArea = np.array([item.intersection(PL).area for item in scpCir])
                scpidx = np.where(scpArea > 0)[0]
                for idx in scpidx:
                    if idx < SCPx:
                        sxy.append(SCP[idx])
                        Csx,Csy = scpCir[idx].exterior.xy
                        ax.fill(Csx,Csy, fc='black',alpha=0.2,ec='blue')
                    else:
                        exy.append(SCP[idx])
                        Cex,Cey = scpCir[idx].exterior.xy
                        ax.fill(Cex,Cey, fc='black',alpha=0.2,ec='red')
                        
                ### plot all 
                #target.plot()
        if sxy and exy:
            park = park.append({'Parking spot':item,'Car_start':int(len(ustart)),'Car_end':int(len(uend))},ignore_index=True)
            sx = list(zip(*list(sxy)))[0]
            sy = list(zip(*list(sxy)))[1]
            
            ex = list(zip(*list(exy)))[0]
            ey = list(zip(*list(exy)))[1]

            ax.plot(NPsxp,NPsyp,c='green',marker='$P$', linestyle = ' ', ms=15,label='Parking location')
            ax.plot(sx,sy,c='blue',marker='$C_S$', linestyle = ' ', ms=15,label='$C_S$ of a SCP trip')
            ax.plot(ex,ey,c='red',marker='$C_E$', linestyle = ' ', ms=15,label='$C_E$ of another SCP trip')
            plt.legend(loc = 'lower left',prop={'size': 10})
            x, y, arrow_length = 0, 1, 0.1
            ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                    arrowprops=dict(facecolor='black', width=2, headwidth=10),
                    ha='center', va='center', fontsize=15,
                    xycoords=ax.transAxes)
            scalebar = ScaleBar(1) # 1 pixel = 0.2 meter
            plt.gca().add_artist(scalebar)
        ##        ratio = 1.0
        ##        xleft, xright = ax.get_xlim()
        ##        ybottom, ytop = ax.get_ylim()
        ##        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
            
            ax.set_aspect('equal')
            ax.set_rasterized(True)
            count = count + 1
            figname = 'SCP' + str(count) + '.png'
            plt.savefig(figname,dpi=2000)    
            #plt.show()
        
park['Car_start'] = park.Car_start.astype(int)
park['Car_end'] = park.Car_end.astype(int)
park.drop_duplicates(inplace=True)
print(park.to_latex())

    ## plot all parking spaces
    ### remove 
if 1 == 1:
    PLp = park['Parking spot'].to_list()#park.apply(lambda x: utm.from_latlon(x.lat,x.lon)[0:2],axis=1)
    x1 = np.arange(0,len(PLp))
    poix = []
    poiy = []
    lat = []
    lon = []
    for poi in PLp:
        lat.append(poi[0])
        lon.append(poi[1])
        poip = utm.from_latlon(poi[0],poi[1])[0:2]  
        poix.append(poip[0])    
        poiy.append(poip[1])
         
    lat = list(filter((0.0).__ne__, lat))
    lon = list(filter((0.0).__ne__, lon))
    bbox = [max(lat) + 0.005,min(lat)-0.005, max(lon)+0.005, min(lon)-0.005] ##[maxLat,minLat,maxLon,minLon]
    drive_filter = ox.core.get_osm_filter('drive')
    og = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3],custom_filter=drive_filter,simplify=True)
    
    ogp = ox.projection.project_graph(og)
#if 1 == 1:
    fig, ax = ox.plot_graph(ogp, show=False, close=False,node_alpha=.2,edge_alpha=.3, bgcolor='w',node_color='b', node_size=1)
    
    ax.plot(poix,poiy, c='green',marker='$P$', linestyle = ' ', ms=15,label='Mapped parking locations from SCP trips')
    plt.legend(loc='lower left')
    x, y, arrow_length = 0, 1, 0.1
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax.transAxes)
    scalebar = ScaleBar(1) # 1 pixel = 0.2 meter
    plt.gca().add_artist(scalebar)
    #ax.set_rasterized(True)
    #plt.savefig('rasterized_fig.eps')
    plt.show()


##### Plot SCP points ->
##if 1 == 1:
##    fig, ax = ox.plot_graph(ogp, show=False, close=False)
##    scpx = []
##    scpy = []
##    for item in SCP:
##        scpx.append(item[0])
##        scpy.append(item[1])
##    ax.plot(scpx,scpy, c='green',marker='$P$', linestyle = ' ', ms=15,label='Mapped parking locations from SCP trips')
##    plt.legend(loc='lower left')
##    x, y, arrow_length = 0, 1, 0.1
##    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
##            arrowprops=dict(facecolor='black', width=5, headwidth=15),
##            ha='center', va='center', fontsize=20,
##            xycoords=ax.transAxes)
##    scalebar = ScaleBar(1) # 1 pixel = 0.2 meter
##    plt.gca().add_artist(scalebar)
##    #ax.set_rasterized(True)
##    #plt.savefig('rasterized_fig.eps')
##    plt.show()
