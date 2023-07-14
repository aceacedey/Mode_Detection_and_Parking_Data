
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

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm, metrics
import hdbscan


plt.rcParams.update({'font.size': 20})

FMT = '%H:%M:%S'
def DTtoFloat(dstring):
    x = datetime.strptime(dstring, FMT)
    return x.hour * 3600 + x.minute * 60 + x.second

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



def RF(X,y):

    X, y = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0, random_state=0, shuffle=False)
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)

    #RandomForestClassifier(...)
    #print(clf.predict([[0, 0, 0, 0]]))

def SWF(testdf,predDWT,delTs): ## SLIDING WINDOW FILTERING
    ## sequential sliding window filtering > Check for consecutive 0 and 1
    
    predictDWT = []
    lastGroupValue = -1 # create a flag to keep track of the window:
    #Grouped_item = [list(group) for idx, (key, group) in enumerate(itertools.groupby(predDWT))]
    end_index = 0
    for idx, (key, group) in enumerate(itertools.groupby(predDWT)):
        GrpList = list(group)
        #print(key, GrpList)
        start_index = end_index 
        end_index = start_index + len(GrpList) # end index is the number of elements in the grp
        tempTimeOfStay = sum(delTs[start_index:end_index])
        #print(tempTimeOfStay)
        if tempTimeOfStay < 60:
            if lastGroupValue != -1:
                newTempList = [lastGroupValue] * len(GrpList)
                predictDWT = predictDWT + newTempList
            else:
                 predictDWT = predictDWT + GrpList
        else:
            predictDWT = predictDWT + GrpList
        lastGroupValue = predictDWT[-1]

    return predictDWT 

def DWT_noise_filter(signal):
    ## DWT noise filtering of probabilities
    
    signalf = signal[:,1]
##    signal = predSVMProb[:,1]
##    
    rec = lowpassfilter(signalf, 0.6)
    lenDiff = abs(len(rec) - len(testY)) 

    recDWT = rec[:(len(rec) - lenDiff)]
    mean = 0.5 #recDWT.mean()
    
    predDWT = np.array([val > mean for val in recDWT]) * 1 ## enable/disable DWT based filtering
    return predDWT
   

def PlotTime(x,y,xlabel,title,ylabel,lg1,lg2,xticks):
    
    poix = list(range(1,len(x)+1))
    fig, ax = plt.subplots()
    ax.plot(poix,x, c='red',marker='*', linestyle='-',label=lg1)
    #for an in x1: 
        #ax.annotate(an,poi[an])
    ax.plot(poix,y,c='blue',marker='o', linestyle='-',label=lg2)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(1, len(x)+1))
    ax.legend(loc='lower right', frameon=False)
    plt.xticks(poix, xticks, rotation=90)
    plt.grid(True)
    #plt.ylim(0,1)
    plt.show()   

def CM(CMSVM):
    sen1 = CMSVM[0][0]/(CMSVM[0][0] + CMSVM[0][1])
    sen2 = CMSVM[1][1]/(CMSVM[1][1] + CMSVM[1][0])
    pr1 = CMSVM[0][0]/(CMSVM[0][0] + CMSVM[1][0])
    pr2 = CMSVM[1][1]/(CMSVM[1][1] + CMSVM[0][1])
    return np.round_([sen1,sen2,pr1,pr2],2)


dfTrainTest = pd.read_csv("TrainTestFile_filtered_CWSelected.csv",index_col=0) # + str(file)
#dfTrainTest.drop(['Unnamed: 0.1', 'Unnamed: 0.1.1'], axis=1,inplace=True)
dfTrainTest.replace([np.inf, -np.inf], 0,inplace=True)
dfTrainTest.fillna(0,inplace=True)


dfTrainTest.replace([np.inf, -np.inf], 0)
dfTrainTest.fillna(0)
df = dfTrainTest.reset_index(drop=True)

Gdf = df.groupby('filename').head(n=1)
start_list = Gdf.index.values.tolist()

TimeofStay = df.TimeOfStay.values
minTime = min(TimeofStay)

acSVM = []
acSVMf = []
acSVMdwt = []

acRF = []
acRFf = []
acRFdwt = []

acGB = []
acGBf = []
acGBdwt = []

acKM = []
acKMf = []

acHD = []
acHDf = []
acHDdwt = []

fn = []

trainMean = []
trainStd = []
dci = []
DS = []
te = []
qMeanStdTest = []
Entropy = []

CMSVMf = np.zeros((2,2))
CMSVM = np.zeros((2,2))

CMRFf = np.zeros((2,2))
CMRF = np.zeros((2,2))

CMGBf = np.zeros((2,2))
CMGB = np.zeros((2,2))

clsSVM = svm.SVC(probability=True,kernel='linear')#'rbf',gamma='auto')
clsSVMf = svm.SVC(probability=True,kernel='linear')#'rbf',gamma='auto')
clsRF = RandomForestClassifier(random_state=0)
clsRFf = RandomForestClassifier(random_state=0)
clsGB = GradientBoostingClassifier(random_state=0)
clsGBf = GradientBoostingClassifier(random_state=0)


TRfiles = list(dfTrainTest.filename.unique())
for i_start,cfile in enumerate(TRfiles): #enumerate(start_list):#

    testdf = df[df.filename == cfile]    
    
    TIMELIST = [DTtoFloat(i) for i in testdf.time.values]
    delTs = np.ediff1d(TIMELIST).tolist()
    #minTime = min(delTs)
    delTs.insert(0,0)
    
    testY = testdf.observed.values #
    testX = testdf[['PointVelocityPrj','PointAccPrj','PointAnglePrj']].values
    
    
    testX =  testX/testX.max(axis=0)
    testXf = testdf[['velocityFiltered','accelFiltered','AngleFiltered']].values

    testXf =  testXf/testXf.max(axis=0)
    
    ### define train dataset
    traindf = df.drop(testdf.index.values, axis=0)
    traindf = traindf[traindf.PointVelocityPrj != 0]

    trainY = traindf.observed.values 

    trainX = traindf[['PointVelocityPrj','PointAccPrj','PointAnglePrj']].values
    trainX =  trainX/trainX.max(axis=0)
  
    trainXf = traindf[['velocityFiltered','accelFiltered','AngleFiltered']].values
    trainXf =  trainXf/trainXf.max(axis=0)


       #### 
    clsSVM.fit(trainX, trainY)
    clsSVMf.fit(trainXf, trainY)

    clsRF.fit(trainX, trainY)
    clsRFf.fit(trainXf, trainY)
    
    clsGB.fit(trainX, trainY)
    clsGBf.fit(trainXf, trainY)
    ##################
     ### prediction using SVM
    
    predSVM = clsSVM.predict(testX)
    #predSVMProb = clsSVM.predict_proba(testX)

    predSVMf = clsSVMf.predict(testXf)
    predSVMfProb = clsSVMf.predict_proba(testXf)
    
    ### prediction using RF
    
    predRF = clsRF.predict(testX)
    #predRFProb = clsRF.predict_proba(testX)
    
    predRFf = clsRFf.predict(testXf)
    predRFfProb = clsRFf.predict_proba(testXf)

    ### prediction using GB
    predGB = clsGB.predict(testX)
    #predGBProb = clsGB.predict_proba(testX)
    predGBf = clsGBf.predict(testXf)
    predGBfProb = clsGBf.predict_proba(testXf)


    ### append scores SVM
    acSVM.append(metrics.accuracy_score(testY,y_pred=predSVM))
    y_pred_SVMf = SWF(testdf,predSVMf,delTs)
    acSVMf.append(metrics.accuracy_score(testY,y_pred=y_pred_SVMf))

    SVMdwt = DWT_noise_filter(predSVMfProb)
    y_pred_SVMdwt = SWF(testdf,SVMdwt,delTs)
    acSVMdwt.append(metrics.accuracy_score(testY,y_pred=y_pred_SVMdwt))

    
    ### append scores RF
    acRF.append(metrics.accuracy_score(testY,y_pred=predRF))    
    y_pred_RFf = SWF(testdf,predRFf,delTs)
    acRFf.append(metrics.accuracy_score(testY,y_pred=y_pred_RFf))

    RFdwt = DWT_noise_filter(predRFfProb)
    y_pred_RFdwt = SWF(testdf,RFdwt,delTs)
    acRFdwt.append(metrics.accuracy_score(testY,y_pred=y_pred_RFdwt))


    ### append scores GB
    acGB.append(metrics.accuracy_score(testY,y_pred=predGB))    
    y_pred_GBf = SWF(testdf,predGBf,delTs)
    acGBf.append(metrics.accuracy_score(testY,y_pred=y_pred_GBf))

    GBdwt = DWT_noise_filter(predGBfProb)
    y_pred_GBdwt = SWF(testdf,GBdwt,delTs)
    acGBdwt.append(metrics.accuracy_score(testY,y_pred=y_pred_GBdwt))
    ###############
    
    te.append(len(testdf)/sum(delTs))
    ModeCounts = int(scipy.stats.mode(delTs)[1])
    dci.append(ModeCounts / len(testdf))
    DS.append(len(testdf))
    qMeanStdTest.append([np.mean(delTs),np.std(delTs)])
    Entropy.append(calculate_entropy(delTs))
    print(acSVM[-1])
    print(acSVMf[-1])
    print(acRF[-1])
    print(acRFf[-1])
    print(cfile)
       
    CMSVMf = np.add(CMSVMf,confusion_matrix(testY,y_pred_SVMf))
    CMSVM = np.add(CMSVM,confusion_matrix(testY,predSVM))

    CMRFf = np.add(CMRFf,confusion_matrix(testY,y_pred_RFf))
    CMRF = np.add(CMRF,confusion_matrix(testY,predRF))

    CMGBf = np.add(CMGBf,confusion_matrix(testY,y_pred_GBf))
    CMGB = np.add(CMGB,confusion_matrix(testY,predGB))


errordf = pd.DataFrame({'Dataset':TRfiles,'acRF':acRF,'acRFf':acRFf,'acRFdwt':acRFdwt,'acSVM':acSVM,'acSVMf':acSVMf,'acSVMdwt':acSVMdwt,'acGB':acGB,
                        'acGBf':acGBf,'acGBdwt':acGBdwt,
                      'Entropy':Entropy,'dci':dci,'MeanStd':qMeanStdTest,'TE':te})
errordf.to_csv('errordf_Subset_only.csv')


if 1 == 1:
    print(CMSVM)
    print(CMSVMf)
    print(CMRF)
    print(CMRFf)
    print(CMGB)
    print(CMGBf)
    
    print(CM(CMSVM))
    print(CM(CMSVMf))
    print(CM(CMRF))
    print(CM(CMRFf))
    print(CM(CMGB))
    print(CM(CMGBf))
    
    print(errordf.acRFf.mean())
    print(errordf.acRF.mean())
    print(errordf.acSVMf.mean())
    print(errordf.acSVM.mean())
    print(errordf.acGBf.mean())
    print(errordf.acGB.mean())
      
