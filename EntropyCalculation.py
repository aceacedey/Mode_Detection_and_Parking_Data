import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime,timedelta
import math
from math import exp, sqrt
import itertools
from collections import Counter, defaultdict
import scipy
from scipy import spatial
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
from sklearn.ensemble import RandomForestClassifier
from scipy import signal

plt.rcParams.update({'font.size': 18})

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


def mode_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy

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

dfTrainTest = pd.read_csv("TrainTestFile_filtered_CWAll.csv",index_col=0) # + str(file)

dfTrainTest.replace([np.inf, -np.inf], 0,inplace=True)
dfTrainTest.fillna(0,inplace=True)


dfTrainTest.replace([np.inf, -np.inf], 0)
dfTrainTest.fillna(0)
df = dfTrainTest.reset_index(drop=True)

Gdf = df.groupby('filename').head(n=1)
start_list = Gdf.index.values.tolist()

trainMean = []
trainStd = []
dci = []
te = []
DS = []
qMeanStdTest = []
Entropy = []
SDist = []
MDist = []

TRfiles = list(dfTrainTest.filename.unique())
for i_start,cfile in enumerate(TRfiles): #enumerate(start_list):#

    rawdf = df[df.filename == cfile]    
    pointsp = list(rawdf.pointsp.apply(lambda x: ast.literal_eval(x)))
    TIMELIST = [DTtoFloat(i) for i in rawdf.time.values]
    delTs = np.ediff1d(TIMELIST).tolist()
    #minTime = min(delTs)
    delTs.insert(0,0)

    Dist = [np.linalg.norm(np.array(x) - np.array(pointsp[idx-1])) for idx,x in enumerate(pointsp[1::])]

    MDist.append(np.mean(Dist))
    SDist.append(np.std(Dist))

    te.append(len(rawdf)/sum(delTs))
    DS.append(len(rawdf))
    ModeCounts = int(scipy.stats.mode(delTs)[1])
    dci.append(ModeCounts / len(rawdf))
    
    qMeanStdTest.append([np.mean(delTs),np.std(delTs)])
    Entropy.append(calculate_entropy(delTs))
    
    print(qMeanStdTest[-1])
    print(Entropy[-1])
    print(cfile)
    
sdf = pd.DataFrame({'Dataset':TRfiles,'Entropy':Entropy,'dci':dci,'MeanStd':qMeanStdTest,'TE':te,'Size':DS,'mdist':MDist,'sdist':SDist})
sdf[['mean','std']] = pd.DataFrame(sdf.MeanStd.to_list(), index= sdf.index)
sdf['ratio'] = sdf['std'].divide(sdf['mean'])
sdf.to_csv('TemporalFilteringAllData.csv')

####selection_criteria based on DCI and Entropy 
selected_files = sdf.Dataset[(sdf['mean'] < 5) & (sdf['ratio'] < 13) & (sdf['ratio'] >= 1) & (sdf.Entropy <= 1) & (sdf.dci > 0.8)].to_list()

#selected_files = sdf.Dataset[(sdf['std'].divide(sdf['mean']) <= 13) & (sdf['ratio'] >= 1) & (sdf.dci > 0.9)].to_list()

selecteddf = df[df.filename.isin(selected_files)]
selecteddf.to_csv("TrainTestFile_filtered_CWSelected.csv")


if 1 == 1:
        fig, ax1 = plt.subplots()
        import numpy.polynomial.polynomial as poly

        #y1 = dci
        y2 = Entropy
        idxs = np.argsort(dci)
        y1 = np.sort(dci)
        y2 = [Entropy[item] for item in idxs]
        
        ax1.plot(y1,y2,'ro', ms=5)
        
        coefs = poly.polyfit(y1[0:-1], y2[0:-1], 2)
        xs = np.linspace(0,max(y1), 2 * len(dci))
        ffit = poly.polyval(xs, coefs)
        ax1.plot(xs, ffit,'b', lw=3,label='Second order polynomial fit')
        ax1.set_xlabel('Data completeness index (DCI)')
        ax1.set_ylabel('Entropy of sampling intervals')
        ax1.legend(loc='upper right', frameon=False)
        plt.grid(True)
        plt.show() 

##if 1 == 1:
##    import os
##    import shutil
##    src = 'CWall'
##    dest = 'CWSelected'
##    src_files = selected_files#sdf.Dataset[(sdf.dci >= 0.3) & (sdf.Entropy <= 1) & (sdf.TE > 0.8)].to_list()
##    for file_name in src_files:
##        full_file_name = os.path.join(src, file_name)
##        if os.path.isfile(full_file_name):
##            shutil.copy(full_file_name, dest)


#PlotTime(kfir,kf1,'dataset number','prediction accuracy ','Effect of different samling interval of the  Kalman filer','irregular sampling','1 Hz sampling')

##
##if 1 == 1:
##    trEntropy = [0] * len(Entropy)
##
##    for iN,en in enumerate(Entropy):
##        Ent = np.array(Entropy)
##        temp = np.delete(Ent,iN)
##        trEntropy[iN] = sum(temp)/(len(Entropy) - 1)
##        print(sum(temp))
##    testMean = [item[0] for item in qMeanStdTest]
##    testStd = [item[1] for item in qMeanStdTest]
##    coef_of_variation = np.divide(testStd,testMean)
##    if 1 == 1:
##        fig, ax1 = plt.subplots()
##        ax2 = ax1.twinx()
##        x = np.arange(1,len(dci)+1)
##        y1 = dci
##        y2 = Entropy
##        ax1.plot(x, y1,marker='o', linestyle='--',color = 'r')
##        ax2.plot(x, y2,marker='*', linestyle='-',color='g')
##        ax1.set_xlabel('Dataset number')
##        ax1.set_xticks(x)
##        ax1.set_ylabel('Data completeness index', color='r')
##        ax2.set_ylabel('Entropy of sampling time intervals', color='g')
##        plt.grid(True)
##        plt.show()
##
##    if 1 == 1:
##        fig, ax1 = plt.subplots()
##        import numpy.polynomial.polynomial as poly
##
##        #y1 = dci
##        y2 = Entropy
##        idxs = np.argsort(dci)
##        y1 = np.sort(dci)
##        y2 = [Entropy[item] for item in idxs]
##        
##        ax1.plot(y1,y2,'ro', ms=5)
##        
##        coefs = poly.polyfit(y1[0:-1], y2[0:-1], 3)
##        xs = np.linspace(0,max(y1), 2 * len(dci))
##        ffit = poly.polyval(xs, coefs)
##        ax1.plot(xs, ffit,'b', lw=3,label='3rd order polynomial fit')
##        ax1.set_xlabel('Data completeness index (DCI)',color='g')
##        ax1.set_ylabel('Entropy of sampling time intervals', color='r')
##        ax1.legend(loc='upper right', frameon=False)
##        plt.grid(True)
##        plt.show()
    
##    if 1==1:
##        z = testMean
##        y = testStd
##        n = te
##        
##        fig, ax = plt.subplots()
##        ax.scatter(z,y)
##        ax.set_xlabel('Mean of sampling time intervals of a trip data')
##        ax.set_ylabel('Standard deviation of sampling intervals of a trip data')
##        #ax.set_title('Entropy vairies according to mean and standard deivation')
##        for i, txt in enumerate(n):
##            ax.annotate(round(txt,2), (z[i], y[i]))
##        plt.grid(True)
##        plt.show()
##        
##    Enfiles = []
##    Enfiles1 = []
##    stdByMean = np.divide(testStd,testMean)
##    for i, txt in enumerate(Entropy):
##        if (txt < 1 and y[i]/z[i] < 7 and y[i]/z[i] >= 1):
##            print(TRfiles[i])
##            Enfiles.append(TRfiles[i])
##    print('Based on Consistency:---->')
##    for i, txt in enumerate(dci):
##        if (txt >= 0.3):
##            print(TRfiles[i])
##            Enfiles1.append(TRfiles[i]) 
##
##print(Enfiles1)
##idx = [TRfiles.index(item) for item in Enfiles1]
##selectedTe = np.array(te)[idx]
###Entropy = [0.4321860267301245, 0.464737750098121, 0.5581574271499267, 0.07186459317307961, 0.5571212123258662, 0.25667957210817205, 0.010798536863905405, 0.06570255104453643, 0.01103269231880239, 0.01918855223744909, 0.06194708566965354, 0.008284627521671533, 0.9144844583632782, 0.6547206454267093, 0.48676682202692656, 0.3719220714910929, 0.08241710247545041, 0.5682344006774512]
##a = pd.DataFrame(np.reshape(selectedTe,(len(selectedTe),1)))
###np.reshape(selectedTe,(len(selectedTe),1))
##a.to_clipboard(excel=True)
##
####all Temporal error:
##allT = pd.DataFrame(np.reshape(te,(len(te),1)))
##allT.to_clipboard(excel=True)
##
##allS = pd.DataFrame(np.reshape(DS,(len(DS),1)))
##allS.to_clipboard(excel=True)
##
##        
