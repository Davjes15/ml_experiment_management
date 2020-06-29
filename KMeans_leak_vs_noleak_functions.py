#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import pickle
from scipy.fftpack import fft
import gc
import matplotlib.ticker as ticker


from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn import metrics

# To see multiple outputs from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# # Function definition

# In[13]:


def prep_metadata():
    path =  "//de-s-0124116.crc.de.abb.com/public/Studenten/2019/MarcoRettig/4 Data/Acoustic Leak Detection"
    sheets = {1: "1.4. Medium Leak", 2: " 3.6. M.L. (hydrostatic flow)", 3: "3.6. M.L. (long meas)"}

    metadata = pd.DataFrame()
    for item in sheets:
        #print(sheets[item])
        new_data = pd.read_excel(path + "/List of all Measurements3.xlsx", sheet_name=sheets[item])
        metadata = pd.concat([metadata, new_data], ignore_index=True, sort=False)
    
    return metadata


def getLeakType(metadata, pickleFileName):
    leakType = metadata[metadata["Parent Folder"] == pickleFileName]["Leak"].values[0]
        
    if (leakType == "leak"):
        leakType = "medium-leak"
    elif (leakType == "no leak" or leakType == "no-leak"):
        leakType = "no-leak"
    elif (leakType == "no leak - leak - no leak"):
        leakType = "no-leak + medium-leak"
    else:
        leakType = "error"
        
    return leakType


def collect_all_ts_data(data_origin):
    assert data_origin in ['04_2019', 'Hydrostatic_Flow'], "Argument must be '04_2019' or 'Hydrostatic_Flow'"
    InputBaseFolder =  "//de-s-0124116.crc.de.abb.com/public/Studenten/2019/MarcoRettig/4 Data/Acoustic Leak Detection/Raw_Data/" + data_origin 
    all_files_orig = glob.glob(InputBaseFolder + "/*.pkl", recursive=True)
    
    all_files_mod = all_files_orig[:]
    for i in range(len(all_files_orig)): 
        time = os.path.basename(all_files_orig[i]).split('.')[0].split('-')
        all_files_mod[i] = time[3] + '-' + time[4] + '-' + time[5] 
    
    return all_files_orig, all_files_mod
    

def collect_all_fft_data(data_origin):
    assert data_origin in ['04_2019', 'Hydrostatic_Flow'], "Argument must be '04_2019' or 'Hydrostatic_Flow'"
    InputBaseFolder =  "//de-s-0124116.crc.de.abb.com/public/Studenten/2019/MarcoRettig/4 Data/Acoustic Leak Detection/FFT_Data/until_15kHz/" + data_origin 
    paths = glob.glob(InputBaseFolder + "/*", recursive=True)

    all_files = []
    for path in paths:
        file_folders = glob.glob(path + "/*", recursive=True)
        for file_folder in file_folders:
            files = glob.glob(file_folder + "/*.pkl", recursive=True)
            all_files += files
            
    return all_files


def select_data(metadata, in_files, dtype):
    assert dtype in ['medium-leak', 'leak', 'no-leak', 1500, 2000, 2500], "Argument must be 'medium-leak', 'leak', 'no-leak', 1500, 2000 or 2500"
    out_files = []
    
    if dtype == 'medium-leak' or dtype == 'leak':
        for i in range(len(in_files)):
            if metadata[metadata["Parent Folder"] == os.path.basename(in_files[i]).split('.')[0]]['Leak'].values[0] == 'leak': 
                out_files.append(in_files[i])  
    elif dtype == 'no-leak':
        for i in range(len(in_files)):
            if metadata[metadata["Parent Folder"] == os.path.basename(in_files[i]).split('.')[0]]['Leak'].values[0] == 'no leak': 
                out_files.append(in_files[i]) 
            elif metadata[metadata["Parent Folder"] == os.path.basename(in_files[i]).split('.')[0]]['Leak'].values[0] == 'no-leak': 
                out_files.append(in_files[i]) 
    else:
        for i in range(len(in_files)):
            if metadata[metadata["Parent Folder"] == os.path.basename(in_files[i]).split('.')[0]]['Pump Speed (RPM)'].values[0] == dtype: 
                out_files.append(in_files[i])
            
    return out_files


def divide_raw_data(in_file, time_interval):
    '''time_interval: Desired length of subdatasets in seconds'''
    assert type(in_file) == pd.DataFrame, "1st argument must be a Pandas DataFrame"
    idx_interval = int(time_interval * 1e6)
    num_intervals = int(len(in_file)/idx_interval)
    ts_subdata = {}
    for i in range(num_intervals):
        ts_subdata[i] = in_file.iloc[i*idx_interval:(i+1)*idx_interval] 
        
    return ts_subdata, idx_interval


def gen_fft_data(ts_subdata_dict, data_set_length, time_interval):
    assert type(ts_subdata_dict) == type({}), "Argument must be a dictionary"
    up_freq_lim = int(15000 * time_interval) #corresponds to frequency range up to 15 kHz
    T = 1e-6 #time step size of ts_datasets [s]
    N = int(data_set_length)
    channels = ['ch0', 'ch1', 'ch4']
    xf = np.linspace(0.0, 1/(2*T), 1+N//2)[1:]
    
    fft_subdata = {}
    for item in ts_subdata_dict:
        fft_data = pd.DataFrame(columns = ['frequency'] + channels) 
        fft_data['frequency'] = xf[0:up_freq_lim]
        for channel in channels:
            yf = 2.0/N * np.abs(fft(ts_subdata_dict[item][channel])[0:N//2])[1:]
            fft_data[channel] = yf[:up_freq_lim]
        fft_subdata[item] = fft_data
        
    return fft_subdata


def divide_raw_data_gen_fft_data(in_file, time_interval):
    ts_subdatasets, data_set_length = divide_raw_data(in_file, time_interval)
    fft_subdatasets = gen_fft_data(ts_subdatasets, data_set_length, time_interval)
    
    return fft_subdatasets


def plot_real_vs_kmeans(clustered_data, data_origin, data_spec, scaler, channel, low_freq, up_freq, real_leaktypes, cluster_labels, inv_cluster_centers, score_eu, up_ylim, metadata, time_interval):
    assert data_origin in ['04_2019', 'Hydrostatic_Flow', 'Long_Measurement'], "2nd argument must be '04/2019', 'Hydrostatic_Flow' or 'Long_Measurement'"
    #assert data_spec in ['all', '1500rpm', '2000rpm', '2500rpm'], "3rd argument must be 'all', '1500rpm', '2000rpm' or '2500rpm'"
    assert scaler in ['MinMax', 'Standard', 'Normalizer'], "4th argument must be 'MinMax', 'Standard' or 'Normalizer'"
    assert channel in ['ch0', 'ch1'], "5th argument must be 'ch0' or 'ch1'"
    up_ylim_centroids = 1.05 * np.max(inv_cluster_centers)
    #up_ylim_centroids = up_ylim

    lower_xlim = -0.3
    upper_xlim = 15.3
    fig = plt.figure(figsize=(20,24))
    axes = fig.subplots(3, 2)
    x = np.array((clustered_data.columns[3:]))/1000

    times_real_left = []
    times_real_right = []
    times_kmeans_left = []
    times_kmeans_right = []
    #a, b, c, d = 0, 0, 0, 0 #Variables ensuring identical alpha values for different subplots
    for i in range(len(cluster_labels)):
        y = clustered_data.iloc[i][3:].values
        time = clustered_data.iloc[i][0].split('-')[3] + '-' + clustered_data.iloc[i][0].split('-')[4] + '-' + clustered_data.iloc[i][0].split('-')[5]
        section = str(clustered_data.iloc[i, 1]) 
        leaktype = getLeakType(metadata, clustered_data.iloc[i][0])
        pump_speed = str(int(metadata[metadata["Parent Folder"] == clustered_data.iloc[i][0]]['Pump Speed (RPM)'].values[0])) + 'rpm'
        
        label = time + '  ' + leaktype + '  ' + pump_speed
        
        if real_leaktypes[i] == 0:
            if time in times_real_left:
                #print("time already in times_real_left")
                axes[0, 0].plot(x, y, label='_nolegend_', alpha=1)
            elif time not in times_real_left:
                axes[0, 0].plot(x, y, label=label, alpha=1)
                times_real_left.append(time)
            #a += 1
        elif real_leaktypes[i] == 1:
            if time in times_real_right:
                #print("time already in times_real_right")
                axes[0, 1].plot(x, y, label='_nolegend_', alpha=1)
            elif time not in times_real_right:
                axes[0, 1].plot(x, y, label=label, alpha=1)
                times_real_right.append(time)
            #b += 1
        if cluster_labels[i] == 0:
            if time in times_kmeans_left:
                #print("time already in times_kmeans_left")
                axes[1, 0].plot(x, y, label='_nolegend_', alpha=1)  
            elif time not in times_kmeans_left:
                axes[1, 0].plot(x, y, label=label, alpha=1)
                times_kmeans_left.append(time)
            #c += 1
        elif cluster_labels[i] == 1:
            if time in times_kmeans_right:
                #print("time already in times_kmeans_right")
                axes[1, 1].plot(x, y, label='_nolegend_', alpha=1)  
            elif time not in times_kmeans_right:
                axes[1, 1].plot(x, y, label=label, alpha=1)
                times_kmeans_right.append(time)
            #d += 1
    
    if time_interval > 5.25:
        idx_low = int(low_freq * 10.5)
        idx_up = int(up_freq * 10.5)   
    else:
        idx_low = int(low_freq * time_interval)
        idx_up = int(up_freq * time_interval)  
        
    legend_fontsize = 14
    tick_fontsize = 11
     
    #exp1 = np.zeros(idx_low)
    #exp2 = np.zeros(len(clustered_data.columns) - idx_up)
    #axes[2, 0].plot(x, np.concatenate((exp1, inv_cluster_centers[0, :], exp2)))
    #axes[2, 1].plot(x, np.concatenate((exp1, inv_cluster_centers[1, :], exp2)))
    axes[2, 0].plot(x[idx_low:idx_up], inv_cluster_centers[0, :])
    axes[2, 1].plot(x[idx_low:idx_up], inv_cluster_centers[1, :])   

    axes[0, 0].set_title('Real classification: medium-leak (0)', fontsize=18)
    axes[0, 0].legend(fontsize=legend_fontsize)
    axes[0, 0].set_xlabel('Frequency [kHz]', fontsize=14)
    axes[0, 0].set_ylabel('Amplitude', fontsize=14)
    axes[0, 0].set_xlim(lower_xlim, upper_xlim)
    axes[0, 0].set_ylim(0, up_ylim)
    axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[0, 0].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axes[0, 0].grid(which='both', linestyle=':', linewidth=1)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=tick_fontsize)


    axes[0, 1].set_title('Real classification: no-leak (1)', fontsize=18)
    axes[0, 1].legend(fontsize=legend_fontsize)
    axes[0, 1].set_xlabel('Frequency [kHz]', fontsize=14)
    #axes[0, 1].set_ylabel('Amplitude', fontsize=14)
    axes[0, 1].set_xlim(lower_xlim, upper_xlim)
    axes[0, 1].set_ylim(0, up_ylim)  
    axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[0, 1].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axes[0, 1].grid(which='both', linestyle=':', linewidth=1)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
      
    axes[1, 0].set_title('KMeans classification: Cluster 1 (0)', fontsize=18)
    axes[1, 0].legend(fontsize=legend_fontsize)
    axes[1, 0].set_xlabel('Frequency [kHz]', fontsize=14)
    axes[1, 0].set_ylabel('Amplitude', fontsize=14)
    axes[1, 0].set_xlim(lower_xlim, upper_xlim)
    axes[1, 0].set_ylim(0, up_ylim)
    axes[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[1, 0].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axes[1, 0].grid(which='both', linestyle=':', linewidth=1)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
 
    axes[1, 1].set_title('KMeans classification: Cluster 2 (1)', fontsize=18)
    axes[1, 1].set_xlabel('Frequency [kHz]', fontsize=14)
    #ax2.set_ylabel('Amplitude')
    axes[1, 1].legend(fontsize=legend_fontsize)
    axes[1, 1].set_xlim(lower_xlim, upper_xlim)
    axes[1, 1].set_ylim(0, up_ylim)
    axes[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[1, 1].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axes[1, 1].grid(which='both', linestyle=':', linewidth=1)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    axes[2, 0].set_title('Centroid of Cluster 1', fontsize=18)
    axes[2, 0].set_xlabel('Frequency [kHz]', fontsize=14)
    axes[2, 0].set_ylabel('Amplitude', fontsize=14)
    #axes[2, 0].set_xlim(lower_xlim, upper_xlim)
    if scaler != 'Normalizer':
        axes[2, 0].set_ylim(0, up_ylim_centroids)
    #axes[2, 0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    #axes[2, 0].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axes[2, 0].grid(which='both', linestyle=':', linewidth=1)
    axes[2, 0].tick_params(axis='both', which='major', labelsize=tick_fontsize)

    axes[2, 1].set_title('Centroid of Cluster 2', fontsize=18)
    axes[2, 1].set_xlabel('Frequency [kHz]', fontsize=14)
    #axes[2, 1].set_xlim(lower_xlim, upper_xlim)
    if scaler != 'Normalizer':
        axes[2, 1].set_ylim(0, up_ylim_centroids)
    #axes[2, 1].xaxis.set_major_locator(ticker.MultipleLocator((up_freq - low_freq)/1000))
    #axes[2, 1].xaxis.set_minor_locator(ticker.MultipleLocator((up_freq - low_freq)/10000))
    axes[2, 1].grid(which='both', linestyle=':', linewidth=1)
    axes[2, 1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    plt.figtext(0.5, 0.08, "Euclidean silhouette score: {}".format(round(score_eu, 2)), wrap=True,
            horizontalalignment='center', fontsize=18)

    if time_interval > 5.25:
        data_set_length = 10.5
    else:
        data_set_length = time_interval
    
    n_sections = int(10.5/time_interval)
    
    #fig.suptitle("KMeans clustering of  {}  {}  {}  FFT data (scaled using '{}Scaler')\nFrequency range used for clustering: {} - {} kHz\nLength of data sets: {} s\nReal vs. KMeans classification".format(data_origin, data_spec, channel, scaler, round(low_freq/1000, 1), round(up_freq/1000, 1), data_set_length), fontsize=20, y=0.96)
    print("Saving plot ...")
    plt.savefig('{}_{}_{}_FFT_data_KMeans_leak_noleak_{}Scaler_{}_to_{}_Hz_{}_section(s)_real_vs_kmeans_classification.pdf'.format(data_origin, data_spec, channel, scaler, low_freq, up_freq, n_sections), bbox_inches='tight', format='PDF')
    plt.savefig('{}_{}_{}_FFT_data_KMeans_leak_noleak_{}Scaler_{}_to_{}_Hz_{}_section(s)_real_vs_kmeans_classification.jpg'.format(data_origin, data_spec, channel, scaler, low_freq, up_freq, n_sections), bbox_inches='tight', format='jpg')
    

def plot_false_pos_neg(clustered_data, data_origin, data_spec, scaler, channel, low_freq, up_freq, real_leaktypes, cluster_labels, up_ylim, metadata, time_interval):
    leak_minus_cluster1 = []
    noleak_minus_cluster2 = []
    cluster1_minus_leak = []
    cluster2_minus_noleak = []
    fig = plt.figure(figsize=(20,16))
    axes = fig.subplots(2, 2)
    x = np.array((clustered_data.columns[3:]))/1000
    
    lower_xlim = -0.3
    upper_xlim = 15.3

    for i in range(len(cluster_labels)):
        y = clustered_data.iloc[i][3:].values
        time = clustered_data.iloc[i][0].split('-')[3] + '-' + clustered_data.iloc[i][0].split('-')[4] + '-' + clustered_data.iloc[i][0].split('-')[5]
        section = str(clustered_data.iloc[i, 1])
        leaktype = getLeakType(metadata, clustered_data.iloc[i][0])
        pump_speed = str(int(metadata[metadata["Parent Folder"] == clustered_data.iloc[i][0]]['Pump Speed (RPM)'].values[0])) + 'rpm'
        
        label = time + '  ' + leaktype + '  ' + pump_speed
        
        if real_leaktypes[i] == 0:
            if cluster_labels[i] != 0:
                if time in leak_minus_cluster1:
                    axes[0, 0].plot(x, y, label ='_nolegend_')
                elif time not in leak_minus_cluster1:
                    axes[0, 0].plot(x, y, label =label)
                    leak_minus_cluster1.append(time)
        elif real_leaktypes[i] == 1:
            if cluster_labels[i] != 1:
                if time in noleak_minus_cluster2:
                    axes[0, 1].plot(x, y, label ='_nolegend_')
                elif time not in noleak_minus_cluster2:
                    axes[0, 1].plot(x, y, label =label)
                    noleak_minus_cluster2.append(time)
    
        if cluster_labels[i] == 0:
            if real_leaktypes[i] != 0:
                if time in cluster1_minus_leak:
                    axes[1, 0].plot(x, y, label ='_nolegend_')
                elif time not in cluster1_minus_leak:
                    axes[1, 0].plot(x, y, label =label)
                    cluster1_minus_leak.append(time)
        elif cluster_labels[i] == 1:
            if real_leaktypes[i] != 1:
                if time in cluster2_minus_noleak:
                    axes[1, 1].plot(x, y, label ='_nolegend_')
                elif time not in cluster2_minus_noleak:
                    axes[1, 1].plot(x, y, label =label)
                    cluster2_minus_noleak.append(time)
        
    axes[0, 0].set_title("Data present in 'medium-leak' but not in 'Cluster 1'", fontsize=18)
    axes[0, 0].legend()
    axes[0, 0].set_xlabel('Frequency [kHz]', fontsize=14)
    axes[0, 0].set_ylabel('Amplitude', fontsize=14)
    axes[0, 0].set_xlim(lower_xlim, upper_xlim)
    axes[0, 0].set_ylim(0, up_ylim)
    axes[0, 0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[0, 0].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axes[0, 0].grid(which='both', linestyle=':', linewidth=1)

    axes[0, 1].set_title("Data present in 'no-leak' but not in 'Cluster 2'", fontsize=18)
    axes[0, 1].legend()
    axes[0, 1].set_xlabel('Frequency [kHz]', fontsize=14)
    #axes[0, 1].set_ylabel('Amplitude', fontsize=14)
    axes[0, 1].set_xlim(lower_xlim, upper_xlim)
    axes[0, 1].set_ylim(0, up_ylim)  
    axes[0, 1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[0, 1].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axes[0, 1].grid(which='both', linestyle=':', linewidth=1)
      
    axes[1, 0].set_title("Data present in 'Cluster 1' but not in 'medium-leak'", fontsize=18)
    axes[1, 0].legend()
    axes[1, 0].set_xlabel('Frequency [kHz]', fontsize=14)
    axes[1, 0].set_ylabel('Amplitude', fontsize=14)
    axes[1, 0].set_xlim(lower_xlim, upper_xlim)
    axes[1, 0].set_ylim(0, up_ylim)
    axes[1, 0].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[1, 0].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axes[1, 0].grid(which='both', linestyle=':', linewidth=1)

    axes[1, 1].set_title("Data present in 'Cluster 2' but not in 'no-leak'", fontsize=18)
    axes[1, 1].legend()
    axes[1, 1].set_xlabel('Frequency [kHz]', fontsize=14)
    #axes[0, 1].set_ylabel('Amplitude', fontsize=14)
    axes[1, 1].set_xlim(lower_xlim, upper_xlim)
    axes[1, 1].set_ylim(0, up_ylim)   
    axes[1, 1].xaxis.set_major_locator(ticker.MultipleLocator(1))
    axes[1, 1].xaxis.set_minor_locator(ticker.MultipleLocator(0.5))
    axes[1, 1].grid(which='both', linestyle=':', linewidth=1)

    #print("Data present in 'medium-leak' but not in 'Cluster 1': {} \n".format(leak_minus_cluster1))
    #print("Data present in 'no-leak' but not in 'Cluster 2': {}".format(noleak_minus_cluster2))
    #print("Data present in 'Cluster 1' but not in 'medium-leak': {} \n".format(cluster1_minus_leak))
    #print("Data present in 'Cluster 2' but not in 'no-leak': {}".format(cluster2_minus_noleak))
    
    if time_interval > 5.25:
        data_set_length = 10.5
    else:
        data_set_length = time_interval
    
    n_sections = int(10.5/time_interval)
    
    fig.suptitle("KMeans clustering of  {}  {}  {}  FFT data (scaled using '{}Scaler')\nFrequency range used for clustering: {} - {} kHz\nLength of data sets: {} s\nFalse negatives vs. false positives".format(data_origin, data_spec, channel, scaler, round(low_freq/1000, 1), round(up_freq/1000, 1), data_set_length), fontsize=20, y=1.0)
    print("Saving plot ...")
    plt.savefig('{}_{}_{}_FFT_data_KMeans_leak_noleak_{}Scaler_{}_to_{}_Hz_{}_section(s)_false_neg_vs_false_pos.pdf'.format(data_origin, data_spec, channel, scaler, low_freq, up_freq, n_sections), format='PDF')

    
def prepare_clustering_data(data_origin, data_spec, time_interval, channel):
    metadata = prep_metadata()
    
    if time_interval > 5.25:
        all_files = collect_all_fft_data(data_origin)
        if data_spec == 'all':
            in_files = all_files
        elif data_spec == '1500rpm':
            in_files = select_data(metadata, all_files, 1500)    
        elif data_spec == '2000rpm':
            in_files = select_data(metadata, all_files, 2000)
        elif data_spec == '2500rpm':
            in_files = select_data(metadata, all_files, 2500)
            
        ref_file = pd.read_pickle(in_files[0])
        columns = ['file', 'section', 'Leak'] + [freq for freq in ref_file['frequency']]
        clusteringData = pd.DataFrame(columns = columns, index = [i for i in range(len(in_files))])
    
        for i in range(len(in_files)):
            fft_data = pd.read_pickle(in_files[i])
            clusteringData.iloc[i, :1] = os.path.basename(in_files[i]).split('.')[0]
            clusteringData.iloc[i, 1:2] = 'all'
            clusteringData.iloc[i, 2:3] = getLeakType(metadata, os.path.basename(in_files[i]).split('.')[0])
            clusteringData.iloc[i, 3:] = fft_data[channel].values
            #clusteringData.iloc[i, 2+idx_low : 2+idx_up] = fft_data[channel][idx_low:idx_up].values
            
    else:
        all_files, discard = collect_all_ts_data(data_origin)
        if data_spec == 'all':
            in_files = all_files
        elif data_spec == '1500rpm':
            in_files = select_data(metadata, all_files, 1500)    
        elif data_spec == '2000rpm':
            in_files = select_data(metadata, all_files, 2000)
        elif data_spec == '2500rpm':
            in_files = select_data(metadata, all_files, 2500)
        else:
            in_files = []
            in_files.append(np.asarray(all_files)[np.asarray(discard) == data_spec][0])
            
        divided_fft_in_files = {}
        for ts_file in in_files:
            sample = os.path.basename(ts_file).split('.')[0]
            #time_snippets = os.path.basename(ts_file).split('.')[0].split('-')
            #sample = time_snippets[3] + '-' + time_snippets[4] + '-' + time_snippets[5] 
            ts_dataframe = pd.read_pickle(ts_file)
            divided_fft_in_files[sample] = divide_raw_data_gen_fft_data(ts_dataframe, time_interval)
            
        first_key = list(divided_fft_in_files.keys())[0]     
        columns = ['file', 'section', 'Leak'] + [freq for freq in divided_fft_in_files[first_key][0]['frequency']]
        indices = 0
        for sample in divided_fft_in_files:
            indices += len(divided_fft_in_files[sample])
        clusteringData = pd.DataFrame(columns = columns, index = [i for i in range(indices)])
        
        i = 0
        for sample in divided_fft_in_files:
            section_cnt = 1
            for section in divided_fft_in_files[sample]:
                clusteringData.iloc[i, :1] = sample
                clusteringData.iloc[i, 1:2] = section_cnt
                clusteringData.iloc[i, 2:3] = getLeakType(metadata, sample)
                clusteringData.iloc[i, 3:] = divided_fft_in_files[sample][section][channel].values
                section_cnt += 1
                i += 1
            
        #print(clusteringData)
  
    return metadata, in_files, clusteringData
    
    
def kmeans_clustering(data_origin, data_spec, time_interval, channel, low_freq, up_freq, scalertype, up_ylim):
    assert data_origin in ['04_2019', 'Hydrostatic_Flow'], "1st argument must be '04_2019' or 'Hydrostatic_Flow'"
    if data_origin == 'Hydrostatic_Flow':
        assert data_spec == 'all', "If 'Hydrostatic_Flow' is chosen as 1st argument, 2nd argument must be 'all'"
    discard, file_list = collect_all_ts_data(data_origin)
    file_groups = ['all', '1500rpm', '2000rpm', '2500rpm'] 
    selection_options = file_list + file_groups 
    if time_interval > 5.25:
        assert data_spec in file_groups, "If 3rd argument (time_interval) is greater than 5.25, 2nd argument must be one of {}".format(file_groups)
    assert data_spec in selection_options, "2nd argument must be one of {}".format(selection_options)
    assert time_interval <= 10.5, "3rd argument must be 10.5 or smaller"
    assert channel in ['ch0', 'ch1'], "4th argument must be 'ch0' or 'ch1'"
    assert 0 <= low_freq and low_freq < up_freq and up_freq <= 15000 , "Lower frequency must be smaller than upper frequency and both frequencies must range between 0 and 15000 Hz"
    assert scalertype in ['MinMax', 'Standard', 'Normalizer'], "7th argument must be 'MinMax', 'Standard' or 'Normalizer'"
    
    print("Performing KMeans clustering of {} {} {} FFT data using '{}Scaler' ...\n".format(data_origin, data_spec, channel, scalertype))
    if time_interval > 5.25:  
        print("FFT transformation will be performed on chosen raw data sets with length of 10.5 seconds, as the specified time interval is greater than 5.25 seconds.\n")
        
    metadata, in_files, clusteringData = prepare_clustering_data(data_origin, data_spec, time_interval, channel) 
    
    print("Input files:")
    for item in in_files:
        print(item)
    print("\n")
             
    labelEncoder = LabelEncoder()
    labelEncoder.fit(clusteringData['Leak'])
    clusteringData['Leak'] = labelEncoder.transform(clusteringData['Leak'])
    
    if time_interval > 5.25:
        idx_low = int(low_freq * 10.5)
        idx_up = int(up_freq * 10.5)   
    else:
        idx_low = int(low_freq * time_interval)
        idx_up = int(up_freq * time_interval)      
    leak = np.array(clusteringData['Leak'])
    X = np.array(clusteringData.drop(['file', 'section', 'Leak'], axis=1).astype(float))[:, idx_low:idx_up]
    print("Shape of Clustering input Matrix (samples, features): {}".format(X.shape))
    
    kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
                    random_state=None, tol=0.0001, verbose=0)

    if scalertype == 'MinMax':
        scaler = MinMaxScaler()
    elif scalertype == 'Standard':
        scaler = StandardScaler()
    elif scalertype == 'Normalizer':
        scaler = Normalizer()
    
    X_scaled = scaler.fit_transform(X)
    kmeans.fit(X_scaled)
    
    scaled_clusterCenters = kmeans.cluster_centers_
    if scalertype == 'Normalizer':
        inv_clusterCenters = scaled_clusterCenters
    else:
        inv_clusterCenters = scaler.inverse_transform(scaled_clusterCenters)
    clusterLabels = kmeans.labels_
    inertia = kmeans.inertia_ # Sum of distances of samples to their closest cluster center

    score_eu = metrics.silhouette_score(X_scaled, clusterLabels, metric='euclidean')
    score_sqeu = metrics.silhouette_score(X_scaled, clusterLabels, metric='sqeuclidean')
    print("Euclidian silhouette score: {} \nSquared euclidian silhouette score: {} \nInertia: {}".format(score_eu, score_sqeu, inertia))
    print("Real classification: {} \nKMeans classification: {}\n".format(leak, clusterLabels))
    
    plot_real_vs_kmeans(clusteringData, data_origin, data_spec, scalertype, channel, low_freq, up_freq, leak, clusterLabels, inv_clusterCenters, score_eu, up_ylim, metadata, time_interval)
    plot_false_pos_neg(clusteringData, data_origin, data_spec, scalertype, channel, low_freq, up_freq, leak, clusterLabels, up_ylim, metadata, time_interval)
    
    results = pd.DataFrame(columns=['filename', 'section', 'known label', 'pump speed [rpm]', 'assigned cluster'], index = [i for i in range(len(clusteringData))])
    for i in range(len(clusteringData)):
        results.iloc[i]['filename'] = clusteringData.iloc[i, 0]
        results.iloc[i]['section'] = clusteringData.iloc[i, 1]
        results.iloc[i]['known label'] = getLeakType(metadata, clusteringData.iloc[i, 0]) + ' ' + '(' + str(clusteringData.iloc[i, 2]) + ')'
        results.iloc[i]['pump speed [rpm]'] = metadata[metadata["Parent Folder"] == clusteringData.iloc[i, 0]]['Pump Speed (RPM)'].values[0]
        results.iloc[i]['assigned cluster'] = clusterLabels[i]
    
    n_sections = int(10.5/time_interval)
    
    results.set_index("filename", inplace=True)
    print("Saving .csv file ...")
    results.to_csv('{}_{}_{}_FFT_data_KMeans_leak_noleak_{}Scaler_{}_to_{}_Hz_{}_section(s).csv'.format(data_origin, data_spec, channel, scalertype, low_freq, up_freq, n_sections))
    
    return results #clusteringData
    


# # Experimenting: Plotting scaled FFT data

# In[1]:


def kmeans_clustering1(data_origin, data_spec, channel, scalertype, up_ylim):
    assert data_origin in ['04_2019', 'Hydrostatic_Flow'], "1st argument must be '04/2019' or 'Hydrostatic_Flow'"
    assert data_spec in ['', '1500rpm', '2000rpm', '2500rpm'], "2nd argument must be '', '1500rpm', '2000rpm' or '2500rpm'"
    assert channel in ['ch0', 'ch1'], "3rd argument must be 'ch0' or 'ch1'"
    assert scalertype in ['MinMax', 'Standard', 'Normalizer'], "4th argument must be 'MinMax', 'Standard' or 'Normalizer'"
    
    print("Performing KMeans clustering of {} {} {} FFT data using '{}Scaler' ...\n".format(data_origin, data_spec, channel, scalertype))
    
    metadata = prep_metadata()
    all_files = collect_all_fft_data(data_origin)
    
    if data_spec == '':
        in_files = all_files
    elif data_spec == '1500rpm':
        in_files = select_fft_data(metadata, all_files, 1500)    
    elif data_spec == '2000rpm':
        in_files = select_fft_data(metadata, all_files, 2000)
    elif data_spec == '2500rpm':
        in_files = select_fft_data(metadata, all_files, 2500)
        
    ref_file = pd.read_pickle(in_files[0])
    columns = ['file', 'Leak'] + [freq for freq in ref_file['frequency']]
    clusteringData = pd.DataFrame(columns = columns, index = [i for i in range(len(in_files))])
    
    for i in range(len(in_files)):
        fft_data = pd.read_pickle(in_files[i])
        clusteringData.iloc[i, :1] = os.path.basename(in_files[i]).split('.')[0]
        clusteringData.iloc[i, 1:2] = getLeakType(metadata, os.path.basename(in_files[i]).split('.')[0])
        clusteringData.iloc[i, 2:] = fft_data[channel].values
        
    labelEncoder = LabelEncoder()
    labelEncoder.fit(clusteringData['Leak'])
    clusteringData['Leak'] = labelEncoder.transform(clusteringData['Leak'])
    
    leak = np.array(clusteringData['Leak'])
    X = np.array(clusteringData.drop(['file', 'Leak'], axis=1).astype(float))
    
    kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
                    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
                    random_state=None, tol=0.0001, verbose=0)

    if scalertype == 'MinMax':
        scaler = MinMaxScaler()
    elif scalertype == 'Standard':
        scaler = StandardScaler()
    elif scalertype == 'Normalizer':
        scaler = Normalizer()
    
    
    
    X_scaled = scaler.fit_transform(X)
    kmeans.fit(X_scaled)
    
    clusterCenters = kmeans.cluster_centers_
    if scalertype == 'Normalizer':
        inv_clusterCenters = clusterCenters
    else:
        inv_clusterCenters = scaler.inverse_transform(clusterCenters)
    clusterLabels = kmeans.labels_
    inertia = kmeans.inertia_ # Sum of distances of samples to their closest cluster center

    score_eu = metrics.silhouette_score(X_scaled, clusterLabels, metric='euclidean')
    score_sqeu = metrics.silhouette_score(X_scaled, clusterLabels, metric='sqeuclidean')
    print("Euclidian silhouette score: {} \nSquared euclidian silhouette score: {} \nInertia: {}".format(score_eu, score_sqeu, inertia))
    print("Real classification: {} \nKMeans classification: {}\n".format(leak, clusterLabels))
    
    
    
    
    fig = plt.figure(figsize=(20,16))
    axes = fig.subplots(2, 2)
    x = clusteringData.columns[2:]
    xlim = 52500
    
    for i in range(2,3):
        y_unscaled = X[i,:]
        y_scaled = X_scaled[i,:]
        time = clusteringData.iloc[i][0].split('-')[3] + '-' + clusteringData.iloc[i][0].split('-')[4] + '-' + clusteringData.iloc[i][0].split('-')[5]
        leaktype = getLeakType(metadata, clusteringData.iloc[i][0])
        pump_speed = str(metadata[metadata["Parent Folder"] == clusteringData.iloc[i][0]]['Pump Speed (RPM)'].values[0]) + 'rpm'
    
        if leak[i] == 0:
            axes[0,0].plot(x[:xlim], y_unscaled[:xlim], label = time + '  ' + leaktype + '  ' + pump_speed)
            axes[1,0].plot(x[:xlim], y_scaled[:xlim], label = time + '  ' + leaktype + '  ' + pump_speed)
        elif leak[i] == 1:
            axes[0,1].plot(x[:xlim], y_unscaled[:xlim], label = time + '  ' + leaktype + '  ' + pump_speed)
            axes[1,1].plot(x[:xlim], y_scaled[:xlim], label = time + '  ' + leaktype + '  ' + pump_speed)
            
    
    axes[0,0].set_title('Unscaled: medium-leak (0)', fontsize=18)
    axes[0,0].legend()
    axes[0,0].set_xlabel('Frequency [Hz]', fontsize=14)
    axes[0,0].set_ylabel('Amplitude', fontsize=14)
    #axes[0,0].set_ylim(0, up_ylim)

    axes[0,1].set_title('Unscaled: no-leak (1)', fontsize=18)
    axes[0,1].legend()
    axes[0,1].set_xlabel('Frequency [Hz]', fontsize=14)
    #axes[0, 1].set_ylabel('Amplitude', fontsize=14)
    #axes[0,1].set_ylim(0, up_ylim) 
    
    axes[1,0].set_title('Scaled: medium-leak (0)', fontsize=18)
    axes[1,0].legend()
    axes[1,0].set_xlabel('Frequency [Hz]', fontsize=14)
    axes[1,0].set_ylabel('Amplitude', fontsize=14)
    #axes[1,0].set_ylim(0, up_ylim)

    axes[1,1].set_title('Scaled: no-leak (1)', fontsize=18)
    axes[1,1].legend()
    axes[1,1].set_xlabel('Frequency [Hz]', fontsize=14)
    #axes[0, 1].set_ylabel('Amplitude', fontsize=14)
    #axes[1,1].set_ylim(0, up_ylim)   
    
    fig.suptitle("KMeans clustering of  {}  {}  {}  FFT data (scaled using '{}Scaler')\n Unscaled vs. scaled data".format(data_origin, data_spec, channel, scalertype), fontsize=20, y=1.0)
    
    #return clusteringData



#kmeans_clustering1('Hydrostatic_Flow', '', 'ch1', 'Normalizer', 0.0025)    

