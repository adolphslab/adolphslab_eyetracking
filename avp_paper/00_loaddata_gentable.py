#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Codes to generate the figures shown in the auism ET paper

0- Load the original data and downsample to the frame rate of video stimulus 
and generate Table 1 for participants' demographic information.

"""

import os
import numpy as np
import pandas as pd
from functools import reduce

# The Adolphs Lab data analysis scripts for eye tracking are provided 
# under alabeye package  
from alabeye.etdata import ETdata

from scipy.stats import chi2_contingency, ttest_ind


#%% Main directory for experiment data
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'

# The main directory for experimental data contains several subdirectories such as:
# - ETdata: eye tracking data
# - BehavioralData: demographic and psychological assessment info about subjects
# - StimVids: media files for experimental stimuli [not shared here because of copyright restrictions] 
# - FeatureZoo: various features extracted from video stimuli 

subj_info_file = os.path.join(root_dir,'BehavioralData','participants_info.csv')
stim_dir = os.path.join(root_dir,'StimVids')

#%% Load and preprocess the ET data collected with the Office - Episode 1 (called Episode A in the paper)

# HDF file of gaze data for Episode A (see ../examples/preprocess_rawdata/avp_officevids about how to prepare this file)
hdf_fname_epA = 'ET_Tobii_Ep1_v0.hdf5'
hdf_file_epA = os.path.join(root_dir,'ETdata',hdf_fname_epA)

# initialize an ETdata object to load some basic info about the recorded ET data.
etdata_init = ETdata(data_file=hdf_file_epA)
epA_tasks = etdata_init.available_tasks

# find subjects who viewed all three parts of Episode A.
subjs_epAclips = []
for task_ii in epA_tasks:
    etdata_tii = ETdata(task_ii, data_file=hdf_file_epA, subj_info_file=subj_info_file,
                       use_subj_info=True, stim_dir=stim_dir)
    subjs_epAclips.append(etdata_tii.subjs)

epA_subjs = reduce(np.intersect1d, subjs_epAclips).tolist()

#%% Load and preprocess the ET data collected with the Office - Episode 4 (called Episode B in the paper)

# HDF file of gaze data for Episode B (see ../examples/preprocess_rawdata/avp_officevids about how to prepare this file)
hdf_fname_epB = 'ET_Tobii_Ep4_AQNR_v0.hdf5' 
hdf_file_epB = os.path.join(root_dir,'ETdata',hdf_fname_epB)

epB_etdata = ETdata(data_file=hdf_file_epB)
epB_tasks = epB_etdata.available_tasks
assert len(epB_tasks)==1

epB_etdata = ETdata(epB_tasks[0], data_file=hdf_file_epB, subj_info_file=subj_info_file,
                    use_subj_info=True, stim_dir=stim_dir)
epB_subjs = epB_etdata.subjs


#%%
# read group info from subj_info_file. 
subj_info_df = pd.read_csv(subj_info_file)

print('--- Info about the participants [initial numbers] ---')
epA_subj_info_df = subj_info_df.loc[subj_info_df['ID'].isin(epA_subjs)]
epA_subj_groups = epA_subj_info_df.Group.to_numpy(dtype=int)
print('Episode A:',
      f'ASD={sum(epA_subj_groups==1)}, TD={sum(epA_subj_groups==2)}, Total={len(epA_subj_groups)}')

epB_subj_info_df = subj_info_df.loc[subj_info_df['ID'].isin(epB_subjs)]
epB_subj_groups = epB_subj_info_df.Group.to_numpy(dtype=int)
print('Episode B:', 
      f'ASD={sum(epB_subj_groups==1)}, TD={sum(epB_subj_groups==2)}, Total={len(epB_subj_groups)}')

epAB_subjs = reduce(np.intersect1d, [epA_subjs,epB_subjs]).tolist()
epAB_subj_info_df = subj_info_df.loc[subj_info_df['ID'].isin(epAB_subjs)]
epAB_subj_groups = epAB_subj_info_df.Group.to_numpy(dtype=int)
print('Both episodes:', 
      f'ASD={sum(epAB_subj_groups==1)}, TD={sum(epAB_subj_groups==2)}, Total={len(epAB_subj_groups)}')

#%% Downsample gaze data to video frame rates (i.e., one average gaze point per frame)
# this step allows us to remove subjects with more than half missing data points

dsample_subjs_eps = []
for (tasks,hdf_file) in [ (epA_tasks,hdf_file_epA), (epB_tasks,hdf_file_epB) ]:
    
    for task_ii in tasks:
        print(f'\n\t Processing {task_ii}...')
        etdata_tii = ETdata(task_ii, data_file=hdf_file, subj_info_file=subj_info_file,
                           use_subj_info=True, stim_dir=stim_dir)
        etdata_tii.get_timebinned_data(load_these_subjs=epAB_subjs, 
                                       rm_subj_withhalfmissingdata=True,
                                       bin_operation='mean')
        dsample_subjs_eps.append( etdata_tii.data_subjs[0] )

dsample_subjs = reduce(np.intersect1d, dsample_subjs_eps).tolist()

# remove 2 outlier subjects:
rm_subjs_list = [ 'A00651', 'C00866' ]

for sii in rm_subjs_list:
    if sii in dsample_subjs:
        dsample_subjs.remove(sii)

epAB_final_subj_info_df = subj_info_df.loc[subj_info_df['ID'].isin(dsample_subjs)]
epAB_final_subj_groups = epAB_final_subj_info_df.Group.to_numpy(dtype=int)
print('\nBoth episodes:', 
      f'ASD={sum(epAB_final_subj_groups==1)}, TD={sum(epAB_final_subj_groups==2)}, Total={len(epAB_final_subj_groups)}')

#%% Re-visit data to downsample and save only for these final subjects

output_dir = os.path.join(root_dir,'ETdata','down2frame_data')

for (tasks,hdf_file) in [ (epA_tasks,hdf_file_epA), (epB_tasks,hdf_file_epB) ]:
    for task_ii in tasks:
        print(f'\n\t Processing {task_ii}...')
        etdata_tii = ETdata(task_ii, data_file=hdf_file, subj_info_file=subj_info_file,
                           use_subj_info=True, stim_dir=stim_dir)
        etdata_tii.get_timebinned_data(load_these_subjs=dsample_subjs, 
                                       rm_subj_withhalfmissingdata=True,
                                       bin_operation='mean', fix_length=True,
                                       split_groups=True, save_output=True, 
                                       output_overwrite=False, output_dir=output_dir)

#%% Some stat test on epAB_final_subj_info_df

epAB_final_subj_info_df.reset_index(drop=True,inplace=True)
# epAB_final_subj_info_df.to_csv('subj_data.csv',index=False)

asd_subjs_df = epAB_final_subj_info_df.loc[epAB_final_subj_info_df.Group==1]
td_subjs_df = epAB_final_subj_info_df.loc[epAB_final_subj_info_df.Group==2]

print('\nfinal data- female ASD:', np.sum(asd_subjs_df['Gender'].values==2) )
print('final data- female TD:', np.sum(td_subjs_df['Gender'].values==2) )

asd_sexratio = np.sum(asd_subjs_df['Gender'].values==1) / len(asd_subjs_df['Gender'].values)
td_sexratio = np.sum(td_subjs_df['Gender'].values==1) / len(td_subjs_df['Gender'].values)

obs_sex = np.array([[np.sum(asd_subjs_df['Gender'].values==1), np.sum(td_subjs_df['Gender'].values==1) ],
                    [np.sum(asd_subjs_df['Gender'].values==2), np.sum(td_subjs_df['Gender'].values==2) ]])


print(f'\nFraction male: TD: {td_sexratio:0.3f}, ASD: {asd_sexratio:0.3f}')

print('\nchi2:',chi2_contingency(obs_sex,correction=True)[0:2])
print('---------- o ----------\n')

print('Age ASD:', np.nanmean(asd_subjs_df['Age'].values), np.nanstd(asd_subjs_df['Age'].values) )
print('Age ASD (min,max):', np.nanmin(asd_subjs_df['Age'].values), np.nanmax(asd_subjs_df['Age'].values) )
print()
print('Age TD:', np.nanmean(td_subjs_df['Age'].values), np.nanstd(td_subjs_df['Age'].values) )
print('Age TD (min,max):', np.nanmin(td_subjs_df['Age'].values), np.nanmax(td_subjs_df['Age'].values) )
print('t-test:', ttest_ind(td_subjs_df['Age'].values,asd_subjs_df['Age'].values)[:])
print('---------- o ----------\n')


#%%
print('FS IO - ASD:', np.nanmean(asd_subjs_df['FSIQ'].values), np.nanstd(asd_subjs_df['FSIQ'].values) )
print('FS IO - ASD (min,max):', np.nanmin(asd_subjs_df['FSIQ'].values), np.nanmax(asd_subjs_df['FSIQ'].values) )
print()
print('FS IO - TD:', np.nanmean(td_subjs_df['FSIQ'].values), np.nanstd(td_subjs_df['FSIQ'].values) )
print('FS IO - TD (min,max):', np.nanmin(td_subjs_df['FSIQ'].values), np.nanmax(td_subjs_df['FSIQ'].values) )

# ----- stat tests -----  
val_td = td_subjs_df['FSIQ'].values
val_asd = asd_subjs_df['FSIQ'].values

print('t-test', ttest_ind(val_td,val_asd,nan_policy='omit')[:], sum(~np.isnan(val_td)), sum(~np.isnan(val_asd)), 
      sum(~np.isnan(val_td))+sum(~np.isnan(val_asd)), len(epAB_final_subj_info_df) )
print('---------- o ----------\n')

#%%
comb_scores = asd_subjs_df['CSS'].values
print('--- CSS value ---')
print( comb_scores.min(), comb_scores.max(), comb_scores.mean(), comb_scores.std() )
