#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Codes to generate the figures shown in the auism ET paper

3- Compute heatmap correlations between individual subject-wise heatmaps and a group heatmap.


"""

import os
import sys
import numpy as np
import pickle
from tqdm import tqdm

from sklearn.model_selection import LeaveOneOut
from scipy.spatial.distance import squareform

# The Adolphs Lab data analysis scripts for eye tracking are provided 
# under alabeye package  
from alabeye.etdata import ETdata, makedir
from alabeye.etutils import compute_ioc, et_heatmap, compute_nss, norm_zs
from alabeye.stats import cross_correlation

#%% Main directory for experiment data
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'

# The main directory for experimental data contains several subdirectories such as:
# - ETdata: eye tracking data
# - ETdata/down2frame_data: gaze data downsampled to the frame rate of video stimulus
# - BehavioralData: demographic and psychological assessment info about subjects
# - StimVids: media files for experimental stimuli [not shared here because of copyright restrictions] 
# - FeatureZoo: various features extracted from video stimuli 

stim_dir = os.path.join(root_dir,'StimVids')
features_dir = os.path.join(root_dir,'FeatureZoo')
prepdata_dir = os.path.join(root_dir,'ETdata','down2frame_data')

# Directory to save outputs.
output_dir = os.path.join(root_dir,'Results_v1','CorrsNSS','CombVals_tbin_1n0')


#%% Other settings

perc_threshold = 75 # 75 to keep top 25 in IOC metric.

split_duration = 1.0 # sec. TRs: 0.72
sigma = 21.0 # Standard deviation for Gaussian kernel. Equal for both axes.
hp_down_factor = float(5.0) # should be float.

# Videos to compute gaze times.
vidclips = ['Ep1_Clip1', 'Ep1_Clip2', 'Ep1_Clip3', 'Ep4_AQNR']

# --- need to be run once with setting 'td' and once with 'asd' ---
ref_group = 'td' # 'td' or 'asd'

# set up the output directory
makedir(output_dir,sysexit=False)

#%%
for vii, vid_ii in enumerate(vidclips):
    
    print(f'Processing video file: {vid_ii}...')

    # load downsampled gaze data, which were prepared in 00_loaddata.py
    data_file = os.path.join(prepdata_dir,f'timebinned_data_{vid_ii}.pkl')
    vid_etdata = ETdata(data_file=data_file,stim_dir=stim_dir)
    ngroups = vid_etdata.data_ngroups

    # ----- Load some information about the video -----
    nframes = vid_etdata.stim_mediainfo['nframes']
    frame_width = vid_etdata.stim_mediainfo['frame_width']
    frame_height = vid_etdata.stim_mediainfo['frame_height']
    vid_fps = vid_etdata.stim_mediainfo['fps']
    vid_duration = vid_etdata.stim_mediainfo['duration']

    frame_duration = 1000./vid_fps # msec. 
    framesize = [frame_width,frame_height]

    asd_subjs = vid_etdata.data_subjs[0]
    td_subjs = vid_etdata.data_subjs[1]
    
    if ref_group == 'td':
        comp_idx, ref_idx = 0, 1 # asd, td
        comp_group = 'asd'
    elif ref_group == 'asd':
        comp_idx, ref_idx = 1, 0 # td, asd
        comp_group = 'td'

    comp_subjs = vid_etdata.data_subjs[comp_idx]
    et_xy_comp = vid_etdata.data[comp_idx]

    ref_subjs = vid_etdata.data_subjs[ref_idx]
    et_xy_ref = vid_etdata.data[ref_idx]

    
    n_splits = vid_duration // split_duration 
    if n_splits == 0: n_splits = 1
    
    scene_frames = np.arange(nframes).astype(int)
    scene_frames_splits = np.array_split(scene_frames,n_splits)
    scene_frames_splits_list = [ [ sp_ii[0], sp_ii[-1]+1 ] for sp_cnt, sp_ii in enumerate(scene_frames_splits) ]
    timebins_msec = [ sp_ii[1]*frame_duration for sp_ii in scene_frames_splits_list ] # sp_ii[1] is already depythonized count due to +1 above. 
    

    # --- Generate heatmaps ---
    keep_corrs_comp = []
    keep_corrs_ref = []

    keep_nss_comp = []
    keep_nss_ref = []

    keep_ioc_comp = []
    keep_ioc_ref = []
    
    keep_all_xcorrs = []

    for sc_cnt, split_ii in enumerate(tqdm(scene_frames_splits_list,total=len(scene_frames_splits_list))):
        
        frames_use = np.zeros(nframes,dtype=bool)
        frames_use[ split_ii[0]:split_ii[1] ] = True
    
        # get ET data for this time bin. 
        et_bin_comp = [ sub_et_ii[frames_use] for sub_et_ii in et_xy_comp ]
        et_bin_ref = [ sub_et_ii[frames_use] for sub_et_ii in et_xy_ref ]

        # --- Gaussian Heatmap ---
        heatmap_down_comp = np.vstack( [ et_heatmap(et_bin_ii,framesize,sigma,hp_down_factor,get_full=False,nan_ratio=0.5).ravel() for et_bin_ii in et_bin_comp ] )
        heatmap_down_ref = np.vstack( [ et_heatmap(et_bin_ii,framesize,sigma,hp_down_factor,get_full=False,nan_ratio=0.5).ravel() for et_bin_ii in et_bin_ref ] )            

        # all cross-correlations. 
        all_xcorrs = np.corrcoef(np.vstack((heatmap_down_comp,heatmap_down_ref)) )
        all_xcorrs_tri = squareform(all_xcorrs,checks=False)

        # --- Reference aggregate heatmap ---
        et_bin_agg = np.vstack(et_bin_ref)
        heatmap_agg_down = et_heatmap(et_bin_agg,framesize,sigma,hp_down_factor,get_full=False)
        comp_corrs = cross_correlation(heatmap_down_comp,heatmap_agg_down.reshape(1,-1)).squeeze()

        
        thrs_val = np.percentile(heatmap_agg_down.ravel(),perc_threshold)
        heatmap_agg_down_binary = np.zeros(heatmap_agg_down.shape,dtype=bool)
        heatmap_agg_down_binary[heatmap_agg_down>thrs_val] = True 
        comp_ioc = [ compute_ioc(heatmap_agg_down_binary,xy_ii/hp_down_factor) for xy_ii in et_bin_comp ]
                
        heatmap_agg_down_norm = norm_zs(heatmap_agg_down) # normalize once for speed. 
        comp_nss = [ compute_nss(heatmap_agg_down_norm,xy_ii/hp_down_factor) for xy_ii in et_bin_comp ]

        
        loo = LeaveOneOut()
        loo_list = list(loo.split(np.arange(len(et_bin_ref))))
        ref_corrs_loo = []
        ref_ioc_loo = []
        ref_nss_loo = []
        for train_ii, test_ii in loo_list:
            train_mat = np.vstack([ et_bin_ref[tr_ii] for tr_ii in train_ii ])
            train_heatmap = et_heatmap(train_mat,framesize,sigma,hp_down_factor,get_full=False)
            
            test_heatmap = heatmap_down_ref[test_ii[0]]
            if np.sum(test_heatmap)>0:
                ref_corrs_loo.append( np.corrcoef(test_heatmap.ravel(),train_heatmap.ravel())[0,1] )
            else:
                ref_corrs_loo.append( np.NaN )
                
            thrs_val_tr = np.percentile(train_heatmap.ravel(),perc_threshold)
            train_heatmap_binary = np.zeros(train_heatmap.shape,dtype=bool)
            train_heatmap_binary[train_heatmap>thrs_val_tr] = True 
            
            ref_ioc_loo.append( compute_ioc(train_heatmap_binary, et_bin_ref[test_ii[0]]/hp_down_factor ) )
            ref_nss_loo.append( compute_nss(norm_zs(train_heatmap), et_bin_ref[test_ii[0]]/hp_down_factor ) )


        # keep data to save at the end. 
        keep_corrs_comp.append(comp_corrs)
        keep_ioc_comp.append(comp_ioc)
        keep_nss_comp.append(comp_nss)
        
        keep_corrs_ref.append(ref_corrs_loo)
        keep_ioc_ref.append(ref_ioc_loo)
        keep_nss_ref.append(ref_nss_loo)
        
        keep_all_xcorrs.append(all_xcorrs_tri)
        
        
#%%
    # save video outputs
    if ref_group=='asd':
        vid_ii += '_asdref'

    np.save(os.path.join(output_dir,f'subjs_{comp_group}_{vid_ii}'), comp_subjs)    
    np.save(os.path.join(output_dir,f'subjs_{ref_group}_{vid_ii}'), ref_subjs)

    np.save(os.path.join(output_dir,f'corrs_{comp_group}_{vid_ii}'), keep_corrs_comp)
    np.save(os.path.join(output_dir,f'corrs_{ref_group}_{vid_ii}'), keep_corrs_ref)

    np.save(os.path.join(output_dir,f'ioc_{comp_group}_{vid_ii}'), keep_ioc_comp)
    np.save(os.path.join(output_dir,f'ioc_{ref_group}_{vid_ii}'), keep_ioc_ref)

    np.save(os.path.join(output_dir,f'nss_{comp_group}_{vid_ii}'), keep_nss_comp)
    np.save(os.path.join(output_dir,f'nss_{ref_group}_{vid_ii}'), keep_nss_ref)
    
    np.save(os.path.join(output_dir,f'xcorrs_{comp_group}{ref_group}_{vid_ii}'), keep_all_xcorrs)

    np.save(os.path.join(output_dir,f'scene_frames_splits_{vid_ii}'), np.asarray(scene_frames_splits,dtype=object) )
    np.save(os.path.join(output_dir,f'scene_frames_splits_list_{vid_ii}'), scene_frames_splits_list)
    np.save(os.path.join(output_dir,f'timebins_msec_{vid_ii}'), timebins_msec)

