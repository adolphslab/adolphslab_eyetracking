#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Codes to generate the figures shown in the auism ET paper

06 - Re-structures the initial results of heatmap correlations for plotting. 

"""

import os
import numpy as np
import pandas as pd

# The Adolphs Lab data analysis scripts for eye tracking are provided 
# under alabeye package  
from alabeye.etdata import makedir
from alabeye.stats import cohen_d_ci

#%% Main directory for experiment data
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'

hcorrs_results_dir = os.path.join(root_dir,'Results_v1','CorrsNSS','CombVals_tbin_1n0')

# --- need to be run once with setting 'td' and once with 'asd' ---
ref_group = 'td' # 'td' or 'asd'

# Directory to save outputs.
output_dir = os.path.join(root_dir,'Results_v1','Cohend_pool')

vidclips = [ ['Ep1_Clip1', 'Ep1_Clip2', 'Ep1_Clip3'], 'Ep4_AQNR', ['Ep1_Clip1', 'Ep1_Clip2', 'Ep1_Clip3', 'Ep4_AQNR'] ]
vidclips_txt = [ 'Ep1', 'Ep4_AQNR', 'AllVids' ]

# set up the output directory
makedir(output_dir,sysexit=False)

#%%
for vid_ii,vid_txt in zip(vidclips,vidclips_txt):

    print(f'processing {vid_txt}...')
    
    if ref_group == 'td':
        ref_txt = ''
    elif ref_group=='asd':
        ref_txt = '_asdref'
    
    if isinstance(vid_ii, list):

        subjs = None
        asd_corrs = []
        td_corrs = []
        frames_split_list = []
        for vid_jj in vid_ii:
            
            asd_subjs_ids = np.load(os.path.join(hcorrs_results_dir,f'subjs_asd_{vid_jj}{ref_txt}.npy'))
            td_subjs_ids = np.load(os.path.join(hcorrs_results_dir,f'subjs_td_{vid_jj}{ref_txt}.npy'))
            
            if subjs is None:
                subjs = np.r_[asd_subjs_ids,td_subjs_ids]
            else:
                assert np.array_equal(subjs, np.r_[asd_subjs_ids,td_subjs_ids])

            asd_corrs.append( np.load(os.path.join(hcorrs_results_dir,f'corrs_asd_{vid_jj}{ref_txt}.npy')).T )
            td_corrs.append( np.load(os.path.join(hcorrs_results_dir,f'corrs_td_{vid_jj}{ref_txt}.npy')).T )
            frames_split_list.append( np.load(os.path.join(hcorrs_results_dir,f'scene_frames_splits_list_{vid_jj}{ref_txt}.npy')) )
            

        asd_corrs = np.hstack(asd_corrs)
        td_corrs = np.hstack(td_corrs)

        # recount frames across the parts of episode 1        
        for lii in range(len(frames_split_list)-1):
            frames_split_list[lii+1] = frames_split_list[lii+1] + frames_split_list[lii][-1,1]
        frames_split_list = np.vstack(frames_split_list)

    elif isinstance(vid_ii, str):

        asd_subjs_ids = np.load(os.path.join(hcorrs_results_dir,f'subjs_asd_{vid_ii}{ref_txt}.npy'))
        td_subjs_ids = np.load(os.path.join(hcorrs_results_dir,f'subjs_td_{vid_ii}{ref_txt}.npy'))
        subjs = np.r_[asd_subjs_ids,td_subjs_ids]
            
        asd_corrs = np.load(os.path.join(hcorrs_results_dir,f'corrs_asd_{vid_ii}{ref_txt}.npy')).T 
        td_corrs = np.load(os.path.join(hcorrs_results_dir,f'corrs_td_{vid_ii}{ref_txt}.npy')).T 
        frames_split_list = np.load(os.path.join(hcorrs_results_dir,f'scene_frames_splits_list_{vid_ii}{ref_txt}.npy')) 


    # ----- process data -----
    print('# of ASD: %d, # of TD: %d\n'%(len(asd_subjs_ids),len(td_subjs_ids)))

    asd_corrs_mean = np.nanmean(asd_corrs, 1)
    td_corrs_mean = np.nanmean(td_corrs, 1)
    
    dvals = cohen_d_ci(td_corrs_mean, asd_corrs_mean, rm_extreme=False)
    dval_cols = [ 'd-direct', 'd-bootstrap-mean', 'd CI lower', 'd CI upper', 'd-pval' ]
    
    dvals_df = pd.DataFrame(data=np.asarray(dvals).reshape(1,-1), index=[f'GazeCorrs{ref_txt}'], columns=dval_cols)
    dvals_df.to_csv(os.path.join(output_dir,f'GazeCorr_dvals_{vid_txt}{ref_txt}.csv'))
    
    np.save(os.path.join(output_dir,f'GazeCorr_asd_subjs_{vid_txt}{ref_txt}'), asd_subjs_ids)
    np.save(os.path.join(output_dir,f'GazeCorr_td_subjs_{vid_txt}{ref_txt}'), td_subjs_ids)
    
    np.save(os.path.join(output_dir,f'GazeCorr_asd_{vid_txt}{ref_txt}'), asd_corrs_mean)
    np.save(os.path.join(output_dir,f'GazeCorr_td_{vid_txt}{ref_txt}'), td_corrs_mean)
    

#%%

    np.save(os.path.join(output_dir,f'GazeCorr_asd_split_{vid_txt}{ref_txt}'), asd_corrs)
    np.save(os.path.join(output_dir,f'GazeCorr_td_split_{vid_txt}{ref_txt}'), td_corrs)
    np.save(os.path.join(output_dir,f'frames_split_list_{vid_txt}{ref_txt}'), frames_split_list)
    
    