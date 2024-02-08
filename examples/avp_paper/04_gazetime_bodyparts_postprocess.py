#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Codes to generate the figures shown in the auism ET paper

04 - Re-structures the initial results of gaze onto body parts for plotting. 
Converts frame based counts of gaze to some area of interest to 
the percentage of total gaze time to these areas.  

"""

import os
import numpy as np
import pandas as pd
import pickle

# The Adolphs Lab data analysis scripts for eye tracking are provided 
# under alabeye package  
from alabeye.etdata import makedir
from alabeye.stats import cohen_d_ci, list_flatten


#%% Main directory for experiment data
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'

gaze_results_file = os.path.join(root_dir,'Results_v1','BodyPart_GazeTime','%s_fixtime_coeff_21.0.pkl')

# Directory to save outputs.
output_dir = os.path.join(root_dir,'Results_v1','Cohend_pool')

vidclips = [ ['Ep1_Clip1', 'Ep1_Clip2', 'Ep1_Clip3'], 'Ep4_AQNR', ['Ep1_Clip1', 'Ep1_Clip2', 'Ep1_Clip3', 'Ep4_AQNR'] ]
vidclips_txt = [ 'Ep1', 'Ep4_AQNR', 'AllVids' ]

# set up the output directory
makedir(output_dir)

#%%
for vid_ii,vid_txt in zip(vidclips,vidclips_txt):

    print(f'processing {vid_txt}...')
    
    if isinstance(vid_ii, list):
        
        # merge videos
        aio1_fix_time_pk_all = []
        aio2_fix_time_pk_all = []
        aio3_fix_time_pk_all = []
        aio4_fix_time_pk_all = []
        aio6_fix_time_pk_all = []
        subjs = None
        subj_groups = None        
        for vid_jj in vid_ii:
            
            this_result_file = gaze_results_file%vid_jj
        
            with open(this_result_file,'rb') as pf:
                fixation_times = pickle.load(pf)
            
            aio1_fix_time_pk = fixation_times['onscreen']
            aio2_fix_time_pk = fixation_times['heads']
            aio3_fix_time_pk = fixation_times['hands']
            aio4_fix_time_pk = fixation_times['nonheadbody']
            aio6_fix_time_pk = fixation_times['nonbody']
            
            if subjs is None:
                subjs = fixation_times['subj_info']
                subj_groups = fixation_times['subj_groups']
                if isinstance(subj_groups,list):
                    subj_groups = np.hstack(subj_groups)
            else:
                assert np.array_equal(subjs, fixation_times['subj_info'])
                if isinstance(fixation_times['subj_groups'],list):
                    assert np.array_equal(subj_groups, np.hstack(fixation_times['subj_groups']))
                else:    
                    assert np.array_equal(subj_groups, fixation_times['subj_groups'])
            
            aio1_fix_time_pk_all.append(aio1_fix_time_pk)
            aio2_fix_time_pk_all.append(aio2_fix_time_pk)
            aio3_fix_time_pk_all.append(aio3_fix_time_pk)
            aio4_fix_time_pk_all.append(aio4_fix_time_pk)
            aio6_fix_time_pk_all.append(aio6_fix_time_pk)
        
        aio1_fix_time_pk = np.hstack(aio1_fix_time_pk_all)
        aio2_fix_time_pk = np.hstack(aio2_fix_time_pk_all)
        aio3_fix_time_pk = np.hstack(aio3_fix_time_pk_all)
        aio4_fix_time_pk = np.hstack(aio4_fix_time_pk_all)
        aio6_fix_time_pk = np.hstack(aio6_fix_time_pk_all)
            
    elif isinstance(vid_ii, str):

        this_result_file = gaze_results_file%vid_ii

        with open(this_result_file,'rb') as pf:
            fixation_times = pickle.load(pf)
        
        aio1_fix_time_pk = fixation_times['onscreen']
        aio2_fix_time_pk = fixation_times['heads']
        aio3_fix_time_pk = fixation_times['hands']
        aio4_fix_time_pk = fixation_times['nonheadbody']
        # aio5_fix_time_pk = fixation_times['wbody']
        aio6_fix_time_pk = fixation_times['nonbody']
        
        subjs = np.array(fixation_times['subj_info'])
        subj_groups = fixation_times['subj_groups']
        if isinstance(subj_groups,list):
            subj_groups = np.hstack(subj_groups)


    # process data
    asd_subjs = subj_groups==1
    td_subjs = subj_groups==2
    asd_subjs_ids = subjs[asd_subjs]
    td_subjs_ids = subjs[td_subjs]
    print('# of ASD: %d, # of TD: %d\n'%(len(asd_subjs_ids),len(td_subjs_ids)))
    

    aio1_fix_time = []
    aio1_fix_time_keep = []
    for oii in aio1_fix_time_pk:
        aio1_fix_time.append(np.sum(oii)/len(oii))
        aio1_fix_time_keep.append(np.sum(oii))

    aio1_fix_time_keep = np.asarray(aio1_fix_time_keep)
    aio1_fix_time_keep[aio1_fix_time_keep==0] = 1 # against invalid division below. 

    aio1_fix_time = np.asarray(aio1_fix_time)
    aio2_fix_time = np.asarray([ np.sum(oii)/aio1_fix_time_keep[cii] for cii,oii in enumerate(aio2_fix_time_pk) ])
    aio3_fix_time = np.asarray([ np.sum(oii)/aio1_fix_time_keep[cii] for cii,oii in enumerate(aio3_fix_time_pk) ])
    aio4_fix_time = np.asarray([ np.sum(oii)/aio1_fix_time_keep[cii] for cii,oii in enumerate(aio4_fix_time_pk) ])
    aio6_fix_time = np.asarray([ np.sum(oii)/aio1_fix_time_keep[cii] for cii,oii in enumerate(aio6_fix_time_pk) ])
    
    aio1_fix_time_asd = aio1_fix_time[asd_subjs]
    aio2_fix_time_asd = aio2_fix_time[asd_subjs]
    aio3_fix_time_asd = aio3_fix_time[asd_subjs]
    aio4_fix_time_asd = aio4_fix_time[asd_subjs]
    aio6_fix_time_asd = aio6_fix_time[asd_subjs]
    
    aio1_fix_time_td = aio1_fix_time[td_subjs]
    aio2_fix_time_td = aio2_fix_time[td_subjs]
    aio3_fix_time_td = aio3_fix_time[td_subjs]
    aio4_fix_time_td = aio4_fix_time[td_subjs]
    aio6_fix_time_td = aio6_fix_time[td_subjs]
    

    np.save(os.path.join(output_dir,f'BodyParts_asd_subjs_{vid_txt}'), asd_subjs_ids)
    np.save(os.path.join(output_dir,f'BodyParts_td_subjs_{vid_txt}'), td_subjs_ids)
    
    corr_pairs = [('On-screen', (aio1_fix_time_asd,aio1_fix_time_td)),
                  ('Head', (aio2_fix_time_asd,aio2_fix_time_td)),
                  ('Hands', (aio3_fix_time_asd,aio3_fix_time_td)),
                  ('Non-head body', (aio4_fix_time_asd,aio4_fix_time_td)),
                  # ('Social content', (aio5_fix_time_asd,aio5_fix_time_td)) ]
                  ('Non-social content', (aio6_fix_time_asd,aio6_fix_time_td)) ]
    
    keep_dvals = []
    for pii, (aoi, (var_asd,var_td)) in enumerate(corr_pairs):
        
        d_vals = cohen_d_ci(var_td,var_asd,rm_extreme=False)
        keep_dvals.append(list_flatten(d_vals))

    keep_dvals = np.vstack(keep_dvals)

    dp_rows = ['On-screen', 'Head', 'Hands', 'Non-head body', 'Non-social content' ]
    dp_cols = ['d-direct', 'd-bootstrap-mean', 'd CI lower', 'd CI upper', 'd-pval' ]
    dp_vals_pd = pd.DataFrame(data=keep_dvals, index=dp_rows, columns=dp_cols)
    
    dp_vals_pd.to_csv(os.path.join(output_dir,f'BodyPartGaze_dvals_{vid_txt}.csv'))
    
    
    np.save(os.path.join(output_dir,f'OnscreenData_asd_{vid_txt}'), aio1_fix_time_asd)
    np.save(os.path.join(output_dir,f'OnscreenData_td_{vid_txt}'), aio1_fix_time_td)
    
    np.save(os.path.join(output_dir,f'HeadData_asd_{vid_txt}'), aio2_fix_time_asd)
    np.save(os.path.join(output_dir,f'HeadData_td_{vid_txt}'), aio2_fix_time_td)
    
    np.save(os.path.join(output_dir,f'HandsData_asd_{vid_txt}'), aio3_fix_time_asd)
    np.save(os.path.join(output_dir,f'HandsData_td_{vid_txt}'), aio3_fix_time_td)
    
    np.save(os.path.join(output_dir,f'NonheadData_asd_{vid_txt}'), aio4_fix_time_asd)
    np.save(os.path.join(output_dir,f'NonheadData_td_{vid_txt}'), aio4_fix_time_td)
    
    np.save(os.path.join(output_dir,f'NonsocialData_asd_{vid_txt}'), aio6_fix_time_asd)
    np.save(os.path.join(output_dir,f'NonsocialData_td_{vid_txt}'), aio6_fix_time_td)
    
    
#%%

    np.save(os.path.join(output_dir,f'Onscreen_asd_split_{vid_txt}'), aio1_fix_time_pk[asd_subjs])
    np.save(os.path.join(output_dir,f'Onscreen_td_split_{vid_txt}'), aio1_fix_time_pk[td_subjs])
    
    np.save(os.path.join(output_dir,f'Head_asd_split_{vid_txt}'), aio2_fix_time_pk[asd_subjs])
    np.save(os.path.join(output_dir,f'Head_td_split_{vid_txt}'), aio2_fix_time_pk[td_subjs])
    
    np.save(os.path.join(output_dir,f'Hands_asd_split_{vid_txt}'), aio3_fix_time_pk[asd_subjs])
    np.save(os.path.join(output_dir,f'Hands_td_split_{vid_txt}'), aio3_fix_time_pk[td_subjs])
        
    np.save(os.path.join(output_dir,f'Nonhead_asd_split_{vid_txt}'), aio4_fix_time_pk[asd_subjs])
    np.save(os.path.join(output_dir,f'Nonhead_td_split_{vid_txt}'), aio4_fix_time_pk[td_subjs])
    
    np.save(os.path.join(output_dir,f'Nonsocial_asd_split_{vid_txt}'), aio6_fix_time_pk[asd_subjs])
    np.save(os.path.join(output_dir,f'Nonsocial_td_split_{vid_txt}'), aio6_fix_time_pk[td_subjs])
    
    
