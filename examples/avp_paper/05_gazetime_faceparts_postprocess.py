#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Codes to generate the figures shown in the auism ET paper

04 - Re-structures the initial results of gaze onto face parts for plotting. 
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
from alabeye.etutils import get_cohend_vals, get_subj_vals


#%% Main directory for experiment data
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'

gaze_results_file = os.path.join(root_dir,'Results_v1','Face_GazeTime','%s_fixtime_coeff_21.0.pkl')

# Directory to save outputs.
output_dir = os.path.join(root_dir,'Results_v1','Cohend_pool')

vidclips = [['Ep1_Clip1', 'Ep1_Clip2', 'Ep1_Clip3'], 'Ep4_AQNR', ['Ep1_Clip1', 'Ep1_Clip2', 'Ep1_Clip3', 'Ep4_AQNR'] ]
vidclips_txt = [ 'Ep1', 'Ep4_AQNR', 'AllVids' ]

scale2facetime = False

# set up the output directory
makedir(output_dir)

#%%
for vid_ii,vid_txt in zip(vidclips,vidclips_txt):

    print(f'processing {vid_txt}...')
    
    if isinstance(vid_ii, list):

        
        face_use_info = []
        
        face_gaze = []
        eye_gaze = []
        mouth_gaze = []
        face_gaze_dists = []
        face_gaze_dists_valid = []
        face_num = []

        subjs = None
        subj_groups = None        
        for vid_jj in vid_ii:
            this_result_file = gaze_results_file%vid_jj

            with open(this_result_file,'rb') as pf:
                fixation_times = pickle.load(pf)

            if subjs is None:
                subjs = fixation_times['subjs']
                subj_groups = fixation_times['subj_groups']
                if isinstance(subj_groups,list):
                    subj_groups = np.hstack(subj_groups)
            else:
                assert np.array_equal(subjs, fixation_times['subjs'])
                if isinstance(fixation_times['subj_groups'],list):
                    assert np.array_equal(subj_groups, np.hstack(fixation_times['subj_groups']))
                else:    
                    assert np.array_equal(subj_groups, fixation_times['subj_groups'])
            
        
            face_use_info += fixation_times['face_use_info'] # instead of appending and hstack object type.
            face_gaze.append(fixation_times['face_gaze'])
            eye_gaze.append(fixation_times['eye_gaze'])
            mouth_gaze.append(fixation_times['mouth_gaze'])
            face_gaze_dists.append(fixation_times['face_gaze_dists'])
            face_gaze_dists_valid.append(fixation_times['face_gaze_dists_valid'])
            face_num.append(fixation_times['face_num'])

        face_gaze = np.hstack(face_gaze)
        eye_gaze = np.hstack(eye_gaze)
        mouth_gaze = np.hstack(mouth_gaze)
        face_gaze_dists = np.hstack(face_gaze_dists)
        face_gaze_dists_valid = np.hstack(face_gaze_dists_valid)
        face_num = np.hstack(face_num)
        
        
    elif isinstance(vid_ii, str):

        this_result_file = gaze_results_file%vid_ii

        with open(this_result_file,'rb') as pf:
            fixation_times = pickle.load(pf)

        subjs = fixation_times['subjs']
        subj_groups = fixation_times['subj_groups']
        if isinstance(subj_groups,list):
            subj_groups = np.hstack(subj_groups)

        face_use_info = fixation_times['face_use_info']
        face_gaze = fixation_times['face_gaze']
        eye_gaze = fixation_times['eye_gaze']
        mouth_gaze = fixation_times['mouth_gaze']
        face_gaze_dists = fixation_times['face_gaze_dists']
        face_gaze_dists_valid = fixation_times['face_gaze_dists_valid']
        face_num = fixation_times['face_num']


    # ----- process data -----
    asd_subjs = subj_groups==1
    td_subjs = subj_groups==2
    asd_subjs_ids = subjs[asd_subjs]
    td_subjs_ids = subjs[td_subjs]
    print('# of ASD: %d, # of TD: %d\n'%(len(asd_subjs_ids),len(td_subjs_ids)))

    
    # -1 not onscreen; 0 onscreen nonface; 1 onscreen face. 
    onscreen_fix_sum = np.sum(face_gaze != -1, 1)
    onscreen_fix = onscreen_fix_sum / face_gaze.shape[1]
    
    d_os = cohen_d_ci(onscreen_fix[td_subjs], onscreen_fix[asd_subjs], rm_extreme=False)
    
    d_f, d_ef, d_mf = get_cohend_vals(face_gaze[td_subjs],face_gaze[asd_subjs],
                                      eye_gaze[td_subjs],eye_gaze[asd_subjs],
                                      mouth_gaze[td_subjs],mouth_gaze[asd_subjs],
                                      scale2facetime=scale2facetime)
    
    face_data_asd, eye_data_asd, mouth_data_asd = get_subj_vals(face_gaze[asd_subjs],\
                            eye_gaze[asd_subjs], mouth_gaze[asd_subjs], scale2facetime=scale2facetime)
    
    face_data_td, eye_data_td, mouth_data_td = get_subj_vals(face_gaze[td_subjs],\
                           eye_gaze[td_subjs], mouth_gaze[td_subjs], scale2facetime=scale2facetime)
    

    keep_dvals = np.vstack( [ list_flatten(d_os), list_flatten(d_f), 
                              list_flatten(d_ef), list_flatten(d_mf) ] )
        
    rows = ['On-screen', 'Face', 'Eyes', 'Mouth' ]
    cols = ['d-direct', 'd-bootstrap-mean', 'd CI lower', 'd CI upper', 'd-pval' ]
    
    dvals_pd = pd.DataFrame(data=keep_dvals, index=rows, columns=cols)
    
    if scale2facetime:
        vid_txt += '_scale2facetime'
    
    dvals_pd.to_csv(os.path.join(output_dir,f'FaceGaze_dvals_{vid_txt}.csv'))
    
    np.save(os.path.join(output_dir,f'FaceData_asd_subjs_{vid_txt}'), asd_subjs_ids)
    np.save(os.path.join(output_dir,f'FaceData_td_subjs_{vid_txt}'), td_subjs_ids)
    
    np.save(os.path.join(output_dir,f'OnscreenDataFace_asd_{vid_txt}'), onscreen_fix[asd_subjs])
    np.save(os.path.join(output_dir,f'OnscreenDataFace_td_{vid_txt}'), onscreen_fix[td_subjs])
    
    np.save(os.path.join(output_dir,f'FaceData_asd_{vid_txt}'), face_data_asd)
    np.save(os.path.join(output_dir,f'FaceData_td_{vid_txt}'), face_data_td)
    
    np.save(os.path.join(output_dir,f'EyeData_asd_{vid_txt}'), eye_data_asd)
    np.save(os.path.join(output_dir,f'EyeData_td_{vid_txt}'), eye_data_td)
    
    np.save(os.path.join(output_dir,f'MouthData_asd_{vid_txt}'), mouth_data_asd)
    np.save(os.path.join(output_dir,f'MouthData_td_{vid_txt}'), mouth_data_td)

    
#%%
    num_of_faces = np.asarray([ 0 if fii is None else len(fii) for fii in face_use_info ])
    np.save(os.path.join(output_dir,f'frames_num_of_faces_{vid_txt}'), num_of_faces)

    np.save(os.path.join(output_dir,f'Facegaze_asd_split_{vid_txt}'), face_gaze[asd_subjs])
    np.save(os.path.join(output_dir,f'Facegaze_td_split_{vid_txt}'), face_gaze[td_subjs])

    np.save(os.path.join(output_dir,f'Eyegaze_asd_split_{vid_txt}'), eye_gaze[asd_subjs])
    np.save(os.path.join(output_dir,f'Eyegaze_td_split_{vid_txt}'), eye_gaze[td_subjs])

    np.save(os.path.join(output_dir,f'Mouthgaze_asd_split_{vid_txt}'), mouth_gaze[asd_subjs])
    np.save(os.path.join(output_dir,f'Mouthgaze_td_split_{vid_txt}'), mouth_gaze[td_subjs])

    