#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

# Walk through subject-wise data folders, laod relevant information from three types of files:
- metadata ....mat file, ..._EVENTS.txt, ..._TRACKING.txt 
and save into a .HDF file. 

use for Episode 1.

--- To run, change directory names: ---
exp_folder (directory consisting of raw data collected in the experiment)
hdf_output_file (file name for output hdf file)

"""

#%%
import os
import re
import numpy as np

from avp_helpers import get_avp_rawdata

#%%

# Eye-tracking data (location of Tobii ET data)
exp_folder = '/media/umit/C015-BC49/ETdata/TobiiETdata_Ep1_3clips' 

# Output file to save processed ET data. 
hdf_output_file = '/home/umit/Documents/Research_ET/AutismVids/ET_Tobii_Ep1_v0.hdf5'
hdf_output_dir = os.path.dirname(hdf_output_file)

# if ET data contains very large number of subjects or very long tracking durations, 
# it might be better to compress hdf file. 
compress_hdf = True

# While loading subject-wise data, control whether these parameters are consistent across subjects. 
expected_resolution_vals_Cal = {'mov_height_w2':480, 'mov_width_w2':720, 'ScreenHeight_w1':1080, 'ScreenWidth_w1':1920, 'ScreenHeight_w2':900, 'ScreenWidth_w2': 1440}
expected_resolution_vals_IU = {'mov_height_w2':480, 'mov_width_w2':720, 'ScreenHeight_w1':1080, 'ScreenWidth_w1':1920, 'ScreenHeight_w2':1440, 'ScreenWidth_w2': 2560}  

if not os.path.exists(hdf_output_dir):
    os.makedirs(hdf_output_dir)

first_entry = True # to open an HDF file. 
subj_folders = sorted(os.listdir(exp_folder))

# Keep some values across subjects to learn average pixel per degree of visual angle. 
keep_pixelperDVA = []
keep_pixelperDVA_altv = []

# Sampling frequency of ET data. 
keep_rec_time_diff = []
keep_rec_time_diff_median = []
keep_subjID = []

keep_calib_quality = {} 

# Walk through subject-wise data folders. 
for subj in subj_folders:
    
    print(subj)
    if len(subj) != 6 or ' ' in subj:
        raise ValueError('Unexpected subj folder format at:\n%s'%os.path.join(exp_folder,subj))
    
    ET_files = sorted(os.listdir(os.path.join(exp_folder,subj)))
    ET_files = [ f for f in ET_files if re.match(r'.*\.(mat|txt)', f, flags=re.I) and not '~' in f ] # and not 'calib_quality' in f]

    # for each recording we expect 3 files: Events, Tracking, metadata .mat, 
    # otherwise probably theres is a problem...
    # if np.divmod(len(ET_files),3)[1] != 0: 
        # print('a potential problem in %s'%os.path.join(exp_folder,subj))


    this_calib_quality = {}
    for f in ET_files:
        
        if not (f[:len(subj)] == subj or f.split('_')[0] ==  subj[2:] ): # all filenames start with subj-ID. 
            print('Mismatch between subj-ID and filename at subj file:\n --> %s'%os.path.join(exp_folder,subj,f))
            
        # if f.split('_')[1] != 'tobii':
            # raise SystemExit('Unexpected file format at subj file:\n%s'%os.path.join(exp_folder,subj,f)) 
        
        if 'EVENTS' in f:
            # split on the last occurance: # or usef.rpartition('_')
            session_name = f.rsplit('_',1)[0]
            # Tacking file:
            session_ETdata = '%s_TRACKING.txt'%session_name
            if not session_ETdata in ET_files: raise SystemExit('TRACKING.txt file is not found for %s'%session_ETdata)
            session_metafile = '%s.mat'%session_name
            if not session_metafile in ET_files: raise SystemExit('Metadata file is not found for %s'%session_metafile)
            session_calibqaul_file = '%s_calib_quality.mat'%session_name
            if not session_calibqaul_file in ET_files: raise SystemExit('Calib_auality file is not found for %s'%session_calibqaul_file)
                        
            session_eventsfile = f
            
            # Corresponding video clip:
            if 'Ep1_Clip1' in f: session_video = 'Ep1_Clip1' 
            elif 'Ep1_Clip2' in f: session_video = 'Ep1_Clip2' 
            elif 'Ep1_Clip3' in f: session_video = 'Ep1_Clip3'
            else: raise SystemExit('Undefined clip!')
            
            session_ETdata_fullpath = os.path.join(exp_folder,subj,session_ETdata)
            session_eventsfile_fullpath = os.path.join(exp_folder,subj,session_eventsfile)
            session_metafile_fullpath = os.path.join(exp_folder,subj,session_metafile)
            session_calibqaul_fullpath = os.path.join(exp_folder,subj,session_calibqaul_file)

            assert os.path.isfile(session_ETdata_fullpath), 'File does not exist: %s'%session_ETdata_fullpath
            assert os.path.isfile(session_eventsfile_fullpath), 'File does not exist: %s'%session_eventsfile_fullpath
            assert os.path.isfile(session_metafile_fullpath), 'File does not exist: %s'%session_metafile_fullpath
            assert os.path.isfile(session_calibqaul_fullpath), 'File does not exist: %s'%session_calibqaul_fullpath

            if subj.startswith('RA'):
                expected_resolution_vals = expected_resolution_vals_Cal.copy()
            elif subj.startswith('C') or subj.startswith('A'):
                expected_resolution_vals = expected_resolution_vals_IU.copy()
            else:
                raise ValueError('Undefined subj name format!')
                
            ETdata_session_df, pixelperDVA, pixelperDVA_altv, calib_quality = get_avp_rawdata(session_ETdata_fullpath,session_eventsfile_fullpath,
                                                                session_metafile_fullpath,session_calibqaul_fullpath,scale2frame=True,**expected_resolution_vals)

            # print(ETdata_session_df['RecTime'].values[-1])

            keep_pixelperDVA.append(pixelperDVA)
            keep_pixelperDVA_altv.append(pixelperDVA_altv)

            session_ID = '%s/%s'%(subj,session_video)

            if first_entry:
                if compress_hdf:
                    ETdata_session_df.to_hdf(hdf_output_file,session_ID,mode='w',complevel=9, complib='zlib')
                else:    
                    ETdata_session_df.to_hdf(hdf_output_file,session_ID,mode='w')
                first_entry=False
            else:
                if compress_hdf:
                    ETdata_session_df.to_hdf(hdf_output_file,session_ID,mode='a',complevel=9, complib='zlib')
                else:
                    ETdata_session_df.to_hdf(hdf_output_file,session_ID,mode='a')


            this_calib_quality[session_video] = calib_quality
            
            rec_times = ETdata_session_df['RecTime'].values
            rec_times_diffs = np.diff(rec_times)
            
            keep_rec_time_diff.append(np.mean(rec_times_diffs))
            keep_rec_time_diff_median.append(np.median(rec_times_diffs))
            

            keep_subjID.append('%s_%s'%(subj,session_video))          
    
    # collect calibration quality metric.        
    keep_calib_quality[subj] = this_calib_quality
            
        
# 
print()
print('pixelperDVA mean=%3.3f, median=%3.3f'%(np.nanmean(keep_pixelperDVA),np.nanmedian(keep_pixelperDVA)) )
print('pixelperDVA_altv mean=%3.3f, median=%3.3f'%(np.mean(keep_pixelperDVA_altv),np.median(keep_pixelperDVA_altv)) )

import pandas as pd
sampling_info_pd = pd.DataFrame({'SubjIDs': keep_subjID, 'SamplingTime (ms)':np.round(keep_rec_time_diff_median,5)*1000. })
sampling_info_pd.to_csv(os.path.join(hdf_output_dir,'sampling_info_Ep1_v0.csv'))


import pickle
pkl_output_file = os.path.join(hdf_output_dir,'CalibQuality_Ep1_v0.pkl')
with open(pkl_output_file,'wb') as pfile:
    pickle.dump(keep_calib_quality,pfile)


