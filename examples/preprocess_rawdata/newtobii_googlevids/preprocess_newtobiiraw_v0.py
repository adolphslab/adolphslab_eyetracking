#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Load Na Yeon's mat files and re-store in HDF file format.  

"""


import os
import numpy as np
import pandas as pd
import pims


from alabeye.io.loaddata import loadmat
from alabeye.stats import nanmasked_mean


# Directory of collected eye-tracking data (location of Tobii ET data)
exp_folder = '/media/umit/9E76-DC46/ETdata/NaYeon/Tobii_Data/nTobiiETdata_allclips'

# Directory of stimulus videos
stim_dir = '/media/umit/9E76-DC46/ETdata/NaYeon/StimVids'


# Output file to save processed ET data. 
hdf_output_file = '/home/umit/Documents/Research_ET/AutismNData/ET_nTobii_v0.hdf5'
hdf_output_dir = os.path.dirname(hdf_output_file)

# if ET data contains very large number of subjects or very long tracking durations, 
# it might be better to compress hdf file. Compressed size is nearly the half of non-compressed size. 
# But compressing/uncompressing is a slow process.
compress_hdf = False


# Walk through subject-wise data folders. 
subj_files = sorted(os.listdir(exp_folder))
first_entry = True # to open an HDF file. 
video_metainfo = {}

for fname_ii in subj_files:
    
    subj = fname_ii.split('_')[0]
    print(subj)
    if len(subj) != 6 or ' ' in subj:
        raise ValueError(f'Unexpected subj ID format: {subj}')
    
    # Read .mat file to load experiment data. 
    gaze_matfile = loadmat(os.path.join(exp_folder,fname_ii))

    # Load raw gaze data for all movies. 
    gaze_data_all = gaze_matfile['AllData']
    vids_this = list(gaze_data_all.keys())
    
    # Load experiment parameters to rescale raw gaze data to movie frame size.
    params = gaze_matfile['Params']
    screenrect = params['ScreenRect']
    placeholder = params['placeholder']
    
    # from Na Yeon's scripts. 
    movHshown = placeholder[3] - placeholder[1]
    movWshown = placeholder[2] - placeholder[0]
    scale_factor = screenrect[3]/movHshown
    sub_factor = (screenrect[3]/movHshown - 1)/2

    for vid_ii in vids_this:

        # Rescale raw gaze data to movie frame size and save to an hdf file. 
        gaze_raw = gaze_data_all[vid_ii]
        
        data_len = len(gaze_raw['device_timestamp'])
        extra_info = {}
        for key,val in gaze_raw.items():
            if isinstance(val,int):
                extra_info[key] = val
            elif len(val) != data_len:
                extra_info[key] = val
        
        # delete extra_info from gaze_raw before converting to dataframe.
        for key in extra_info.keys():
            gaze_raw.pop(key)
        
        # Actual file name of the video.
        vid_fname = f"{vid_ii.replace('mv','')}.mp4"
        if vid_fname not in video_metainfo:
            # define VideoReader object and get metadata from the video.
            video_file = os.path.join(stim_dir,vid_fname)
            if not os.path.isfile(video_file):
                raise SystemExit(f'Could not find the video file:\n{video_file}')
            
            vr = pims.PyAVReaderTimed(video_file)
            frame_width, frame_height = vr.frame_shape[1], vr.frame_shape[0]
            vid_duration, nframes, vid_fps = vr.duration, len(vr), vr.frame_rate

            video_metainfo[vid_fname] = {'frame_width':frame_width, 
                                         'frame_height':frame_height,
                                         'vid_duration':vid_duration,
                                         'nframes':nframes,'vid_fps':vid_fps}

        # take and save relevant info.
        gaze_data_df = pd.DataFrame.from_dict(gaze_raw,dtype=float)
        
        frame_width = video_metainfo[vid_fname]['frame_width']
        frame_height = video_metainfo[vid_fname]['frame_height']
        
        # the same as first averaging left and right eyes and scaling to frame size. 
        gaze_data_df[ ['left_x_coord', 'right_x_coord'] ] = ( (gaze_data_df[ ['left_x_coord', 'right_x_coord'] ]*scale_factor - sub_factor)*frame_width)
        gaze_data_df[ ['left_y_coord', 'right_y_coord'] ] = ( (gaze_data_df[ ['left_y_coord', 'right_y_coord'] ]*scale_factor - sub_factor)*frame_height)
        
        gaze_x = nanmasked_mean(gaze_data_df[ ['left_x_coord', 'right_x_coord'] ].values, axis=1)
        gaze_y = nanmasked_mean(gaze_data_df[ ['left_y_coord', 'right_y_coord'] ].values, axis=1)
        
        # it is tricky to average pupil diameters from two eyes. Let's keep both for now. 
        # pupil_diameter = nanmasked_mean(gaze_data_df[ ['left_pupil_diameter', 'right_pupil_diameter'] ].values, axis=1)

        # put data into a format that we used in previous studies. 
        ETdata_df = pd.DataFrame({'RecTime': gaze_data_df['device_timestamp'].values, 
                                  'GazeX': gaze_x, 'GazeY': gaze_y, 
                                  'PupilDiameter_left':gaze_data_df['left_pupil_diameter'].values,
                                  'PupilDiameter_right':gaze_data_df['right_pupil_diameter'].values })

        session_ID = '%s/%s'%(subj,vid_ii) # using original stimulus name used in .mat files, not vid_fname.
        if first_entry:
            if compress_hdf:
                ETdata_df.to_hdf(hdf_output_file,session_ID,mode='w',complevel=9, complib='zlib')
            else:    
                ETdata_df.to_hdf(hdf_output_file,session_ID,mode='w')
            first_entry=False
        else:
            if compress_hdf:
                ETdata_df.to_hdf(hdf_output_file,session_ID,mode='a',complevel=9, complib='zlib')
            else:
                ETdata_df.to_hdf(hdf_output_file,session_ID,mode='a')


        # ---------------------------------------------------------------------
        # TODO: add calibration quality analysis/metric.
        # ---------------------------------------------------------------------
        
