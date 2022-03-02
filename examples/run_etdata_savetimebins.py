#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
from alabeye.etdata import ETdata

# Main directory for experiment data.
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'

# HDF file of gaze data. (see ../preprocess_rawdata about how to prepare this file)
hdf_file_list = [ 'ET_Tobii_Ep1_v0.hdf5', 'ET_Tobii_Ep4_AQNR_v0.hdf5' ]


subj_info_file = os.path.join(root_dir,'BehavioralData','participants_info.csv')
stim_dir = os.path.join(root_dir,'StimVids')

timebin_outdir = os.path.join(root_dir,'ETdata','down2frame_data_test')

for h5ii in hdf_file_list:
    
    hdf_file = os.path.join(root_dir,'ETdata',h5ii)
    
    etdata_init = ETdata(data_file=hdf_file)

    for task_ii in etdata_init.available_tasks:
        
        print(f'processing {task_ii}')
        etdata_task = ETdata(taskname=task_ii, data_file=hdf_file, 
                             subj_info_file=subj_info_file, use_subj_info=True,
                             stim_dir=stim_dir)

        # etdata_task.load_rawdata()
        
        # default is to bin to frame duration.
        etdata_task.load_timebinned_data(split_groups=True, fix_length=True,
                                         save_output=True, output_dir=timebin_outdir,
                                         rm_subj_withhalfmissingdata=True)
        
