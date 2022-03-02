#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os

from alabeye.etdata import ETdata

# Main directory for experiment data.
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'


pickle_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data/ETdata/down2frame_data'

subj_info_file = os.path.join(root_dir,'BehavioralData','participants_info.csv')
stim_dir = os.path.join(root_dir,'StimVids')

pickle_input_dir = os.path.join(root_dir,'ETdata','down2frame_data')

pickle_file = '/home/umit/Documents/Research_ET/AutismVids/avp_data/ETdata/down2frame_data/timebinned_data_Ep1_Clip1.pkl'

# Just to load data. 
# etdata_video = ETdata(data_file=pickle_file)

# Load data and stimulus video media info, such as fps, the number of frames, etc.
etdata_video = ETdata(data_file=pickle_file, stim_dir=stim_dir)


output_dir = '/home/umit/Documents/Research_ET/AutismVids/sample_viz'


# etdata_video.visualize_gaze(merge_groups=False,show_viz=True,
                           # save_viz=True,output_dir=output_dir)

# etdata_video.visualize_2groups(show_viz=True,
                               # save_viz=True,output_dir=output_dir)

    