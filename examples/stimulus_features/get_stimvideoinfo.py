#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Get and save video metadata info.

Feature extraction steps can be included here. 
 
"""


import os
import re
import pims

# Location of the videos used in the experiment.
stimvids_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data/StimVids'
# stimvids_dir = '/media/umit/9E76-DC46/ETdata/NaYeon/StimVids'


# Files in the folder. 
videofiles = sorted(os.listdir(stimvids_dir))
# Get only the video files. Add other media file extensions, if any used. 
videofiles = [ f for f in videofiles if re.match(r'.*\.(mov|mp4|avi)', f, flags=re.I)]

vids_info = {}
for vid_ii in videofiles:
    print('\nProcessing: %s...'%vid_ii)
    video_file = os.path.join(stimvids_dir,vid_ii) 

    # define VideoReader object and video info.
    vr = pims.PyAVReaderTimed(video_file)
    frame_width, frame_height = vr.frame_shape[1], vr.frame_shape[0]
    vid_duration, nframes, vid_fps = vr.duration, len(vr), vr.frame_rate
    
    dum_dict = {'duration':vid_duration, 'nframes':nframes, 'fps':vid_fps, 
                'frame_width':frame_width, 'frame_height':frame_height }
    vids_info[vid_ii] = dum_dict

    
# Save results. 
import json
with open(os.path.join(stimvids_dir,'vidsinfo.json'), 'w') as fp:
    json.dump(vids_info, fp, indent=True)

