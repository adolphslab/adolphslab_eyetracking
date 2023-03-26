#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
import re
from pims import PyAVReaderTimed, PyAVReaderIndexed
import json

def get_video_info(stimvids_dir):

    json_file = os.path.join(stimvids_dir,'vidsinfo.json')
    if os.path.isfile(json_file):
        with open(json_file, 'r') as fp:
            video_info = json.load(fp)
        return video_info
        
    # if json_file does not exist, then collect media info from video files themselves. 
    videofiles = sorted(os.listdir(stimvids_dir))
    # Get only the video files. Add other media file extensions, if any used. 
    videofiles = [ f for f in videofiles if re.match(r'.*\.(mov|mp4|avi)', f, flags=re.I)]
    
    video_info = {}
    for vid_ii in videofiles:
        print(f'\nProcessing: {vid_ii}...')
        video_file = os.path.join(stimvids_dir,vid_ii) 
    
        vr, vid_info = load_video(video_file)
        video_info[vid_ii] = vid_info
    
    # save values to have a quick access for later use
    if not os.path.isfile(json_file):
        with open(json_file, 'w') as fp:
            json.dump(video_info, fp, indent=True)
    
    return video_info


def load_video(video_file):
    
    # define VideoReader object and video info.
    vr = PyAVReaderTimed(video_file)
    frame_width, frame_height = vr.frame_shape[1], vr.frame_shape[0]
    vid_duration, _, vid_fps = vr.duration, len(vr), vr.frame_rate
    
    # indexed version provides a more accurate count for the number of frames
    vr = PyAVReaderIndexed(video_file)
    nframes = len(vr)
    
    vid_info = {'duration':vid_duration, 'nframes':nframes, 'fps':vid_fps, 
                'frame_width':frame_width, 'frame_height':frame_height }
    
    return vr, vid_info
    
    