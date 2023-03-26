#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import os
import re
import json
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm

from .io.loaddata import get_subj_info, run_et_timebins
from .stimulusdata import get_video_info
from .viz import plot_gaze_basic, plot_compare_2groups

implemented_tasktypes = ['video']


def makedir(output_dir, warn=True, verbose=True, sysexit=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose: 
            print(f'{output_dir} is ready.')
        return True
    else:
        if sysexit:
            raise SystemExit(f'\nThe folder {output_dir} already exists. Risk of overwriting files!'+
                             "\nEnter another output_dir, allow overwriting with 'gendir(output_dir, sysexit=False)', "+
                             "or delete 'output_dir' manually.\n")
        if warn:
            if verbose:
                print(f'\nThe folder {output_dir} already exists. Risk of overwriting files!\n') 
            return False
        else:
            if verbose:
                print(f'\nThe folder {output_dir} already exists. Overwriting files!\n') 
            return True


class ETdata:
    """ETdata class provides an interface for loading and handling eye-tracking data. 
    """
    
    def __init__(self, taskname=None, tasktype='video', 
                 data_file=None, subj_info_file=None, use_subj_info=False,
                 stim_dir=None, stimname_map=None, load_data=False):
        
        self.taskname = taskname
        
        # tasktype can be 'video', 'image', 'nostim', etc. depending on experiment task.
        if tasktype not in implemented_tasktypes:
            raise NotImplementedError(f"Task type '{tasktype}' is not implemented yet!\n" +
                                      f'\tImplemented task types are: {implemented_tasktypes!r}')
        self.tasktype = tasktype

        self.data_file = data_file
        self.subj_info_file = subj_info_file
        self.use_subj_info = use_subj_info
        self.stim_dir = stim_dir
        self.stimname_map = stimname_map
        
        self._load_hdf = False
        self._load_pkl = False
        if data_file is not None:
            if not os.path.isfile(data_file):
                raise SystemExit(f'Could not find the data_file: {data_file}!')

            if bool(re.match(r'.*\.(hdf|h5|hdf5)', data_file, flags=re.I)):
                self._load_hdf = True
            elif data_file.endswith(('.pkl', '.pickle', '.p')):
                self._load_pkl = True
            else:
                raise ValueError('Unknown file extension in data_file!')
            
        
        if subj_info_file is not None:
            if not os.path.isfile(subj_info_file):
                raise SystemExit(f'Could not find: {subj_info_file}!')
        
            if not subj_info_file.endswith('.csv'):
                raise SystemExit("Subj info file is not in assumed format of a csv file with " +
                                 " at least two columns of subject 'ID' and 'Group' !")
                

        if self._load_hdf:
            # load basic info.
            subjs_all, vidclips, datakeys, subjs_group = get_subj_info(data_file, subj_info_file, use_subj_info)
            self.available_tasks = vidclips
            
            if self.taskname is not None and self.taskname not in vidclips:
                print(f'Available tasks in data_file are: {vidclips!r}')
                print(f"---> Entered taskname '{self.taskname}' is not in the list!")
            
            self.hdf_allsubjs = subjs_all
            self.subjs = [ sii for sii in subjs_all if f'/{sii}/{self.taskname}' in datakeys ]
            
            if subjs_group is None:
                self.subjs_group = None
                self.ngroups = None
            else:
                self.subjs_group = [subjs_group[s_idx] for s_idx,sii in enumerate(subjs_all) \
                                    if f'/{sii}/{self.taskname}' in datakeys ]
                self.ngroups = len(np.unique(self.subjs_group))

            self.hdf_datakeys = datakeys
            
            if load_data:
                self.load_rawdata()


        if self._load_pkl:
            with open(self.data_file, 'rb') as f:
                data2load = pickle.load(f)
            
            self.taskname = data2load['taskname']
            self.available_tasks = data2load['taskname']
            self.data = data2load['et_data']
            self.data_subjs = data2load['subjs']
            self.data_subjs_group = data2load['group_info']
            self.data_ngroups = len(data2load['group_info'])
            assert self.data_ngroups == len(self.data)
            assert self.data_ngroups == len(self.data_subjs)
            print(f"Loaded data for taskname: {self.taskname}")            


        if self.stim_dir is not None:
            
            if not os.path.isdir(stim_dir):
                raise SystemExit(f'Could not find the directory: {stim_dir}!')
            
            if self.tasktype=='video' and self.taskname is not None:
                    self._set_stim_mediainfo()

    # getter
    def _get_stim_mediainfo(self):
        try:
            return self._stim_mediainfo
        except AttributeError:
            print('stim_mediainfo has not been initialized yet! \n' + \
                  "   Enter 'taskname' and 'stim_dir' info while calling ETdata()!\n")
    
    # setter
    def _set_stim_mediainfo(self):

        video_info = get_video_info(self.stim_dir)
        
        if self.stimname_map is None:
            # try reading from stim_dir
            json_file = os.path.join(self.stim_dir,'stimname_map.json')
            if os.path.isfile(json_file):
                with open(json_file, 'r') as fp:
                    stimname_map = json.load(fp)
                self.stimname_map = stimname_map
            else:
                raise SystemExit("Need to enter 'stimname_map' to read video media info!")
            
        if self.taskname not in self.stimname_map:
            raise SystemExit(f"taskname '{self.taskname}' is not in {list(self.stimname_map.keys())!r}!")
        
        self.stim_videoname = self.stimname_map[self.taskname]
        self._stim_mediainfo = video_info[self.stimname_map[self.taskname]]

    stim_mediainfo = property(_get_stim_mediainfo, _set_stim_mediainfo)


    def load_rawdata(self, subjs_use=None, skip_these_subjs=[]):
        
        if self.taskname is None:
            raise SystemExit('Need to define a taskname first to load the corresponding raw data!')
        
        if self._load_hdf is False:
            raise SystemExit("Please provide an hdf file as 'data_file' in ETdata() to load raw data!")
        
        if skip_these_subjs is None:
            skip_these_subjs = []

        if isinstance(subjs_use, str):
            subjs_use = [ subjs_use ]
        
        if subjs_use is None:
            subjs_use = self.subjs
        
        if len(subjs_use) == 0:
            print("No information was provided about subjects using 'subj_info_file'."+\
                  "Loading data from all subjects available in the hdf file!")
            subjs_use = self.hdf_allsubjs
        
        self.rawdata = {}
        print('Loading raw data...')
        for sii in tqdm(subjs_use):
            
            if sii in skip_these_subjs:
                continue

            key2load = f'/{sii}/{self.taskname}'
            if key2load not in self.hdf_datakeys:
                raise SystemExit(f'{key2load} is not available in hdf file!\n' +\
                                 'Please check the inputs you provided!')
            
            et_data_pd = pd.read_hdf(self.data_file, key2load, mode='r')
            self.rawdata[sii] = et_data_pd
            
    
    def get_timebinned_data(self, timebin_sec='frame_duration', 
                            subjs_use=None, skip_these_subjs=[], split_groups=False, 
                            save_output=False, output_dir=None, output_overwrite=True,
                            rm_subj_withhalfmissingdata=False, bin_operation='mean',
                            fix_length=False, nbins=None):
        
        if self._load_hdf is False:
            raise SystemExit("Please provide an hdf file as 'data_file' to load raw data!")

        if split_groups and self.subjs_group is None:
            raise SystemExit("To split groups on should enter a 'subj_info_file'" + 
                             " and set 'use_subj_info=True in ETdata() initialization.")
        
        if save_output and output_dir is None:
            raise SystemExit("Please provide a directory location 'output_dir' to save time binned data!")
        
        if save_output:
            if output_overwrite:
                gen_success = makedir(output_dir,warn=False,verbose=False)
                assert gen_success
            else:
                gen_success = makedir(output_dir,warn=True,verbose=True)
                if not gen_success:
                    raise SystemExit("Enter another 'output_dir', delete 'output_dir' manually,"+
                                     " or set 'output_overwrite=True'")
                    
            pickle_file = os.path.join(output_dir,f'timebinned_data_{self.taskname}_{timebin_sec}.pkl')
            if os.path.isfile(pickle_file) and not output_overwrite:
                raise SystemExit(f"Overwriting pickle file: {pickle_file}" + 
                                 "Please change 'output_dir' or delete the file manually!")

        if skip_these_subjs is None:
            skip_these_subjs = []

        if subjs_use is None:
            subjs_use = self.subjs
        
        if not isinstance(subjs_use,list):
            subjs_use = subjs_use.tolist()
        
        if split_groups:
            _subjs_group = [ self.subjs_group[self.subjs.index(sii)] for sii in subjs_use ] 
        else:
            _subjs_group = None
            
        if isinstance(timebin_sec, str):
            if timebin_sec != 'frame_duration':
                raise ValueError("timebin_sec should be either 'frame_duration' or a duration in seconds!")
            
            vid_fps = self.stim_mediainfo['fps']
            timebin_sec = 1./vid_fps # frame duration in sec.
            
            if nbins is None:
                nbins = self.stim_mediainfo['nframes']
        else:
            if fix_length and nbins is None:
                print("\n--->'nbins' not provided. Fixing length across subjects using total video duration...\n")

                frame_duration_sec = 1. / self.stim_mediainfo['fps']
                # using frame_duration_sec * nframes is in general safer than depending on vid_duration_sec, 
                # especially while using pims.PyAVReaderIndexed()
                # nbins = int(frame_duration_sec * nframes / timebin_sec) # int() rounds to floor
                nbins = np.round(frame_duration_sec * self.stim_mediainfo['nframes'] / timebin_sec).astype(int) # last bin will be smaller.
                
    
        print('Loading raw data and applying timebins...')
        # Note that if rm_subj_withhalfmissingdata=True, bins_subjs might be different than subjs_use.
        bins_subjs, bins_groups, bins_etdata = run_et_timebins(self.taskname, subjs_use, 
                                               self.data_file, self.hdf_datakeys, timebin_sec, 
                                               rm_subj_withhalfmissingdata, bin_operation = bin_operation,
                                               split_groups=split_groups, subjs_group=_subjs_group,
                                               fix_length=fix_length, nbins=nbins)

        self.data = bins_etdata
        self.data_subjs = bins_subjs
        self.data_subjs_group = bins_groups
         
        if save_output:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump({'taskname':self.taskname, 'et_data':bins_etdata, 
                             'subjs':bins_subjs, 'group_info':bins_groups }, pfile)


    def visualize_gaze(self, merge_groups=False,
                       colors=['C3', 'C0', 'C1', 'C9'], zorder=[1,2,3,4],
                       save_viz=False, output_dir='gaze_visualization', 
                       show_viz=False, prep_viz=True):

        video_file = os.path.join(self.stim_dir,self.stim_videoname)

        try:
            self.data
        except AttributeError:
            print("---> Generating timebinned data automatically. If you have specific settings, "+\
                  "use ETdata.get_timebinned_data() before running ETdata.visualize_gaze()")
            self.get_timebinned_data(timebin_sec='frame_duration',
                                     split_groups=True, bin_operation='mean',
                                     fix_length=True)

        # for general use, this instance method is defined as a separate function. 
        plot_gaze_basic(self.data, video_file, merge_groups=merge_groups,
                        colors=colors, zorder=zorder,
                        save_viz=save_viz, output_dir=output_dir, 
                        show_viz=show_viz, prep_viz=prep_viz)


    def visualize_2groups(self, sigma=21, plot_groups=[0,1],
                       colors=['C3', 'C0', 'C1', 'C9'], zorder=[1,2,3,4],
                       save_viz=False, output_dir='gaze_visualization', 
                       show_viz=False, prep_viz=True):

        video_file = os.path.join(self.stim_dir,self.stim_videoname)

        try:
            self.data
        except AttributeError:
            print("---> Generating timebinned data automatically. If you have specific settings, "+\
                  "use ETdata.get_timebinned_data() before running ETdata.visualize_2groups()")
            self.get_timebinned_data(timebin_sec='frame_duration',
                                     split_groups=True, bin_operation='mean',
                                     fix_length=True)

        # for general use, this instance method is defined as a separate function. 
        plot_compare_2groups(self.data, video_file, sigma=sigma, plot_groups=plot_groups,
                             colors=colors, zorder=zorder,
                             save_viz=save_viz, output_dir=output_dir, 
                             show_viz=show_viz, prep_viz=prep_viz)

