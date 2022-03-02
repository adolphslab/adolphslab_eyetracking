#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import pandas as pd
from tqdm import tqdm 

from ..stats import nanmasked_mean


# --- Load matlab .mat files ---
# from: https://stackoverflow.com/a/8832212
import scipy.io as spio
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def items_to_arrays(hdf_group,keys_load=None):
    arrays = {}
    for keys, vals in hdf_group.items():
        if keys_load is not None:
            if keys in keys_load:
                arrays[keys] = np.array(vals)
        else:
            arrays[keys] = np.array(vals)

    return arrays



def get_subj_info(hdf_file, subj_info_file=None, use_subj_info=False, use_subjs_across_clips=False):
    '''
    Read info related to available subjects and videos from a given HDF file.  
    
    if use_subj_info:
        Read subject IDs and their groups info (.e.g., ASD vs TD) from a given CSV file (subj_info_file).
    
    if subjs_across_clips:
        Keep subjects that have ET data across all videos in this HDf file. 
    '''
    
    # ----- Open hdf file to get info related to participant and used clips -----
    hdf = pd.HDFStore(hdf_file,'r')
    datakeys = hdf.keys()
    # Close HDF file, we will access it with pd.read_hdf() rather than hdf.get()
    hdf.close()
    
    # Data files with '_FrameInfo' extension is irrelavant at this point (if exists).
    datakeys = [ dii  for dii in datakeys if not 'FrameInfo' in dii]
    
    # Obtain participant (subject) IDs and video clips used in the experiment. 
    subjs = [ dii.split('/')[1]  for dii in datakeys]
    vidclips = [ dii.split('/')[2]  for dii in datakeys]

    if len(subjs) != len(vidclips): 
        raise SystemExit('Problem while reading subjects and video clips information!')
    
    subjs = np.unique(subjs).tolist()
    vidclips = np.unique(vidclips).tolist()

    if use_subjs_across_clips:
        subjs_keep = []
        for sii in subjs:
            vids_avail = [ True if f'/{sii}/{vii}' in datakeys else False for vii in vidclips ]
            if all(vids_avail):
                subjs_keep.append(sii)
                
        subjs = subjs_keep

    if not use_subj_info:
        return subjs, vidclips, datakeys, None

    # ------ Subject info file ------
    if use_subj_info:
        subj_info_pd = pd.read_csv(subj_info_file)
        subj_IDs_csv = subj_info_pd['ID'].values.tolist()
        subj_group_csv = subj_info_pd['Group'].values.astype(int)
    
        subj_csv = []
        subj_csv_group = []

        for sii in subjs:
            if sii in subj_IDs_csv:
                s_indx = subj_IDs_csv.index(sii)
                subj_csv.append(sii)
                subj_csv_group.append(subj_group_csv[s_indx])
            
        return subj_csv, vidclips, datakeys, subj_csv_group



def run_et_timebins(taskname, subjs, hdf_file, datakeys, timebin_msec, 
                    rm_missingdata, bin_operation='mean',
                    split_groups = False, subjs_group=None,
                    fix_length=False, nbins=None):
    
    if split_groups:
        assert len(subjs) == len(subjs_group)
    
    # if bin_operation is None:
        # raise NotImplementedError("bin_operation=None is not implemented here!")
    
    et_xy = []
    group_info = []
    subjs_used = []
    for cnt_ii, sii in enumerate(tqdm(subjs)):

        key2load = f'/{sii}/{taskname}'
        assert key2load in datakeys, print(f'{key2load} is not available in hdf file!')

        et_data_pd = pd.read_hdf(hdf_file, key2load, mode='r')
        
        gaze_xy_init = et_data_pd[['GazeX','GazeY']].values
        data_quality_dum_init = np.mean(gaze_xy_init,axis=1)

        if rm_missingdata:
            if ( np.isnan(data_quality_dum_init).sum() / len(data_quality_dum_init)) >= 0.5:
                print(f'More than half of the data points are missing for {key2load}. Skipped!')
                continue

        # Apply timebins: to make this faster set use_maskedmean=False.
        et_xy_binned = get_et_timebins(et_data_pd, timebin_msec, do_op=bin_operation, use_maskedmean=True, 
                                         fix_length=fix_length, nbins=nbins)
        
        if bin_operation in ['mean', 'maxrep']:
            et_xy_binned = np.vstack(et_xy_binned) # used to assess whether more than half of data points are missing.
            assert all(np.diff(et_xy_binned[:,0]) == 1), 'Some frames are missing! Need to fix this!'
        
            if rm_missingdata:
                data_quality_dum = np.mean(et_xy_binned,axis=1)
                if ( np.isnan(data_quality_dum).sum() / len(data_quality_dum)) >= 0.5:
                    print(f'More than half of the data points are missing for {key2load}. Skipped!')
                    continue

            if fix_length:
                if cnt_ii == 0:        
                    ntimebins_use = len(et_xy_binned)  # in case we removed some timebins from the beginning of the video. 
                else:
                    assert ntimebins_use == len(et_xy_binned), 'ntimebins_use should not change across subjects if fix_length=True!'

        # 
            et_xy.append(et_xy_binned[:,[1,2]])
        else:
            et_xy.append([ bin_xy[:,[1,2]] for bin_xy in et_xy_binned])

        subjs_used.append(sii) # if rm_subj_withhalfmissingdata=True, then some subjects will be removed.

        if split_groups:
            c_indx = subjs.index(sii) 
            group_info.append(subjs_group[c_indx])

            
    if len(np.unique(group_info)) == 1:
        print('There is only one subject group! Not splitting into groups!')
        split_groups = False

    if split_groups:
        et_xy_split = []
        subjs_split = []
        group_info_split = []
        
        for gii in np.unique(group_info):
            et_xy_group = [ et_ii for et_idx,et_ii in enumerate(et_xy) if group_info[et_idx]==gii ]
            if fix_length and bin_operation is not None:
                et_xy_split.append(np.asarray(et_xy_group))
            else:
                et_xy_split.append(et_xy_group)

            subjs_split.append([ sii for s_idx,sii in enumerate(subjs_used) if group_info[s_idx]==gii ])
            group_info_split.append([ _idx for _idx in group_info if _idx==gii ]) # absurd, but to use the same way as above splits.
                        
        return subjs_split, group_info_split, et_xy_split
    
    else:
        return [subjs_used], None, [et_xy]




def get_et_timebins(ET_data_pd, timebin_msec, do_op=None, use_maskedmean=True, data_unit='sec', 
                    fix_length=False, nbins=None, additional_column=None, fixation_info=None):
    #  do_op = 'mean' # None, 'mean' or 'maxrep'
    
    assert do_op in ['mean', 'maxrep', None], "do_op should be one of these: ['mean', 'maxrep', None]"
    
    if fix_length and nbins is None:
        raise SystemExit("if fix_length=True, then should enter 'nbins'!")
    
    if timebin_msec < 10: raise ValueError('Binsize should be in msecs!')
    if data_unit == 's' or data_unit == 'sec':
        if ET_data_pd['RecTime'].values[2] > 1.:  
            print('\nBe sure that RecTime in the input is in secs (first data: %3.3f)!'%(ET_data_pd['RecTime'].values[2]))
        ET_rec_time = ET_data_pd['RecTime'].values*1000. # second to msec.
    elif data_unit == 'ms' or data_unit == 'msec': 
        if ET_data_pd['RecTime'].values[2] < 1.:  print('\nBe sure that RecTime in the input is in msecs (first data: %3.3f)!'%(ET_data_pd['RecTime'].values[2])) 
        ET_rec_time = ET_data_pd['RecTime'].values # in msec.
    else:
        raise ValueError(f'Undefined data_unit: {data_unit}')

    # needed to split ET_data into chunks/bins of duration of a frame (in heatmap calculations this was binsize).
    chunks = (ET_rec_time//timebin_msec + 1).astype(int) # same as np.asarray([ np.divmod(ii,binsize)[0]+1 for ii in ET_rec_time ])
    if additional_column is None:
        ET_data4bin = np.hstack(( chunks.reshape(-1,1), ET_data_pd[['GazeX','GazeY']].values ))
    else:
        # to take care of pupil diameter.
        ET_data4bin = np.hstack(( chunks.reshape(-1,1), ET_data_pd[['GazeX','GazeY', additional_column]].values ))
        ET_data4bin[ET_data4bin[:,-1] == -1, -1] = np.NaN 
        ET_data4bin[ET_data4bin[:,-1] == 0, -1] = np.NaN 

    # 
    if fixation_info is not None:
        # print('Fixation info is used!')
        ET_data4bin[ fixation_info==0 ,1:] = np.NaN

    # use chunk indices to split data into bins/chunks. 
    ET_data4hp_bins = np.split(ET_data4bin,np.where(np.diff(ET_data4bin[:,0]))[0]+1)
    if do_op=='mean': 
        if use_maskedmean: ET_data4hp_bins = [ nanmasked_mean(eii,axis=0) for eii in ET_data4hp_bins ]
        else: ET_data4hp_bins = [ np.nanmean(eii,axis=0) for eii in ET_data4hp_bins ]
    elif do_op=='maxrep':
        ET_data4hp_bins_dum = []
        for sil, eii in enumerate(ET_data4hp_bins):
            if np.isnan(np.sum(eii,axis=1)).sum() == len(eii): 
                ET_data4hp_bins_dum.append(eii[0])
            else: 
                unq_vals,unq_reps = np.unique(eii[~np.isnan(np.sum(eii,axis=1))],axis=0,return_counts=True) # sum+isnan to remove NaN data points. 
                if len(unq_reps)>1: # if the unique elements have the same frequency take one of them. It is better than averaging. 
                    #print sil, eii
                    #print unq_vals[np.argmax(unq_reps)]
                    ET_data4hp_bins_dum.append(unq_vals[np.argmax(unq_reps)])
                else: ET_data4hp_bins_dum.append(unq_vals[0]) # There is only one unique array. 
        
        ET_data4hp_bins = ET_data4hp_bins_dum


    # add padding to the beginning if there is a problem in ET recording start time. 
    if do_op=='mean' or do_op=='maxrep': 
        if ET_data4hp_bins[0][0] != 1:
            if additional_column is None:
                add_part = [ np.array([ ad_ii, np.NaN, np.NaN]) for ad_ii in range(1,int(ET_data4hp_bins[0][0])) ]
            else:
                add_part = [ np.array([ ad_ii, np.NaN, np.NaN, np.NaN]) for ad_ii in range(1,int(ET_data4hp_bins[0][0])) ]
            ET_data4hp_bins = add_part + ET_data4hp_bins # Add NaNs to the beginning.
    else: 
        if ET_data4hp_bins[0][0,0] != 1:
            if additional_column is None:
                add_part = [ np.array([ ad_ii, np.NaN, np.NaN]).reshape(-1,3) for ad_ii in range(1,int(ET_data4hp_bins[0][0,0])) ]
            else:
                add_part = [ np.array([ ad_ii, np.NaN, np.NaN, np.NaN]).reshape(-1,4) for ad_ii in range(1,int(ET_data4hp_bins[0][0,0])) ]
            ET_data4hp_bins = add_part + ET_data4hp_bins # Add NaNs to the beginning.


    # fill missing frames.
    ET_data4hp_bins_filled = []
    if do_op=='mean' or do_op=='maxrep':
        cnt_etd = 1
        for etd_ii in ET_data4hp_bins:
            while cnt_etd < etd_ii[0]:
                if additional_column is None:
                    ET_data4hp_bins_filled.append( np.array([ cnt_etd, np.NaN, np.NaN]) )
                else:
                    ET_data4hp_bins_filled.append( np.array([ cnt_etd, np.NaN, np.NaN, np.NaN]) )
                cnt_etd += 1
            ET_data4hp_bins_filled.append(etd_ii)
            cnt_etd += 1
    else: 
        cnt_etd = 1
        for etd_ii in ET_data4hp_bins:
            while cnt_etd < etd_ii[0,0]:
                if additional_column is None:
                    ET_data4hp_bins_filled.append( np.array([ cnt_etd, np.NaN, np.NaN]).reshape(-1,3) )
                else:    
                    ET_data4hp_bins_filled.append( np.array([ cnt_etd, np.NaN, np.NaN, np.NaN]).reshape(-1,4) )
                cnt_etd += 1
            ET_data4hp_bins_filled.append(etd_ii)
            cnt_etd += 1
            
    
    # Extend or cut to nframes.     
    if fix_length:
        if len(ET_data4hp_bins_filled) < nbins:
            while len(ET_data4hp_bins_filled) < nbins:
                if do_op=='mean' or do_op=='maxrep': 
                    if additional_column is None:
                        ET_data4hp_bins_filled.append( np.array([ET_data4hp_bins_filled[-1][0]+1, np.NaN, np.NaN]) ) # Extend with NaNs. 
                    else:
                        ET_data4hp_bins_filled.append( np.array([ET_data4hp_bins_filled[-1][0]+1, np.NaN, np.NaN, np.NaN]) ) # Extend with NaNs. 
                else:
                    if additional_column is None:
                        ET_data4hp_bins_filled.append( np.array([ET_data4hp_bins_filled[-1][0][0]+1, np.NaN, np.NaN]).reshape(-1,3)) # Extend with NaNs. 
                    else:    
                        ET_data4hp_bins_filled.append( np.array([ET_data4hp_bins_filled[-1][0][0]+1, np.NaN, np.NaN, np.NaN]).reshape(-1,4)) # Extend with NaNs. 
                        
        
        elif len(ET_data4hp_bins_filled) > nbins:
            ET_data4hp_bins_filled = ET_data4hp_bins_filled[:nbins]


    return ET_data4hp_bins_filled


