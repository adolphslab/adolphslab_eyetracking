#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


"""


import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import pandas as pd

from .stats import cohen_d_ci


# --- Heatmap utilities ---
def get_heatmap_sci(x, y, sigma=None, framesize = [1000,1000] ):
    if sigma is None: 
        raise ValueError('Be sure that sigma value is proper!')
    
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=framesize, 
                                range = [[0,framesize[0]],[0,framesize[1]]])
    # needed .T due to the settings of histogram2d's output. 
    heatmap = gaussian_filter(heatmap.T, sigma=sigma, 
                              mode='constant',cval=0.0) 

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap, extent


def et_heatmap(et_xy_in, framesize, sigma, hp_down_factor, get_full=False, get_down=True,
               nan_ratio=1.0, cut_pad_down=None, cut_pad_up=None, trivial_return=True):
    # et_xy_in: [n_samples, 2], columns are x and y components
    # framesize = [frame_width,frame_height]
    
    if et_xy_in is not None and et_xy_in.ndim == 1:
        et_xy_in = et_xy_in[np.newaxis]
    
    # here we can eliminate video blocks which contain more than 50% problematic ET points.
    if et_xy_in is None or np.isnan(et_xy_in).any(axis=1).sum() >= nan_ratio*et_xy_in.shape[0]:
        trivial_output = True
        if get_down:
            if trivial_return:
                heatmap_down = None
            else:
                heatmap_down = np.zeros((np.array(framesize[::-1])/hp_down_factor).astype(int))
        if get_full:
            if trivial_return:
                heatmap = None
            else:
                heatmap = np.zeros((np.array(framesize[::-1])).astype(int))
    else:
        trivial_output = False
        if get_down:
            heatmap_down, _ = get_heatmap_sci(et_xy_in[:,0]/hp_down_factor,et_xy_in[:,1]/hp_down_factor,
                             sigma=sigma/float(hp_down_factor),framesize=(np.array(framesize)/hp_down_factor).astype(int))
        if get_full:
            heatmap, _ = get_heatmap_sci(et_xy_in[:,0],et_xy_in[:,1],sigma=sigma,framesize=framesize)


    if (not trivial_output) and (cut_pad_down is not None) and (cut_pad_up is not None):
        
        if get_down and not get_full:
            heatmap_down = heatmap_down[cut_pad_down//int(hp_down_factor):cut_pad_up//int(hp_down_factor)]
            return heatmap_down
        elif get_full and not get_down:
            heatmap = heatmap[cut_pad_down:cut_pad_up]
            return heatmap
        else:
            heatmap_down = heatmap_down[cut_pad_down//int(hp_down_factor):cut_pad_up//int(hp_down_factor)] 
            heatmap = heatmap[cut_pad_down:cut_pad_up]
            return heatmap_down, heatmap
            

    if get_down and not get_full:
        return heatmap_down
    elif get_full and not get_down:
        return heatmap
    else:
        return heatmap_down, heatmap



# for an array alternatively use zscore(..., axis=None)
norm_zs = lambda v: (v-v.mean())/v.std() ## z-score function


def compute_nss(sal_norm, gaze_xy, nan_ratio=1.0, penalize_nan=False):
    # provide normalized saliency map. 
    # or normalize saliency map here. 
    # sal_norm = (sal_map - np.mean(sal_map)) / np.std(sal_map)      
    # frame_shape = [frame_height, frame_width] : rows, cols
    assert gaze_xy.ndim == 2
    # otherwise locs = locs[np.newaxis]
    
    assert gaze_xy.shape[1] == 2

    if np.isnan(gaze_xy).any(axis=1).sum() >= nan_ratio*gaze_xy.shape[0]:
        return np.NaN
    
    gaze_xy = gaze_xy.round()

    temp = []
    for cv, rv in gaze_xy:
        if np.isnan(cv) or np.isnan(rv):
            if penalize_nan:
                temp.append(0)
            else:
                continue
        else:
            # correct for rounding above. 
            if rv >= sal_norm.shape[0] : rv = sal_norm.shape[0] - 1
            if cv >= sal_norm.shape[1] : cv = sal_norm.shape[1] - 1
            temp.append(sal_norm[int(rv),int(cv)])
    
    return np.mean(temp) # yields the ratio. 



def compute_ioc(heat_bin, gaze_xy, nan_ratio=1.0, penalize_nan=False):

    gaze_xy = np.array(gaze_xy, copy=True)
    
    # or change temp.append(False) below.
    assert heat_bin.dtype == bool
    
    if gaze_xy.ndim == 1:
        gaze_xy = gaze_xy[np.newaxis]

    assert gaze_xy.shape[1] == 2

    if np.isnan(gaze_xy).any(axis=1).sum() >= nan_ratio*gaze_xy.shape[0]:
        return np.NaN

    gaze_xy = gaze_xy.round()
    
    temp = []
    for cv, rv in gaze_xy:
        if np.isnan(cv) or np.isnan(rv):
            if penalize_nan:
                temp.append(False)
            else:
                continue
        else:
            # correct for rounding above. 
            if rv >= heat_bin.shape[0] : rv = heat_bin.shape[0] - 1
            if cv >= heat_bin.shape[1] : cv = heat_bin.shape[1] - 1
            temp.append(heat_bin[int(rv),int(cv)])
    
    return np.mean(temp) # yields the ratio. 




#%% Face part effect size utils
def get_subj_vals(face_gaze_g1, eye_gaze_g1, mouth_gaze_g1, ons_g1_tot=None,scale2facetime=False, taxis=1):
    # taxis: time or frames axis
    
    if ons_g1_tot is None:
        ons_g1_tot_in = np.sum(face_gaze_g1 != -1, taxis)
    else:
        ons_g1_tot_in = ons_g1_tot.copy()
        
    ons_g1_tot_in[ons_g1_tot_in==0] = 1 # against invalid division below. It is 0/1 anyway not alters results. 

    face_looking_g1 = np.sum(face_gaze_g1==1, taxis)
    eye_looking_g1_reg = np.sum(eye_gaze_g1==1, taxis)
    mouth_looking_g1_reg = np.sum(mouth_gaze_g1==1, taxis)

    facelooking_val = face_looking_g1/ons_g1_tot_in

    if scale2facetime:
        face_looking_g1[face_looking_g1==0] = 1 # against invalid division below. # do if after assessing d_facelooking.
        eyelooking_val = eye_looking_g1_reg/face_looking_g1
        mouthlooking_val = mouth_looking_g1_reg/face_looking_g1

    else:
        eyelooking_val = eye_looking_g1_reg/ons_g1_tot_in
        mouthlooking_val = mouth_looking_g1_reg/ons_g1_tot_in
    
    return facelooking_val, eyelooking_val, mouthlooking_val



def get_cohend_vals(face_gaze_g1, face_gaze_g2, eye_gaze_g1,eye_gaze_g2, mouth_gaze_g1, mouth_gaze_g2, \
                    ons_g1_tot=None, ons_g2_tot=None, scale2facetime=False, taxis=1):
    
    # Cohen's d for g1-g2.
    if ons_g1_tot is None:
        ons_g1_tot_in = np.sum(face_gaze_g1 != -1, taxis)
    else:
        ons_g1_tot_in = ons_g1_tot.copy()
    if ons_g1_tot is None:
        ons_g2_tot_in = np.sum(face_gaze_g2 != -1, taxis)
    else:
        ons_g2_tot_in = ons_g2_tot.copy()    
    
    ons_g1_tot_in[ons_g1_tot_in==0] = 1 # against invalid division below. It is 0/1 anyway not alters results. 
    ons_g2_tot_in[ons_g2_tot_in==0] = 1 # against invalid division below. 
    
    face_looking_g1 = np.sum(face_gaze_g1==1, taxis)
    eye_looking_g1_reg = np.sum(eye_gaze_g1==1, taxis)
    mouth_looking_g1_reg = np.sum(mouth_gaze_g1==1, taxis)

    face_looking_g2 = np.sum(face_gaze_g2==1, taxis)
    eye_looking_g2_reg = np.sum(eye_gaze_g2==1, taxis)
    mouth_looking_g2_reg = np.sum(mouth_gaze_g2==1, taxis)

    d_facelooking = cohen_d_ci(face_looking_g1/ons_g1_tot_in,face_looking_g2/ons_g2_tot_in,rm_extreme=False)

    if scale2facetime:
        face_looking_g1[face_looking_g1==0] = 1 # against invalid division below. # do it after assessing d_facelooking.
        face_looking_g2[face_looking_g2==0] = 1 # against invalid division below. 
        
        # d' given that they looked at a face
        d_eyelooking = cohen_d_ci(eye_looking_g1_reg/face_looking_g1,eye_looking_g2_reg/face_looking_g2,rm_extreme=False)
        d_mouthlooking = cohen_d_ci(mouth_looking_g1_reg/face_looking_g1,mouth_looking_g2_reg/face_looking_g2,rm_extreme=False)
    else:
        # d' given that they looked at a face
        d_eyelooking = cohen_d_ci(eye_looking_g1_reg/ons_g1_tot_in,eye_looking_g2_reg/ons_g2_tot_in,rm_extreme=False)
        d_mouthlooking = cohen_d_ci(mouth_looking_g1_reg/ons_g1_tot_in,mouth_looking_g2_reg/ons_g2_tot_in,rm_extreme=False)
        
    return d_facelooking, d_eyelooking, d_mouthlooking





#%% Load gaze features data
def load_gazedata_episodes(gazedata_dir, behavioraldata_file=None):
    '''
    Usage:
        gazedata_dir = '/home/umit/Documents/Research_ET/AutismVids/PaperResults_v1/Cohend_pool'

        vidnames, asd_data_df, td_data_df = load_gazedata_episodes(gazedata_dir)
    '''

    vidclips = [ 'Ep1', 'Ep4_AQNR', 'AllVids' ]
    vidname_txt = { 'Ep1':'vid1', 'Ep4_AQNR':'vid2', 'AllVids':'comb' }
    measure_class = [ 'bodypart', 'facepart', 'heatmap' ]
    bpart_measures = [ 'OnscreenData', 'HandsData', 'NonheadData', 'NonsocialData' ] 
    face_measures = [ 'FaceData', 'EyeData', 'MouthData' ]
    heatmap_measures = [ 'GazeCorr']

    asd_subjs = None
    td_subjs = None
    
    colnames = []
    asd_data_vals = []
    td_data_vals = []
    for vid_ii in vidclips:
    
        for mclass_ii in measure_class:
            
            if mclass_ii == 'bodypart':
                load_measures = bpart_measures
                subj_txt = 'BodyParts'
            elif mclass_ii == 'facepart':
                load_measures = face_measures
                subj_txt = 'FaceData'
            elif mclass_ii == 'heatmap':
                load_measures = heatmap_measures
                subj_txt = 'GazeCorr'
            else:
                raise SystemExit('Unidentified class!')
    
            # --- check whether subj IDs match across different variable types ---
            if asd_subjs is None:
                asd_subjs = np.load(os.path.join(gazedata_dir,f'{subj_txt}_asd_subjs_{vid_ii}.npy'))
            else:
                asd_subjs_this = np.load(os.path.join(gazedata_dir,f'{subj_txt}_asd_subjs_{vid_ii}.npy'))
                assert np.array_equal(asd_subjs, asd_subjs_this)
                
            if td_subjs is None:
                td_subjs = np.load(os.path.join(gazedata_dir,f'{subj_txt}_td_subjs_{vid_ii}.npy'))
            else:
                td_subjs_this = np.load(os.path.join(gazedata_dir,f'{subj_txt}_td_subjs_{vid_ii}.npy'))
                assert np.array_equal(td_subjs, td_subjs_this)
    
            for meas_ii in load_measures:
                label_txt = f"{meas_ii.replace('Data','')}_{vidname_txt[vid_ii]}"
                label_txt = label_txt.replace('GazeCorr','Heatcorr')
                
                asd_feat = np.load(os.path.join(gazedata_dir,f'{meas_ii}_asd_{vid_ii}.npy'))
                td_feat = np.load(os.path.join(gazedata_dir,f'{meas_ii}_td_{vid_ii}.npy'))
    
                colnames.append(label_txt)
                asd_data_vals.append(asd_feat)
                td_data_vals.append(td_feat)
    
    
                if os.path.isfile(os.path.join(gazedata_dir,f'{meas_ii}_asd_{vid_ii}_asdref.npy')):
                        
                    asd_feat_asdref = np.load(os.path.join(gazedata_dir,f'{meas_ii}_asd_{vid_ii}_asdref.npy'))
                    td_feat_asdref = np.load(os.path.join(gazedata_dir,f'{meas_ii}_td_{vid_ii}_asdref.npy'))
        
                    colnames.append(label_txt+'_asdref')
                    asd_data_vals.append(asd_feat_asdref)
                    td_data_vals.append(td_feat_asdref)

    
    
    # ----- Load behavioral assessment data -----
    if behavioraldata_file is not None:
        
        bhvrldata_df = pd.read_csv(behavioraldata_file,index_col='ID')
        asd_bdata_df = bhvrldata_df.loc[bhvrldata_df.index.isin(asd_subjs)]
        td_bdata_df = bhvrldata_df.loc[bhvrldata_df.index.isin(td_subjs)]
        
        assert np.array_equal(asd_subjs,asd_bdata_df.index)
        assert np.array_equal(td_subjs,td_bdata_df.index)


    # ----- prepare return variables -----
    asd_data_vals = np.asarray(asd_data_vals).T
    td_data_vals = np.asarray(td_data_vals).T
    
    asd_gazedata_df = pd.DataFrame(data=asd_data_vals, columns=colnames, index=asd_subjs)
    td_gazedata_df = pd.DataFrame(data=td_data_vals, columns=colnames, index=td_subjs)

    if behavioraldata_file is not None:
        return vidname_txt.values(), asd_gazedata_df, td_gazedata_df, asd_bdata_df, td_bdata_df
    else:
        return vidname_txt.values(), asd_gazedata_df, td_gazedata_df
        



#%% Load data to resample video epochs and subjects
def load_gazedata_split(input_dir, tbin, corr_timebin=1, n_samples=10000, get_pair_samples=False, step=5):
    '''
    Parameters
    ----------
    input_dir : str
        Directory from raw split ET data.
    tbin : int
        timebin for output split data in seconds.
    corr_timebin : int or float, optional
        timebin used while generating heatmap correlations. 
        The default is 1.
    n_samples : int
        the number of samples to be used in resampling analysis.
    get_pair_samples : boolean
        if True, then returning nonoverlaping paired timebin sampling indicies, (used for correlations)
        else, single indicies (used for d-vals).  

    '''

    print('loading resampled data')
    scale2facetime = False
    
    # ----- load heatmap corrs data splits to combine with face and bodypart gaze data -----
    frames_split_list = np.load(os.path.join(input_dir, 'frames_split_list_AllVids.npy'))
    n_coorbins = frames_split_list.shape[0]
    allinds = np.arange(n_coorbins)
    chunklen = tbin // corr_timebin
    
    # # --- static splits ---
    # n_chunks = n_coorbins // chunklen
    # chunk_inds = np.array_split(allinds,n_chunks)
    # chunks_frames = [ [ frames_split_list[cii[0],0], frames_split_list[cii[-1],1] ] for cii in chunk_inds ] 
    
    # --- use moving window for random sampling of time bins ---
    chunk_inds = [ allinds[idx:idx+chunklen] for idx in range(0,n_coorbins-chunklen+1,step) ]
    n_chunks = len(chunk_inds)
    chunks_frames = [ [ frames_split_list[cii[0],0], frames_split_list[cii[-1],1] ] for cii in chunk_inds ] 
    chunks_frames_full = [ np.arange(fii[0],fii[1]) for fii in chunks_frames ]
    # app_time = chunks_frames_full[0].size * 41 / 1000 # 
    print(f'nchunks: {n_chunks}')
    
    
    # ----- Loading heatmap correlation data -----
    print('Loading heatmap correlation data....')
    
    asd_subjs = np.load(os.path.join(input_dir,'GazeCorr_asd_subjs_Ep1.npy'))
    td_subjs = np.load(os.path.join(input_dir,'GazeCorr_td_subjs_Ep4_AQNR.npy'))
    
    asd_corrs_comb = np.load(os.path.join(input_dir,'GazeCorr_asd_split_AllVids.npy')).T
    td_corrs_comb = np.load(os.path.join(input_dir,'GazeCorr_td_split_AllVids.npy')).T
    
    corrs_asd_split = np.zeros((n_chunks,len(asd_subjs)))
    corrs_td_split = np.zeros((n_chunks,len(td_subjs)))
    for cii,chunk_ii in enumerate(chunk_inds):
        corrs_asd_split[cii] = np.nanmean(asd_corrs_comb[chunk_ii,:],0)
        corrs_td_split[cii] = np.nanmean(td_corrs_comb[chunk_ii,:],0)
    
    # cleaning 
    del asd_corrs_comb, td_corrs_comb
    
    if np.isnan(corrs_asd_split).any():
        corrs_asd_split = np.nan_to_num(corrs_asd_split)
        # corrs_asd_split = interp_nans(corrs_asd_split,axis=0)
    if np.isnan(corrs_td_split).any():
        corrs_td_split = np.nan_to_num(corrs_td_split)
        # corrs_td_split = interp_nans(corrs_td_split,axis=0)


    # ----- loading face gaze time data ----- 
    print('Loading face gaze time data....')
    
    frames_num_of_faces = np.load(os.path.join(input_dir, 'frames_num_of_faces_AllVids.npy'))
    
    facedata_asd_subjs = np.load(os.path.join(input_dir, 'FaceData_asd_subjs_AllVids.npy'))
    facedata_td_subjs = np.load(os.path.join(input_dir, 'FaceData_td_subjs_AllVids.npy'))
    assert np.array_equal(asd_subjs, facedata_asd_subjs)
    assert np.array_equal(td_subjs, facedata_td_subjs)
    
    facegaze_asd = np.load(os.path.join(input_dir, 'Facegaze_asd_split_AllVids.npy'))
    facegaze_td = np.load(os.path.join(input_dir, 'Facegaze_td_split_AllVids.npy'))
    assert frames_num_of_faces.size == facegaze_asd.shape[1] 
    assert frames_num_of_faces.size == facegaze_td.shape[1] 

    eyegaze_asd = np.load(os.path.join(input_dir, 'Eyegaze_asd_split_AllVids.npy'))
    eyegaze_td = np.load(os.path.join(input_dir, 'Eyegaze_td_split_AllVids.npy'))
    
    mouthgaze_asd = np.load(os.path.join(input_dir, 'Mouthgaze_asd_split_AllVids.npy'))
    mouthgaze_td = np.load(os.path.join(input_dir, 'Mouthgaze_td_split_AllVids.npy'))
    
    onscreen_fd_asd_split = np.zeros((n_chunks,len(asd_subjs)))
    onscreen_fd_td_split = np.zeros((n_chunks,len(td_subjs)))
    
    facedata_asd_split = np.zeros((n_chunks,len(asd_subjs)))
    facedata_td_split = np.zeros((n_chunks,len(td_subjs)))
    
    eyedata_asd_split = np.zeros((n_chunks,len(asd_subjs)))
    eyedata_td_split = np.zeros((n_chunks,len(td_subjs)))
    
    mouthdata_asd_split = np.zeros((n_chunks,len(asd_subjs)))
    mouthdata_td_split = np.zeros((n_chunks,len(td_subjs)))
    
    use_split = np.ones(n_chunks,dtype=bool)
    for fii,frames_ii_indx in enumerate(chunks_frames):
        
        frames_ii = np.arange(frames_ii_indx[0],frames_ii_indx[1])
    
        frames_use = np.zeros(frames_num_of_faces.size,dtype=bool)
        frames_use[ frames_ii ] = True
    
        if np.sum(frames_num_of_faces[frames_use])<100:
            use_split[fii] = False
            continue
        
        # -1 not onscreen; 0 onscreen nonface; 1 onscreen face. 
        onscreen_fd_asd_sum = np.sum(facegaze_asd[:,frames_ii] != -1, 1)
        onscreen_fd_td_sum = np.sum(facegaze_td[:,frames_ii] != -1, 1)
        
        onscreen_fd_asd_split[fii] = onscreen_fd_asd_sum / facegaze_asd[:,frames_ii].shape[1]
        onscreen_fd_td_split[fii] = onscreen_fd_td_sum / facegaze_td[:,frames_ii].shape[1]
        
        facedata_asd_split[fii], eyedata_asd_split[fii], mouthdata_asd_split[fii] = \
            get_subj_vals(facegaze_asd[:,frames_ii],eyegaze_asd[:,frames_ii], 
                          mouthgaze_asd[:,frames_ii],scale2facetime=scale2facetime, taxis=1)
        
        facedata_td_split[fii], eyedata_td_split[fii], mouthdata_td_split[fii]  = \
            get_subj_vals(facegaze_td[:,frames_ii], eyegaze_td[:,frames_ii],
                          mouthgaze_td[:,frames_ii],scale2facetime=scale2facetime, taxis=1)

    # cleaning 
    del facegaze_asd, facegaze_td
    
    
    # ----- loading bodypart gaze time data -----
    print('Loading bodypart gaze time data....')

    bodyparts_asd_subjs = np.load(os.path.join(input_dir, 'BodyParts_asd_subjs_AllVids.npy'))
    bodyparts_td_subjs = np.load(os.path.join(input_dir, 'BodyParts_td_subjs_AllVids.npy'))
    
    assert np.array_equal(asd_subjs, bodyparts_asd_subjs)
    assert np.array_equal(td_subjs, bodyparts_td_subjs)
    
    onscreen_asd = np.load(os.path.join(input_dir, 'Onscreen_asd_split_AllVids.npy'))
    onscreen_td = np.load(os.path.join(input_dir, 'Onscreen_td_split_AllVids.npy'))
    
    head_asd = np.load(os.path.join(input_dir, 'Head_asd_split_AllVids.npy'))
    head_td = np.load(os.path.join(input_dir, 'Head_td_split_AllVids.npy'))
    
    nonhead_asd = np.load(os.path.join(input_dir, 'Nonhead_asd_split_AllVids.npy'))
    nonhead_td = np.load(os.path.join(input_dir, 'Nonhead_td_split_AllVids.npy'))
    
    hands_asd = np.load(os.path.join(input_dir, 'Hands_asd_split_AllVids.npy'))
    hands_td = np.load(os.path.join(input_dir, 'Hands_td_split_AllVids.npy'))
        
    nonbody_asd = np.load(os.path.join(input_dir, 'Nonsocial_asd_split_AllVids.npy'))
    nonbody_td = np.load(os.path.join(input_dir, 'Nonsocial_td_split_AllVids.npy'))
    
    onscreen_asd_split = np.zeros((n_chunks,len(asd_subjs)))
    head_asd_split = np.zeros((n_chunks,len(asd_subjs)))
    hands_asd_split = np.zeros((n_chunks,len(asd_subjs)))
    nonheadbody_asd_split = np.zeros((n_chunks,len(asd_subjs)))
    nonbody_asd_split = np.zeros((n_chunks,len(asd_subjs)))
    
    onscreen_td_split = np.zeros((n_chunks,len(td_subjs)))
    head_td_split = np.zeros((n_chunks,len(td_subjs)))
    hands_td_split = np.zeros((n_chunks,len(td_subjs)))
    nonheadbody_td_split = np.zeros((n_chunks,len(td_subjs)))
    nonbody_td_split = np.zeros((n_chunks,len(td_subjs)))
    
    for fii,frames_ii_indx in enumerate(chunks_frames):
        
        frames_ii = np.arange(frames_ii_indx[0],frames_ii_indx[1])
        
        aio1_fix_time_asd = []
        aio1_fix_time_keep_asd = []
        for oii in onscreen_asd[:,frames_ii]:
            aio1_fix_time_asd.append(np.sum(oii)/len(oii))
            aio1_fix_time_keep_asd.append(np.sum(oii))
        
        aio1_fix_time_keep_asd = np.asarray(aio1_fix_time_keep_asd)
        aio1_fix_time_keep_asd[aio1_fix_time_keep_asd==0] = 1 # against invalid division below. 
        
        onscreen_asd_split[fii] = np.asarray(aio1_fix_time_asd)
        head_asd_split[fii] = np.asarray([ np.sum(oii)/aio1_fix_time_keep_asd[cii] for cii,oii in enumerate(head_asd[:,frames_ii]) ])
        nonheadbody_asd_split[fii] = np.asarray([ np.sum(oii)/aio1_fix_time_keep_asd[cii] for cii,oii in enumerate(nonhead_asd[:,frames_ii]) ])
        hands_asd_split[fii] = np.asarray([ np.sum(oii)/aio1_fix_time_keep_asd[cii] for cii,oii in enumerate(hands_asd[:,frames_ii]) ])
        nonbody_asd_split[fii] = np.asarray([ np.sum(oii)/aio1_fix_time_keep_asd[cii] for cii,oii in enumerate(nonbody_asd[:,frames_ii]) ])
        
        
        aio1_fix_time_td = []
        aio1_fix_time_keep_td = []
        for oii in onscreen_td[:,frames_ii]:
            aio1_fix_time_td.append(np.sum(oii)/len(oii))
            aio1_fix_time_keep_td.append(np.sum(oii))
        
        aio1_fix_time_keep_td = np.asarray(aio1_fix_time_keep_td)
        aio1_fix_time_keep_td[aio1_fix_time_keep_td==0] = 1 # against invalid division below. 
        
        onscreen_td_split[fii] = np.asarray(aio1_fix_time_td)
        head_td_split[fii] = np.asarray([ np.sum(oii)/aio1_fix_time_keep_td[cii] for cii,oii in enumerate(head_td[:,frames_ii]) ])
        nonheadbody_td_split[fii] = np.asarray([ np.sum(oii)/aio1_fix_time_keep_td[cii] for cii,oii in enumerate(nonhead_td[:,frames_ii]) ])
        hands_td_split[fii] = np.asarray([ np.sum(oii)/aio1_fix_time_keep_td[cii] for cii,oii in enumerate(hands_td[:,frames_ii]) ])
        nonbody_td_split[fii] = np.asarray([ np.sum(oii)/aio1_fix_time_keep_td[cii] for cii,oii in enumerate(nonbody_td[:,frames_ii]) ])
        
    
    # control between face and bodyparts data.
    assert np.allclose(onscreen_asd_split,onscreen_fd_asd_split) and np.allclose(onscreen_td_split,onscreen_fd_td_split)
    

    # ----- selecting sampling indicies -----
    print('Data were loadded!')
    
    if n_samples is None:
        sample_inds = None 
    else:
        empty_tbins = np.argwhere(use_split != True).squeeze()
        
        print('Selecting sampling indicies...\n')
        sample_inds = []
        
        if get_pair_samples:
            sample_inds = []
            while len(sample_inds) < n_samples:
                sample_inds_this = np.sort(np.random.choice(n_chunks,2,replace=False)).tolist()
                if not np.isin(sample_inds_this,empty_tbins).any() and \
                not np.isin(chunks_frames_full[sample_inds_this[0]],chunks_frames_full[sample_inds_this[1]]).any():
                    sample_inds.append(sample_inds_this)
        
        else:
            while len(sample_inds) < n_samples:
                sample_inds_this = np.random.randint(n_chunks)
                if not np.isin(sample_inds_this,empty_tbins).any():
                    sample_inds.append(sample_inds_this)
        
        
    # ----- return split data and feature sets -----
    featnames = ['Onscreen', 'Face', 'Eye', 'Mouth', 'Hands', 
                 'Nonhead', 'Nonsocial', 'Heatcorr']
    
    asd_feats = np.stack((onscreen_asd_split,facedata_asd_split,eyedata_asd_split,
                          mouthdata_asd_split,hands_asd_split,
                          nonheadbody_asd_split,nonbody_asd_split,
                          corrs_asd_split),axis=-1)
    
    td_feats = np.stack((onscreen_td_split,facedata_td_split,eyedata_td_split,
                         mouthdata_td_split,hands_td_split,
                         nonheadbody_td_split,nonbody_td_split,
                         corrs_td_split),axis=-1)

    return asd_subjs, td_subjs, asd_feats, td_feats, featnames, sample_inds #, chunks_frames
    
