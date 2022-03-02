#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some helper functions to load raw ET data collected in the autism video project.

"""

import numpy as np
from math import atan
import pandas as pd

from sklearn.metrics.pairwise import paired_distances

from alabeye.io.loaddata import loadmat
from alabeye.stats import nanmasked_mean


def get_avp_rawdata(ETdata_file,events_file,meta_file,calib_file=None,scale2frame=True,audioQNR=False,**kwargs):
    
    ETdata_pd = pd.read_csv(ETdata_file, sep="\t", header=None)
    events = [ line.strip().replace('\t',' ') for line in open(events_file).readlines() ]
    
    # with open(events_file) as f: events = [ line.strip().replace('\t',' ') for line in f]
    
    # pause_times = [0, 29., 60., 92.8, 179.7, 251.8, 320., 339., 366., 489.5, 560., 608.5, 618., 649.,
                   # 705.5, 735.5, 740., 772.5, 857., 901., 947.1, 975., 1023.8, 1073., 1077.2, 1163., 1205., 1229., 1255.]
    
    # validation_L = []
    # validation_R = []
    
    if audioQNR:
        question_on_info = []
        question_off_info = []
    
    movie_time_actual = []
    video_start_time_count = []
    video_stop_time_count = []

    question_off_pre = False
    for event_ii in events:

        question_off_this = False

        if 'start_movie' in event_ii: 
            # start_movie = float(event_ii.split()[1])
            # start_movie_altv = float(event_ii.split()[2])
            question_off_pre = True
            question_off_this = True
        
        if 'stop_movie' in event_ii: 
            stop_movie = float(event_ii.split()[1]) # keep for [ if not 'stop_movie' in vars() ] below.
            # stop_movie_altv = float(event_ii.split()[2])
            video_stop_time_count.append([ float(movie_time_actual[-1][0]), float(movie_time_actual[-1][1]), float(movie_time_actual[-1][-1]) ]) 

        if 'movie_time_actual' in event_ii: movie_time_actual.append(event_ii.split()[1:])
    
        if audioQNR:
            if 'question_on' in event_ii: 
                question_on_info.append( event_ii.split()[1:3] + [ event_ii.split()[5] ] )
                video_stop_time_count.append([ float(movie_time_actual[-1][0]), float(movie_time_actual[-1][1]), float(movie_time_actual[-1][-1]) ]) 
            
            if 'question_off' in event_ii: 
                question_off_info.append( event_ii.split()[1:3] + [ event_ii.split()[5] ] )
                question_off_pre = True
                question_off_this = True


        if question_off_pre and not question_off_this:
            if 'movie_time_actual' in event_ii:
                question_off_pre = False
                video_start_time_count.append([ float(event_ii.split()[1]), float(event_ii.split()[2]), float(event_ii.split()[6]) ] )
            elif not 'movie_time' in event_ii:
                raise SystemError('Unexpected case! Check this manually!')
                    

        
    if not 'stop_movie' in vars():
        raise ValueError('stop_movie is not available in events file!')


    video_start_time_count = np.asarray(video_start_time_count)
    video_stop_time_count = np.asarray(video_stop_time_count)
    
    movie_time_start_actual = video_start_time_count[0,0]
    movie_time_start_actual_altv = video_start_time_count[0,2]
    movie_time_start_static = video_start_time_count[0,1]
    
    movie_time_stop_actual_altv = video_stop_time_count[-1,2]
    movie_time_stop_static = video_stop_time_count[-1,1]
    
    movie_length_altv = movie_time_stop_actual_altv - movie_time_start_actual_altv # from events file.   
    movie_length_static = movie_time_stop_static - movie_time_start_static # from events file. 
    # correct time jumps.
    movie_time_stop_actual = movie_length_altv + movie_time_start_actual 

    movie_time_actual = np.asarray(movie_time_actual)
    movie_time_frame_mapping = movie_time_actual[:,[0,3]].astype(float)
    # movie_time_frame_mapping_altv = movie_time_actual[:,[5,3]].astype(float)
    # movie_time_frame_mapping_altv2 = movie_time_actual[:,[1,3]].astype(float)

    # control for frame numbers: 
    if not np.all(np.diff(movie_time_frame_mapping[:,1])):
        raise ValueError('Some frames are skipped in the EVENTS file!')


    if audioQNR:
        question_on_info = np.asarray(question_on_info).astype(float)
        question_off_info = np.asarray(question_off_info).astype(float)
        
        assert question_on_info.shape[0] == question_off_info.shape[0] 
        assert np.all(question_on_info[:,2] == question_off_info[:,2])
        
        question_timeblocks = np.column_stack((question_on_info[:,1],question_off_info[:,1],question_on_info[:,2]))
        # map to ET_rec_time
        question_timeblocks[:,[0,1]] = question_timeblocks[:,[0,1]] - movie_time_start_actual_altv + movie_time_start_actual

        video_timeblocks = np.column_stack((video_start_time_count[:,2],video_stop_time_count[:,2])) # using Gettime info. 
        # map to ET_rec_time
        video_timeblocks = video_timeblocks - movie_time_start_actual_altv + movie_time_start_actual # corrected time.
        assert np.all(np.diff(question_timeblocks,axis=0) > 0 )

        video_timeblocks_static = np.column_stack((video_start_time_count[:,1],video_stop_time_count[:,1]))
        # use to correct ET_rec_time
        video_timeblocks_static_altered = video_timeblocks_static - movie_time_start_static

    
    # According to talk2tobii('SAVE_DATA', eye_trackin_data, events, 'APPENDorTRUNK') function.
    # columns in ETdata are:
    # time in sec
    # time in msec
    # x gaze coordinate of the left eye
    # y gaze coordinate of the left eye
    # x gaze coordinate of the right eye
    # y gaze coordinate of the right eye
    # left camera eye position - x coordinate
    # left camera eye position - y coordinate
    # right camera eye position - x coordinate
    # right camera eye position - y coordinate
    # left eye validity
    # right eye validity
    # diameter of pupil of the left eye
    # diameter of pupil of the right eye
    # distance of the camera from the left eye
    # distance of the camera from the right eye
    # recording time (an additional entry in our data)
    
    # about validity values:
    #               (Validity indicates how likely is it that the eye is found)
    #               0 - Certainly (>99%),
    #               1 - Probably (80%),
    #               2 - (50%),
    #               3 - Likely not (20%),
    #               4 - Certainly not (0%)
    
    
    colnames = ['t_sec', 't_msec', 'left_gaze_x', 'left_gaze_y', 'right_gaze_x', 'right_gaze_y', 
                'left_camera_eye_x', 'left_camera_eye_y', 'right_camera_eye_x', 'right_camera_eye_y',
                'left_eye_validity', 'right_eye_validity', 'diameter_pupil_left', 'diameter_pupil_right', 
                'distance_eye2camera_left', 'distance_eye2camera_right', 'rec_time' ]
    
    # if the number of columns in ETdata is larger than the number of colnames provided by talk2tobii manual, we extend the colnames with dummy column names. 
    if ETdata_pd.shape[1] > len(colnames): colnames += ['dum_col_%s'%dii for dii in range(ETdata_pd.shape[1]-len(colnames)) ] 
    ETdata_pd.columns = colnames
    
    
    # Find on screen gaze points, we also allow data points that are not exactly on screen 
    # but within some padding area (padding ratio).
    # [we need to play with this ratio to see its effect] 
    # Let's make this part generic in case we have data with screen pixel values. 
    frame_width = 1. # columns
    frame_height = 1.  # rows
    
    padding_ratio = 0.0 # keep gaze if out of the screen gaze is close to the screen boundaries.
    width_cut_lower, width_cut_upper = -frame_width*padding_ratio, frame_width*(1.+padding_ratio) # conservative [ 0, frame_width]
    height_cut_lower, height_cut_upper = -frame_height*padding_ratio, frame_height*(1.+padding_ratio) # conservative [ 0, frame_height]
    
    
    # ----- Resolve out of screen fixation and gaze points. Make them np.NaN -----
    fix_and_gaze_cols_width  = ['left_gaze_x', 'right_gaze_x']
    fix_and_gaze_cols_height = ['left_gaze_y', 'right_gaze_y']
    
    for fiin, fiix in enumerate(fix_and_gaze_cols_width):
    
        # Option #1: remove out of screen data points.  
        # problem_inds = np.logical_not(ETdata_pd[fiix].astype(float).between(width_cut_lower, width_cut_upper))
    
        # Option #2: 
        # to include validity condition as well, use this version. (see above note about validity values)
        outof_screen_TF = ETdata_pd[fiix].astype(float).between(width_cut_lower, width_cut_upper)
        if 'left' in fiix: validty_TF = ETdata_pd['left_eye_validity'].astype(float).lt(2) # Dan suggested taking only 0 and 1.
        elif 'right' in fiix: validty_TF = ETdata_pd['right_eye_validity'].astype(float).lt(2) 
        problem_inds = np.logical_not(np.logical_and(outof_screen_TF,validty_TF))
        
        ETdata_pd.loc[problem_inds,fiix] = np.NaN
        # if X component is np.NaN then make Y component also np.NaN.
        ETdata_pd.loc[problem_inds,fix_and_gaze_cols_height[fiin]] = np.NaN
            
    
    for fiin, fiiy in enumerate(fix_and_gaze_cols_height):
        
        # Option #1: remove out of screen data points.  
        # problem_inds = np.logical_not(ETdata_pd[fiiy].astype(float).between(height_cut_lower, height_cut_upper))

        # Option #2:        
        # to include validity condition as well, use this version. 
        outof_screen_TF = ETdata_pd[fiiy].astype(float).between(height_cut_lower, height_cut_upper)
        if 'left' in fiix: validty_TF = ETdata_pd['left_eye_validity'].astype(float).lt(2) # Dan suggested taking only 0 and 1.
        elif 'right' in fiix: validty_TF = ETdata_pd['right_eye_validity'].astype(float).lt(2) 
        problem_inds = np.logical_not(np.logical_and(outof_screen_TF,validty_TF))
        
        ETdata_pd.loc[problem_inds,fiiy] = np.NaN
        # if Y component is np.NaN then make X component also np.NaN.
        ETdata_pd.loc[problem_inds,fix_and_gaze_cols_width[fiin]] = np.NaN
    
    
    # ========================================================================================================
    #
    # Detection of fixations and saccades could be done here, before averaging the left and right eye data!  
    #
    # ========================================================================================================
    
    
    if calib_file is not None:
        calib_info = loadmat(calib_file)
        quality_data = calib_info.get('quality')
        if isinstance(quality_data,int):
            # print('int for %s for %s'%(subj,session_video) )
            # print(quality_data)
            keep_calib_qual = [np.NaN,np.NaN,np.NaN,np.NaN] 
            left_q_ratio = 0.0 
            right_q_ratio= 0.0
        elif len(quality_data)==0:
            # print('empty for %s for %s'%(subj,session_video) )
            keep_calib_qual = [np.NaN,np.NaN,np.NaN,np.NaN]
            left_q_ratio = 0.0   
            right_q_ratio= 0.0
        else:    
            left_q  = quality_data[:,4]
            right_q = quality_data[:,7]
            
            left_q_mask = left_q == 1
            right_q_mask = right_q == 1
            
            left_q_ratio = np.sum(left_q_mask) / len(left_q)
            right_q_ratio = np.sum(right_q_mask) / len(right_q)
            
            # --- such an implementation would be better ---
            # calib_points = quality_data[:,[0,1]].round(1)
            # _,calib_points_inds = np.unique(calib_points,axis=0,return_inverse=True)
            # calib_data_split = np.array_split(quality_data, np.where(np.diff(calib_points_inds)!=0)[0]+1)            
            # for c_kk in calib_data_split:
                # compute paired distances for each calibration point.  
            
            leftValidity_dist  = paired_distances(quality_data[left_q_mask][:,[0,1]], quality_data[left_q_mask][:,[2,3]], metric='euclidean')
            rightValidity_dist = paired_distances(quality_data[right_q_mask][:,[0,1]], quality_data[right_q_mask][:,[5,6]], metric='euclidean')
        
            keep_calib_qual = [np.mean(leftValidity_dist),np.mean(rightValidity_dist), left_q_ratio,right_q_ratio ]
   
    else:
        keep_calib_qual = [np.NaN,np.NaN,np.NaN,np.NaN] 
        left_q_ratio = 0.0 
        right_q_ratio= 0.0
        
    
    # Determine gaze_x and gaze_y based on calibration quality. 
    if (calib_file is not None) and left_q_ratio > 0.8 and right_q_ratio <= 0.8:
        print('using left eye data...')
        gaze_x = ETdata_pd[ 'left_gaze_x' ].values
        gaze_y = ETdata_pd[ 'left_gaze_y' ].values
        pupil_diameter_avg = ETdata_pd [ 'diameter_pupil_left' ].values
    elif (calib_file is not None) and left_q_ratio <= 0.8 and right_q_ratio > 0.8:
        print('using right eye data...')
        gaze_x = ETdata_pd[ 'right_gaze_x' ].values
        gaze_y = ETdata_pd[ 'right_gaze_y' ].values
        pupil_diameter_avg = ETdata_pd [ 'diameter_pupil_right' ].values
    else:
        # ----- Average left and right eye gaze_x to obtain a single gaze_x value (the same for the gaze_y) -----
        gaze_x = nanmasked_mean(ETdata_pd[ fix_and_gaze_cols_width ].values, axis=1)
        gaze_y = nanmasked_mean(ETdata_pd[ fix_and_gaze_cols_height ].values, axis=1)
        
        pupil_diameter_LR = ETdata_pd [ ['diameter_pupil_left', 'diameter_pupil_right'] ].values
        pupil_diameter_LR[pupil_diameter_LR<0] = np.NaN
        pupil_diameter_avg = nanmasked_mean(pupil_diameter_LR,axis=1)
    
    
    # ----- map out of screen gaze (we might have these due to allowing them with padding_ratio above) onto boundaries -----
    nnan_mask = ~np.isnan(gaze_x)
    gaze_x_nnan = gaze_x[nnan_mask]
    # print('Close to the screen boundary gaze points: %d, %d'%(sum(gaze_x_nnan<0),sum(gaze_x_nnan>frame_width)) )
    gaze_x_nnan[ gaze_x_nnan<0] = 0
    gaze_x_nnan[ gaze_x_nnan>frame_width] = frame_width
    gaze_x[nnan_mask] = gaze_x_nnan 
    
    nnan_mask = ~np.isnan(gaze_y)
    gaze_y_nnan = gaze_y[nnan_mask]
    # print('Close to the screen boundary gaze points: %d, %d'%(sum(gaze_y_nnan<0),sum(gaze_y_nnan>frame_height)) )
    gaze_y_nnan[ gaze_y_nnan<0] = 0
    gaze_y_nnan[ gaze_y_nnan>frame_height] = frame_height
    gaze_y[nnan_mask] = gaze_y_nnan 
    
    
    # ----- compute the distance of the camera from eyes to estimate the degrees of visual angle (DVA) -----
    distance_eye2camera = ETdata_pd [ ['distance_eye2camera_left', 'distance_eye2camera_right'] ].values
    distance_eye2camera [ distance_eye2camera < 0] = np.NaN
    
    mask_invalid = np.ma.masked_invalid(distance_eye2camera) # in some cases distance_eye2camera contains inf.
    mean_dist = np.mean(mask_invalid)#.filled(np.nan)
    

    # ----- Load metadata to get some info. -----
    mat_contents = loadmat(meta_file)
    w1 = mat_contents['w1']
    if not audioQNR: w2 = mat_contents['w2']
    fps_w1 = w1['fps']
    # ----- o -----


    # ----- ToDo: We need physical measures from behavioural ET and scanner ET. -----
    DVAperpixel, pixelperDVA = tobii_physical_parameters(mean_dist/10.,w1) # send in mm -> cm
    
    # direct value for metadata file [Dan suggests using the computed one, i.e., pixelperDVA above ]
    pixelperDVA_altv = w1['pixperdva'].mean()
    
    # print(pixelperDVA,pixelperDVA_altv)

    '''    
        mov_height_w1 = w1['mov_height']
        mov_width_w1 = w1['mov_width']
        ScreenHeight_w1 = w1['ScreenHeight']
        ScreenWidth_w1 = w1['ScreenWidth']
        
        mov_height_w2 = w2['mov_height']
        mov_width_w2 = w2['mov_width']
        ScreenHeight_w2 = w2['ScreenHeight']
        ScreenWidth_w2 = w2['ScreenWidth']
        
        if mov_height_w1 != mov_height_w2 or mov_width_w1!=mov_width_w2:
            raise ValueError('Problem in movie resolution between w1 and w2!')

    '''
    # This part is to control for whether mov_height, mov_width, ScreenHeight, ScreenWidth 
    # that are read from metadata files (w1 and w2) are consistent across participants.
    if kwargs.keys():
        for vii in kwargs.keys():
            dum_txt = vii.rsplit('_',1) # split on last occurance.
            if dum_txt[1] == 'w1':
                if w1[dum_txt[0]] != kwargs[vii]:
                    raise ValueError('Problem in the value of %s: %d'%(vii,w1[dum_txt[0]]) )
            elif dum_txt[1] == 'w2' and not audioQNR:
                if w2[dum_txt[0]] != kwargs[vii]:
                    raise ValueError('Problem in the value of %s: %d'%(vii,w2[dum_txt[0]]) )
            else:
                raise ValueError("Input '%s' in kwargs is not in '_w1' or '_w2' format!"%vii)


    # ----- o -----
    movie_width = w1['mov_width'] 
    movie_height = w1['mov_height'] 


    ET_rec_time = ETdata_pd['rec_time'].values.copy()
    assert np.all(np.diff(ET_rec_time) > 0)
    movie_block = ETdata_pd['rec_time'].astype(float).between(movie_time_start_actual,movie_time_stop_actual).values
    # to be used in non-audioQNR case.
    ET_rec_time_r = ET_rec_time - movie_time_start_actual
    ET_rec_time_r = ET_rec_time_r[movie_block]
    
    if audioQNR: 
    
        ET_rec_time_keep = []
        question_info = np.zeros(len(ET_rec_time))

        for vd_cnt, vd_ii in enumerate(video_timeblocks):
            movie_block_this = ETdata_pd['rec_time'].astype(float).between(vd_ii[0],vd_ii[1]).values
            ET_rec_time_this = ET_rec_time[movie_block_this] - vd_ii[0] + video_timeblocks_static_altered[vd_cnt,0]
            # ET_rec_time_this = ET_rec_time[movie_block_this] - vd_ii[0] + pause_times[vd_cnt] # not good. 
        
            # correct movie_block_this based on start time of next video block -- removes a few samples from the end in each video block.
            if vd_cnt<len(video_timeblocks)-1:
                correct_ends = ET_rec_time_this >= video_timeblocks_static_altered[vd_cnt+1,0]
                p1_dum = movie_block_this.sum()
                p2_dum = correct_ends.sum()
                
                if np.any(correct_ends):
                    movie_block_this[movie_block_this] = ~correct_ends
                    assert (p1_dum - p2_dum) == movie_block_this.sum(), 'Correction failed!'
                    ET_rec_time_this = ET_rec_time[movie_block_this] - vd_ii[0] + video_timeblocks_static_altered[vd_cnt,0]
            
            assert np.all(question_info[movie_block_this]==0), 'Should be all zero!'
            question_info[movie_block_this] = vd_cnt+1  # give positive nums to questions. 
                    
            assert np.all(ET_rec_time_this>0)
            assert np.all(np.diff(ET_rec_time_this) > 0)
            
            ET_rec_time_keep.append(ET_rec_time_this)
        
        ET_rec_time_r = np.hstack(ET_rec_time_keep)
        assert np.all(np.diff(ET_rec_time_r) > 0)        
        
        
        for qt_ii in question_timeblocks:
            movie_block_this = ETdata_pd['rec_time'].astype(float).between(qt_ii[0],qt_ii[1]).values
            assert np.all(question_info[movie_block_this]==0), 'Should be all zero!'
            question_info[movie_block_this] = qt_ii[2]*-1 # give negative nums to questions. 

        
        vid_ET_use = question_info>0 # take only video-block parts. 
        movie_block = np.logical_and(movie_block,vid_ET_use)
        assert np.sum(movie_block) == np.sum(vid_ET_use)
    
    
    
    # Here we might multiply GazeX with w1['mov_width'] (actual frame_width) and GazeY with w1['mov_height'] (actual frame_height). 
    if scale2frame:
        ETdata_out_df = pd.DataFrame({'RecTime': ET_rec_time_r, 'GazeX': gaze_x[movie_block]*movie_width, 'GazeY': gaze_y[movie_block]*movie_height, 
                                      'PupilDiameter':pupil_diameter_avg[movie_block] })
        if audioQNR:
            ETdata_out_df['Qinfo'] = question_info[movie_block]
    else:
        ETdata_out_df = pd.DataFrame({'RecTime': ET_rec_time_r, 'GazeX': gaze_x[movie_block], 'GazeY': gaze_y[movie_block], 
                                      'PupilDiameter':pupil_diameter_avg[movie_block] })
        if audioQNR:
            ETdata_out_df['Qinfo'] = question_info[movie_block]
    
    
    # --- Frame to time instances mapping ---
    # movie_time_frame_mapping[:,0] = movie_time_frame_mapping[:,0] - movie_time_start
    # movie_time_frame_mapping_altv[:,0] = movie_time_frame_mapping_altv[:,0] - movie_time_start_altv
    
    # altv_diff = abs(movie_time_frame_mapping[:,0] - movie_time_frame_mapping_altv[:,0])
    # frame_duration = 1./fps_w1
    
    # if np.max(altv_diff) > frame_duration:
        # print('Unexpected difference between movie_time_frame_mapping and _altv!')
        
    # ETdata_frame_info_df = pd.DataFrame({'frame_info_t':movie_time_frame_mapping[:,0], 'frame_info_num':movie_time_frame_mapping[:,1],
                                         # 'frame_info_t_altv':movie_time_frame_mapping_altv[:,0], 'frame_info_num_altv':movie_time_frame_mapping_altv[:,1] })
        # print(np.max(altv_diff))
        # raise SystemExit('Exit at p.max(altv_diff) ')
    # else:
        # ETdata_frame_info_df = pd.DataFrame({'frame_info_t':movie_time_frame_mapping[:,0], 'frame_info_num':movie_time_frame_mapping[:,1] })
    # ----- o -----
    
    return ETdata_out_df, pixelperDVA,pixelperDVA_altv,keep_calib_qual




def tobii_physical_parameters(distance_to_screen,w1):
    
    inches_to_cm_conv = 2.54;
    
    # --- Either measure physical lengths directly and provide here. 
    tobii_x = 20.04*inches_to_cm_conv; #physical measures
    tobii_y = 11.27*inches_to_cm_conv; #physical measures
    # --- OR load from w1 file.  
    # tobii_x = w1['physical_width']*inches_to_cm_conv #physical measures
    # tobii_y = w1['physical_height']*inches_to_cm_conv #physical measures
    
    
    pixpercm_x = w1['ScreenWidth'] / tobii_x
    pixpercm_y = w1['ScreenHeight'] / tobii_y
    
    pixpercm = np.mean([pixpercm_x, pixpercm_y])
    
    pixel_size = 1./pixpercm
    
    # DVAperpixel= 2.*atan(pixel_size/(2*distance_to_screen))*180/np.pi
    DVAperpixel= calcDVA(pixel_size,distance_to_screen)
    
    try:
        pixelperDVA = 1./DVAperpixel;
    except ZeroDivisionError:
        raise ValueError('Problem in tobii_physical_parameters!')

    return DVAperpixel, pixelperDVA



def calcDVA(image_size,distance_to_screen):
    # calcDVA(image_size,distance_to_screen) calculates degrees of visual angle 
    # using a known distance_to_screen (in cm) and a given image_size (in cm)
    #
    # can also input many image_sizes at once e.g., [1 2 3 4]. 
    # dva = rad2deg(2*atan(image_size/(2*distance_to_screen)));

    return 2.*atan(image_size/(2.*distance_to_screen))*180/np.pi



