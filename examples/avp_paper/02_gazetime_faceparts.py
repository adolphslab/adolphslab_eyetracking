#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Codes to generate the figures shown in the auism ET paper

2- Compute gaze time to face area and parts.

"""

import os
import numpy as np
import pickle
from tqdm import tqdm

# The Adolphs Lab data analysis scripts for eye tracking are provided 
# under alabeye package  
from alabeye.etdata import ETdata, makedir
from alabeye.io.loadstimfeatures import get_faceareas_simple, radmask, sort_facedetect_by_salience, get_faceareas

from scipy.spatial.distance import cdist


#%% Main directory for experiment data
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'

# The main directory for experimental data contains several subdirectories such as:
# - ETdata: eye tracking data
# - ETdata/down2frame_data: gaze data downsampled to the frame rate of video stimulus
# - BehavioralData: demographic and psychological assessment info about subjects
# - StimVids: media files for experimental stimuli [not shared here because of copyright restrictions] 
# - FeatureZoo: various features extracted from video stimuli 

stim_dir = os.path.join(root_dir,'StimVids')
features_dir = os.path.join(root_dir,'FeatureZoo')
prepdata_dir = os.path.join(root_dir,'ETdata','down2frame_data')

# Directory to save outputs.
output_dir = os.path.join(root_dir,'Results_v1','Face_GazeTime')

#%% Other settings

# Information and settings about face detection files
# Pickle file that contains retinaface results. 
pickle_file_facedetect = os.path.join(features_dir,'Vids_Features_retinaface','%s_facedetect.pkl')

# Videos to compute gaze times.
vidclips = ['Ep1_Clip1', 'Ep1_Clip2', 'Ep1_Clip3', 'Ep4_AQNR']
gaze_radius = 21.0 # Half of 1 degree visual angle 

# set up the output directory
makedir(output_dir,sysexit=True)

#%%
for vii, vid_ii in enumerate(vidclips):
    
    print('Processing video file: %s'%vid_ii)

    # load downsampled gaze data, which were prepared in 00_loaddata.py
    data_file = os.path.join(prepdata_dir,f'timebinned_data_{vid_ii}.pkl')
    vid_etdata = ETdata(data_file=data_file,stim_dir=stim_dir)
    ngroups = vid_etdata.data_ngroups


    # ----- Load some information about the video -----
    nframes = vid_etdata.stim_mediainfo['nframes']
    frame_width = vid_etdata.stim_mediainfo['frame_width']
    frame_height = vid_etdata.stim_mediainfo['frame_height']


    # ----- Load facedetect / retinaface results -----
    vid_basename = os.path.splitext(vid_etdata.stim_videoname)[0]
    with open(pickle_file_facedetect%vid_basename,'rb') as pf:
        facedetect_results = pickle.load(pf)

    et_xy = np.concatenate(vid_etdata.data, axis=0)
    subj_lists = vid_etdata.data_subjs
    subj_groups = vid_etdata.data_subjs_group
    subjs = np.hstack(subj_lists)

    # 
    face_use_info = []
    face_size_info = []
    
    face_gaze = np.zeros(et_xy.shape[:2],dtype=np.int8)
    eye_gaze = np.zeros(et_xy.shape[:2],dtype=np.int8)
    mouth_gaze = np.zeros(et_xy.shape[:2],dtype=np.int8)
    face_gaze_dists = np.full((*et_xy.shape[:2],3),np.nan) # 3 distances. 
    face_gaze_dists_valid = np.zeros(et_xy.shape[:2],dtype=bool)
    face_num = np.zeros(et_xy.shape[:2],dtype=int)

    for fii in tqdm(range(nframes)):

        # --- Load ET data for both groups ---    
        et_bin = et_xy[:,fii,:]

        scan_area = []
        for sub_ii, sub_et_ii in enumerate(et_bin):
            if not np.isnan(sub_et_ii.mean()):
                this_scan_mask = radmask(sub_et_ii, gaze_radius, [frame_height,frame_width])
                scan_area.append(this_scan_mask)
            else:
                scan_area.append(None)

        # --- Load face-detection results ---
        frame_name = f'frame_{fii+1}'
        frame_results_facedetect = facedetect_results.get(frame_name,None)
        
        face_areas = None
        face_use = None
        face_size = None
        if frame_results_facedetect is not None:

            frame_results_facedetect = sort_facedetect_by_salience(frame_results_facedetect,scan_area,frame_height,frame_width)
            face_areas, landmarks = get_faceareas_simple(frame_results_facedetect,frame_height,frame_width,detection_thrs=0.5)
            # landmarks are: left/right eyes, nose, mouth. 
            # in face areas 1: the most salient, 2: lesser, 3: more lesser, etc. 
            # in landmarks: first one the most salient face, second lesser, ...
            
            _, face_areas_eyes, face_areas_mouth = get_faceareas(frame_results_facedetect,frame_height,frame_width,detection_thrs=0.5)            
            
            # assess face size to determine whether a face size is large enough to reliably measure distances from landmarks. 
            diff_faces = np.unique(face_areas[face_areas>0])
            if len(diff_faces) != len(landmarks):
                print('\n Overlapping face areas!')
                print(diff_faces)
                landmarks = [ landmarks[int(of_ii)] for of_ii in diff_faces-1 ]
                diff_faces = np.arange(1,len(landmarks)+1).astype(float)
                print(diff_faces)
            
            assert len(diff_faces) == len(landmarks)

            face_use = np.ones((len(diff_faces))).astype(bool)
            for la_ii, landmark in enumerate(landmarks):
                # land_dists = np.hstack( (cdist( landmark[[0,1],:], landmark[[2,3],:] ).ravel(), pdist( landmark[[2,3],:]) ))
                land_dists = cdist( landmark[[0,1],:], landmark[[2,3],:] ) # distance between eyes and nose or mouth. 
                # print(land_dists)
                if (land_dists < gaze_radius*2.).any():
                    face_use[la_ii] = False

            face_size = np.zeros((len(diff_faces)))
            for face_cnt,face_ii in enumerate(diff_faces):
                this_face = face_areas==face_ii
                face_size[face_cnt] = np.sum(this_face>0)

        # keep face use info.
        face_use_info.append(face_use)
        face_size_info.append(face_size)

        
        # Count face gaze and measure gaze to landmark distances.
        if not face_areas is None:
            # diff_faces = np.unique(face_areas[face_areas>0])
            for sc_cnt,sc_ii in enumerate(scan_area):
                if sc_ii is None:
                    face_gaze[sc_cnt,fii] = -1
                    eye_gaze[sc_cnt,fii] = -1
                    mouth_gaze[sc_cnt,fii] = -1
                else:
                    for face_cnt,face_ii in enumerate(diff_faces):
                        this_face = face_areas==face_ii
                        this_eyes = np.logical_and(this_face,face_areas_eyes)
                        this_mouth = np.logical_and(this_face,face_areas_mouth)
                        this_landmarks = landmarks[face_cnt]
                        overlap_TF = np.logical_and(this_face, sc_ii).any()
                        # gaze-to-eyes or mouth area:
                        eyes_overlap = np.logical_and(this_eyes, sc_ii).sum()
                        mouth_overlap = np.logical_and(this_mouth, sc_ii).sum()
                        if overlap_TF:
                            face_gaze[sc_cnt,fii] = 1
        
                            if eyes_overlap > mouth_overlap:
                                eye_gaze[sc_cnt,fii] = 1
                            elif eyes_overlap < mouth_overlap:
                                mouth_gaze[sc_cnt,fii] = 1
                            else: # should be rare. 
                                if eyes_overlap > 0:
                                    eye_gaze[sc_cnt,fii] = 1
                                if mouth_overlap > 0:
                                    mouth_gaze[sc_cnt,fii] = 1 

                            dist_dum = cdist([et_bin[sc_cnt,:]],this_landmarks).squeeze()
                            face_gaze_dists[sc_cnt,fii,:] = [ dist_dum[[0,1]].min(), dist_dum[2], dist_dum[3] ]
                            face_gaze_dists_valid[sc_cnt,fii] = face_use[int(face_ii)-1]
                            assert face_cnt == int(face_ii)-1, 'Expected pattern...'
                            face_num[sc_cnt,fii] = face_ii
                            break


        else: # onscreen fixation when there is no face on the screen. 
            for sc_cnt,sc_ii in enumerate(scan_area):
                if sc_ii is None:
                    face_gaze[sc_cnt,fii] = -1
                    eye_gaze[sc_cnt,fii] = -1
                    mouth_gaze[sc_cnt,fii] = -1


    # save this video outputs
    pkl_output_file = os.path.join(output_dir,f'{vid_ii}_fixtime_coeff_{gaze_radius}.pkl')
    data2out = {'face_use_info':face_use_info, 'face_size_info':face_size_info, 
                'subjs':subjs, 'subj_lists':subj_lists, 'subj_groups':subj_groups,
                'face_gaze':face_gaze, 'eye_gaze':eye_gaze, 'mouth_gaze':mouth_gaze,
                'face_gaze_dists':face_gaze_dists, 'face_gaze_dists_valid':face_gaze_dists_valid,
                'face_num':face_num}
    with open(pkl_output_file,'wb') as fp:
        pickle.dump(data2out,fp)

    
