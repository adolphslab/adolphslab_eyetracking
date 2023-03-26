#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Some functions to load densepose feature extraction outputs. 

"""

import numpy as np


import base64
from io import BytesIO
from PIL import Image

from skimage.morphology import convex_hull_image

def radmask(center_xy,radius,array_shape):
    
    if np.array(center_xy).shape != (2,): 
        raise SystemExit('Problem in the input of radMask')
    
    col_val = center_xy[0] 
    row_val = center_xy[1] 
    n_rows,n_cols = array_shape
    rows,cols = np.ogrid[-row_val:n_rows-row_val,-col_val:n_cols-col_val]
    mask = cols*cols + rows*rows <= radius*radius
    return mask    



#%% for bodyparts obtained from densepose

def decode_png_data_original(shape, s):
    """
    From FaceBook's Detectron2/DensePose.
    Decode array data from a string that contains PNG-compressed data
    @param Base64-encoded string containing PNG-compressed data
    @return Data stored in an array of size (3, M, N) of type uint8
    """
    fstream = BytesIO(base64.decodebytes(s.encode()))
    im = Image.open(fstream)
    data = np.moveaxis(np.array(im.getdata(), dtype=np.uint8), -1, 0)
    return data.reshape(shape)


def densepose_reencoder(s_in):
    # Decode densepose's compressed output for iuv matrix. Get only bodypart index part. 
    # Then encode back to compressed form. 
    
    fstream = BytesIO(base64.decodebytes(s_in.encode()))
    # option 1:
    #data_iuv = np.asarray(Image.open(fstream))
    # im_out = Image.fromarray(data_iuv[...,0])
    
    # option 2: # only very slightly faster than option 1.  
    im_out = Image.open(fstream).getchannel(0)
    # im_out = Image.open(fstream).split()[0] # option 3. 
    
    fstream = BytesIO()
    im_out.save(fstream, format="png", optimize=True)
    s_out = base64.encodebytes(fstream.getvalue()).decode()
    
    return s_out


def densepose_decoder(s_in):
    fstream = BytesIO(base64.decodebytes(s_in.encode()))
    return np.asarray(Image.open(fstream))


def masks_encoder(arr_):
    # arr_ is a bool matrix of person mask.
    # Then encode back to compressed form. 
    img_dum = arr_.astype(np.int8)
    im_out = Image.fromarray(img_dum)
    
    fstream = BytesIO()
    im_out.save(fstream, format="png", optimize=True)
    s_out = base64.encodebytes(fstream.getvalue()).decode()
    
    return s_out


def get_boundingbox_dense(arr_):
    rows = np.any(arr_, axis=1)
    cols = np.any(arr_, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # [xmin, ymin, xmax, ymax] corresponding to [ left, top, right, bottom ]
    return cmin, rmin, cmax, rmax 


        
def densepose_results2bodypart_layers(result_,img_shape,pred_score_thrs=0.8,keep_nperson=5000):
    bodyparts_layer = np.zeros(img_shape[:2],dtype=np.uint8)
    bodyparts_dum = np.zeros(img_shape[:2],dtype=np.uint8)

    keep_head_box = []
    for instance_id,pred_score in enumerate(result_["dp_scores"]):
        
        if pred_score >= pred_score_thrs and instance_id < keep_nperson:

            pred_box = result_['dp_boxes_xyxy'][instance_id]
            result_encoded = result_['dp_parts_str'][instance_id]
        
            bodyparts_instance = densepose_decoder(result_encoded)
            
            bodyparts_dum[:] = 0
            bodyparts_dum[pred_box[1]:pred_box[1]+bodyparts_instance.shape[0], 
                          pred_box[0]:pred_box[0]+bodyparts_instance.shape[1]] = bodyparts_instance
            
            head_indx_1, head_indx_2 = 23, 24
            head_inds = np.logical_or(bodyparts_dum==head_indx_1,bodyparts_dum==head_indx_2)                 
            
            # against a potential error in finding bounding box. 
            # head_inds[0,:],head_inds[-1,:] = 0, 0
            # head_inds[:,0],head_inds[:,-1] = 0, 0

            # face_borders = [ x_min, y_min, x_max, y_max ] corresponding to [ left, top, right, bottom ]
            if np.sum(head_inds):
                x_min, y_min, x_max, y_max = get_boundingbox_dense(head_inds)
                keep_head_box.append([ x_min, y_min, x_max, y_max, pred_score ])
            
            # used a masking step (bodyparts_dum), otherwise pred_box areas overlap.
            bodyparts_layer[ bodyparts_dum>0 ] = bodyparts_dum[ bodyparts_dum>0 ].copy()

    if np.sum(bodyparts_layer) < 0.1: # ==0: no person detected with pred_score >= pred_score_thrs.
        return None, None
    else:
        return bodyparts_layer,np.asarray(keep_head_box)



def densepose_get_indvheads(result_,img_shape,pred_score_thrs=0.8,keep_nperson=5000):
        
    bodyparts_layer = np.zeros(img_shape[:2],dtype=np.uint8)
    bodyparts_dum = np.zeros(img_shape[:2],dtype=np.uint8)

    keep_head_box = []
    head_num = 1
    for instance_id,pred_score in enumerate(result_["dp_scores"]):
        
        if pred_score >= pred_score_thrs and instance_id < keep_nperson:

            pred_box = result_['dp_boxes_xyxy'][instance_id]
            result_encoded = result_['dp_parts_str'][instance_id]
        
            bodyparts_instance = densepose_decoder(result_encoded)
            
            bodyparts_dum[:] = 0
            bodyparts_dum[pred_box[1]:pred_box[1]+bodyparts_instance.shape[0], 
                          pred_box[0]:pred_box[0]+bodyparts_instance.shape[1]] = bodyparts_instance
            
            head_indx_1, head_indx_2 = 23, 24
            head_inds = np.logical_or(bodyparts_dum==head_indx_1,bodyparts_dum==head_indx_2)                 
            
            # against a potential error in finding bounding box. 
            # head_inds[0,:],head_inds[-1,:] = 0, 0
            # head_inds[:,0],head_inds[:,-1] = 0, 0

            # face_borders = [ x_min, y_min, x_max, y_max ] corresponding to [ left, top, right, bottom ]
            if np.sum(head_inds):
                x_min, y_min, x_max, y_max = get_boundingbox_dense(head_inds)
                keep_head_box.append([ x_min, y_min, x_max, y_max, pred_score ])
            
            # used a masking step (bodyparts_dum), otherwise pred_box areas overlap.
            bodyparts_layer[ head_inds ] = head_num
            head_num += 1

    if np.sum(bodyparts_layer) < 0.1: # ==0: no person detected with pred_score >= pred_score_thrs.
        return None, None
    else:
        return bodyparts_layer,np.asarray(keep_head_box)
    


def densepose_facehand_layers(multiarea_layer):
    
    bodypart_inds = multiarea_layer.copy()
    bodyparts_reduced = np.zeros_like(bodypart_inds)
    
    head_indx_1, head_indx_2 = 23, 24
    head_inds = np.logical_or(bodypart_inds==head_indx_1,bodypart_inds==head_indx_2)
    
    hand_indx_1, hand_indx_2 = 3, 4
    hand_inds = np.logical_or(bodypart_inds==hand_indx_1,bodypart_inds==hand_indx_2)
    
    bodyparts_reduced[head_inds]= 3 # head.
    bodyparts_reduced[hand_inds]= 2 # hands.
    
    bodypart_inds[head_inds | hand_inds] = 0
    bodyparts_reduced[bodypart_inds>0] = 1 # other body-parts. 

    return bodyparts_reduced



#%% for face detection 

def get_boundingbox(arr_):
    rows = np.any(arr_, axis=1)
    cols = np.any(arr_, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # [xmin, ymin, xmax, ymax] corresponding to [ left, top, right, bottom ]
    return np.array([cmin, rmin, cmax, rmax]) 



def get_talk_areas(frame_results_,frame_height,frame_width):

    action_boxes = frame_results_[0]
    action_texts = frame_results_[1]

    talk_areas = np.zeros((frame_height,frame_width),dtype=bool)
    
    for bii,tii in zip(action_boxes,action_texts):
        if any([True for tii_in in tii if 'talk' in tii_in]):
            b = np.asarray(bii).astype(int)
            b[b<0] = 0
            if b[2]>frame_width: b[2] = frame_width-1 
            if b[3]>frame_height: b[3] = frame_height-1

            talk_areas[b[1]:b[3],b[0]:b[2]] = True

    return talk_areas
        



# used in fixation analyses for RetinaFace detections. 
def sort_facedetect_by_salience(this_frame_results,scan_area_Ctrl,frame_height,frame_width):
    n_subjs = []
    for hii in this_frame_results:
        face_areas = np.zeros((frame_height,frame_width)).astype(bool)

        b = np.maximum(hii.astype(int), 0)
        face_areas[b[1]:b[3],b[0]:b[2]] = True
        
        n_subjs.append( sum( np.logical_and(sii,face_areas).any() for sii in scan_area_Ctrl if (not sii is None) ) )
        
    this_frame_results = this_frame_results[np.argsort(n_subjs)[::-1]] # most salient to less. 
    return this_frame_results



def get_faceareas_simple(this_frame_results, frame_height, frame_width,
                         detection_thrs=0.5, sort_by_area=False,
                         return_corners=False, return_landmarks_full=False):

    # sort faces based on face size, rather than prediction probability.
    if sort_by_area:
        areas_ = []
        for hii in this_frame_results:
            areas_.append((hii[3]-hii[1])*(hii[2]-hii[0]))
        this_frame_results = this_frame_results[np.argsort(areas_)[::-1]]
    
    face_areas = np.zeros((frame_height,frame_width))

    if return_corners:
        corners = []
    
    face_num = 1
    landmarks_all = []
    for hii in this_frame_results:
        if hii[4] > detection_thrs:
            
            b = np.maximum(hii.astype(int), 0)
            if b[2]>frame_width: b[2] = frame_width-1 
            if b[3]>frame_height: b[3] = frame_height-1
            
            face_areas[b[1]:b[3],b[0]:b[2]] = face_num
            face_num += 1
            
            landmarks = b[5:].reshape(-1,2)
            #from viewers perspective: b5,b6: left eye; b7,b8: right eye;  
            # b9,b10: nose; b11,b12: left-side mouth; b13,b14: right-side mouth
            
            # control for out of frame detections. 
            landmarks[:, 0] = np.minimum(landmarks[:, 0], frame_width - 1)
            landmarks[:, 1] = np.minimum(landmarks[:, 1], frame_height - 1)
            
            assert landmarks.shape[0]==5
            # center_of_eyes = np.mean(landmarks[[0,1],:],axis=0).round().astype(int)
            mouth_center = np.mean(landmarks[[3,4],:],axis=0).round().astype(int)
            
            if return_landmarks_full:
                landmarks_all.append(np.vstack((landmarks)))
            else:
                landmarks_all.append(np.vstack((landmarks[:-2,:], mouth_center)))

            if return_corners:
                corners.append(b[0:4])
    
    if return_corners:
        return face_areas, landmarks_all, corners
    
    return face_areas, landmarks_all




# used in fixation analyses for RetinaFace detections. 
def get_faceareas(this_frame_results, frame_height, frame_width, detection_thrs=0.5):

    face_areas = np.zeros((frame_height,frame_width),dtype=bool)
    face_areas_eyes = np.zeros((frame_height,frame_width),dtype=bool)
    face_areas_mouth = np.zeros((frame_height,frame_width),dtype=bool)
    
    for hii in this_frame_results:
        if hii[4] > detection_thrs:
            
            b = np.maximum(hii.astype(int), 0)
            
            face_areas[b[1]:b[3],b[0]:b[2]] = True
            
            landmarks = b[5:].reshape(-1,2)
            #from viewers perspective: b5,b6: left eye; b7,b8: right eye;  
            # b9,b10: nose; b11,b12: left-side mouth; b13,b14: right-side mouth
            
            # control for out of frame detections. 
            landmarks[:, 0] = np.minimum(landmarks[:, 0], frame_width - 1)
            landmarks[:, 1] = np.minimum(landmarks[:, 1], frame_height - 1)
            
            box2 = [ np.min(landmarks[:,0]), np.min(landmarks[:,1]), 
                     np.max(landmarks[:,0]), np.max(landmarks[:,1]) ] 

            med_box = np.vstack((b[:4],box2)).mean(0).astype(int).reshape(-1,2)
            
            # Control for out of frame detections.
            med_box[:, 0] = np.minimum(med_box[:, 0], frame_width - 1)
            med_box[:, 1] = np.minimum(med_box[:, 1], frame_height - 1)
            med_box = med_box.flatten()

            # --- Following 2 parts will fail for non-vertical faces, 
            # but a good approximation for the Office videos. ---
            # upper - eye - 
            box_11 = [np.min([ med_box[0], med_box[2], landmarks[2,0] ]), np.min([med_box[1], landmarks[2,1]]), 
                      np.max([ med_box[0], med_box[2], landmarks[2,0] ]), np.max([med_box[1], landmarks[2,1]]) ]
            box_11 = np.maximum(box_11, 0)
            face_areas_eyes[box_11[1]:box_11[3],box_11[0]:box_11[2]] = True

            # lower - mouth - 
            box_22 = [np.min([ med_box[0], med_box[2], landmarks[2,0] ]), np.min([med_box[3], landmarks[2,1]]), 
                      np.max([ med_box[0], med_box[2], landmarks[2,0] ]), np.max([med_box[3], landmarks[2,1]]) ]
            box_22 = np.maximum(box_22, 0)
            face_areas_mouth[box_22[1]:box_22[3],box_22[0]:box_22[2]] = True

    return face_areas, face_areas_eyes, face_areas_mouth



# used in fixation analyses for RetinaFace detections. 
def get_faceareas_boxes(this_frame_results, frame_height, frame_width, detection_thrs=0.5):

    face_boxes = []
    face_boxes_eyes = []
    face_boxes_mouth = []
    
    for hii in this_frame_results:
        if hii[4] > detection_thrs:
            
            b = np.maximum(hii.astype(int), 0)
            
            face_boxes.append( [b[1], b[3], b[0], b[2]] )
            
            landmarks = b[5:].reshape(-1,2)
            #from viewers perspective: b5,b6: left eye; b7,b8: right eye;  
            # b9,b10: nose; b11,b12: left-side mouth; b13,b14: right-side mouth
            
            # control for out of frame detections. 
            landmarks[:, 0] = np.minimum(landmarks[:, 0], frame_width - 1)
            landmarks[:, 1] = np.minimum(landmarks[:, 1], frame_height - 1)
            
            box2 = [ np.min(landmarks[:,0]), np.min(landmarks[:,1]), 
                     np.max(landmarks[:,0]), np.max(landmarks[:,1]) ] 

            med_box = np.vstack((b[:4],box2)).mean(0).astype(int).reshape(-1,2)
            
            # Control for out of frame detections.
            med_box[:, 0] = np.minimum(med_box[:, 0], frame_width - 1)
            med_box[:, 1] = np.minimum(med_box[:, 1], frame_height - 1)
            med_box = med_box.flatten()

            # --- Following 2 parts will fail for non-vertical faces, 
            # but a good approximation for the Office videos. ---
            # upper - eye - 
            box_11 = [np.min([ med_box[0], med_box[2], landmarks[2,0] ]), np.min([med_box[1], landmarks[2,1]]), 
                      np.max([ med_box[0], med_box[2], landmarks[2,0] ]), np.max([med_box[1], landmarks[2,1]]) ]
            box_11 = np.maximum(box_11, 0)
            face_boxes_eyes.append([box_11[1], box_11[3], box_11[0], box_11[2]] )
            
            # lower - mouth - 
            box_22 = [np.min([ med_box[0], med_box[2], landmarks[2,0] ]), np.min([med_box[3], landmarks[2,1]]), 
                      np.max([ med_box[0], med_box[2], landmarks[2,0] ]), np.max([med_box[3], landmarks[2,1]]) ]
            box_22 = np.maximum(box_22, 0)
            face_boxes_mouth.append([box_22[1], box_22[3], box_22[0], box_22[2]] )
            
    return face_boxes, face_boxes_eyes, face_boxes_mouth


