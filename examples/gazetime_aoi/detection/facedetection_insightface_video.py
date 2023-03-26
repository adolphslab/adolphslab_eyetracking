#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import os
import cv2
import numpy as np

from pims import PyAVReaderTimed, PyAVReaderIndexed
from tqdm import tqdm

from insightface.app import FaceAnalysis

from vizutils import get_faceareas_boxes



def faces2dets(faces):
    # reformat detections to the same format as pytorch_retinaface 
    if len(faces) < 1: 
        # tNo face detected for the image/frame.
        return []
    else:
        dets = []
        for fc_ii in faces:
            dets_this = [ *fc_ii.bbox, fc_ii.det_score, *fc_ii.kps.flatten() ]
            dets.append(dets_this)
    return np.asarray(dets)


def run_face_detection(video_file, output_dir, save_viz=True):

    print(f'Processing face detection for:\n {video_file}\n\n')
    
    # --- preps ---
    vid_basename = os.path.basename(video_file)
    vid_basename = f'{os.path.splitext(vid_basename)[0]}_insightface' 
    
    # --- Output file to save densepose outputs ---
    pkl_output_file = os.path.join(output_dir,'%s.pkl'%vid_basename)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # ----- Insightface settings -----
    
    # model_pack_name = 'buffalo_l' # default 
    # app = FaceAnalysis(name=model_pack_name)
    # app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app = FaceAnalysis(allowed_modules=['detection'],providers=['CUDAExecutionProvider', 'CPUExecutionProvider']) # enable detection model only
    app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5) # requires a lower threshold than pytorch implementation.
    vis_thresh = 0.5
    
    ''' insightface app useage:
    from insightface.app import FaceAnalysis
    from insightface.data import get_image as ins_get_image
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    img = ins_get_image('t1')
    faces = app.get(img)
    rimg = app.draw_on(img, faces)
    cv2.imwrite("t1_output.jpg", rimg)
    '''
    
    # ------- Video info -------
    vr = PyAVReaderTimed(video_file)
    frame_width, frame_height = vr.frame_shape[1], vr.frame_shape[0]
    nframes_a, vid_fps = len(vr), vr.frame_rate
    
    # reload video file for a more accurate count of frames in the video
    vr = PyAVReaderIndexed(video_file)
    nframes = len(vr)
    # ------- o -------
    
    if save_viz:
        output_fname = os.path.join(output_dir,'%s.mp4'%vid_basename)
    
        output_vid_file = cv2.VideoWriter(filename=output_fname,
                    # fourcc=cv2.VideoWriter_fourcc(*"MPEG"), # for .avi
                    fourcc=cv2.VideoWriter_fourcc(*"mp4v"), # for .mp4
                    fps=float(vid_fps),
                    frameSize=(frame_width,frame_height), # (width, height),
                    isColor=True )
    # ------- o -------
    
    feats_data = {}
    
    for fii, frame_ii in enumerate(tqdm(vr, total=nframes)):
        
        # NEED TO SEND IMAGES IN RGB --> BGR FORMAT TO insightface
        img = cv2.cvtColor(frame_ii, cv2.COLOR_RGB2BGR)
        
        faces = app.get(img)
        dets = faces2dets(faces)
        
        if len(dets) == 0:
            pass
        else:
            feats_data[f'frame_{fii+1}'] = dets
    
        if save_viz:
            
            face_boxes, face_boxes_eyes, face_boxes_mouth = get_faceareas_boxes(dets,
                                                frame_height,frame_width,detection_thrs=vis_thresh)            
    
            if len(face_boxes)== 0:
                continue
    
            for face_cnt in range(len(face_boxes)):
    
                fbox = face_boxes[face_cnt]
                cv2.rectangle(img, (fbox[2],fbox[0]), (fbox[3],fbox[1]), (255,255,255), thickness=2)
                # cv2.rectangle(img, (fbox[2],fbox[0]), (fbox[3],fbox[1]), (105,105,105), cv2.FILLED)
    
                mbox = face_boxes_mouth[face_cnt]
                cv2.rectangle(img, (mbox[2],mbox[0]), (mbox[3],mbox[1]), (208,224,64), 2)
                # cv2.rectangle(img, (mbox[2],mbox[0]), (mbox[3],mbox[1]), (208,224,64), cv2.FILLED)
    
                ebox = face_boxes_eyes[face_cnt]
                cv2.rectangle(img, (ebox[2],ebox[0]), (ebox[3],ebox[1]), (0,140,255), 2)
                # cv2.rectangle(img, (ebox[2],ebox[0]), (ebox[3],ebox[1]), (0,140,255), cv2.FILLED)
                
            output_vid_file.write(img)
            
    # Close open files and save variables. 
    if save_viz:
        output_vid_file.release()
        
    import pickle
    with open(pkl_output_file,'wb') as pfile:
        pickle.dump(feats_data, pfile)
    

import argparse

def str2bool(arg_in):
    if isinstance(arg_in, bool):
        return arg_in
    elif arg_in.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif arg_in.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean [True or False] value expected.')


def main(args):

    if not os.path.isfile(args.video):
        raise SystemExit(f'Cannot find the video file. Please check this.\n File: {args.video}')

    run_face_detection(args.video, args.output_dir, args.save_viz)


if __name__ == "__main__":

    # ----- General settings ------
    # Video for feature extraction. 
    video_file = '../../sample_data/sample_input/office_sample_vid.mp4'
    
    # Folder to save output visualization videos. 
    output_dir = os.path.abspath('../../sample_data/sample_output')
    
    # Visualization output setting
    save_viz = True

    parser = argparse.ArgumentParser(description='Face detection using insightface')
    parser.add_argument('-v', '--video', help='enter a video file to process', 
                        type=str, default=video_file)
    
    parser.add_argument('-o', '--output_dir', help='enter a directory to save outputs', 
                        type=str, default=output_dir)
    
    parser.add_argument('-s', '--save_viz', help='Boolean [True or False], whether to save the visualization of detected AOIs [default:True]',
                        type=str2bool, default=save_viz)
    
    args = parser.parse_args()

    main(args)


