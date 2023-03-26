#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import os
import numpy as np
import cv2

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.vis.extractor import DensePoseOutputsExtractor, DensePoseResultExtractor

from vizutils import array2str_encoder
from vizutils import densepose_results2bodypart_layers, densepose_facehand_layers

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pylab as plt

from pims import PyAVReaderTimed, PyAVReaderIndexed
from tqdm import tqdm


def run_bodypart_detection(video_file, output_dir, save_viz=True):
    
    print(f'Processing body parts detection for:\n {video_file}\n\n')

    # --- preps ---
    vid_basename = os.path.basename(video_file)
    vid_basename = f'{os.path.splitext(vid_basename)[0]}_densepose' 
    
    # --- Output file to save densepose outputs ---
    pkl_output_file = os.path.join(output_dir,'%s.pkl'%vid_basename)
    os.makedirs(output_dir, exist_ok=True)
    
    # ------- Densepose settings: -------
    # Download pre-trained models from:
    # https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/DENSEPOSE_IUV.md#ModelZoo
    # Original from Guler et al. 2018
    
    model_zoo_dir = '/home/umit/Downloads/detectron_wts/sample_model_zoo'
    config_fpath = os.path.join(model_zoo_dir,'densepose_rcnn_R_101_FPN_s1x_legacy.yaml')
    model_fpath  = os.path.join(model_zoo_dir,'model_final_ad63b5.pkl')
    
    
    cfg = get_cfg()
    add_densepose_config(cfg)
    
    cfg.merge_from_file(config_fpath)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6 
    cfg.MODEL.WEIGHTS = model_fpath
    # model_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    
    predictor = DefaultPredictor(cfg)
    # ------- o -------
    
    
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
    
    def get_densepose_result(pred_in):
        
        result_prep = {}
        if pred_in.has("pred_boxes"):
            
            pred_boxes_XYXY = pred_in.get("pred_boxes").tensor
            # boxes_XYWH = BoxMode.convert(pred_boxes_XYXY, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        
            result_prep["dp_boxes_xyxy"] = np.round( pred_boxes_XYXY.numpy() ).astype(int)
            result_prep["dp_scores"] = pred_in.get("scores").numpy()
            
            if pred_in.has("pred_densepose"):
        
                if isinstance(pred_in.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(pred_in.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                
                densepose_result = extractor(pred_in)[0]
                
                this_frame_densepose_str = [] 
                for dii in densepose_result:
                    this_frame_densepose_str.append( array2str_encoder(dii.labels.numpy()) )    
        
                result_prep["dp_parts_str"] = this_frame_densepose_str
    
                return result_prep
            
            # No detection. 
            else: 
                return None
    # ------- o -------
            
    feats_data = {}
    
    for fii, frame_ii in enumerate(tqdm(vr, total=nframes)):
        
        # NEED TO SEND IMAGES IN RGB --> BGR FORMAT TO DENSEPOSE/DETECTRON.
        img = cv2.cvtColor(frame_ii, cv2.COLOR_RGB2BGR)
    
        model_output = predictor(img)
        predictions = model_output["instances"].to("cpu")
        result_thisframe = get_densepose_result(predictions)
    
        if save_viz:
            fig = plt.Figure(frameon=False)
            dpi = fig.get_dpi()
            fig.set_size_inches( (frame_width + 1e-2) / dpi,  (frame_height + 1e-2) / dpi )
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.axis("off")        
            ax.imshow(frame_ii)
    
        if result_thisframe is None:
            pass
        else:
            feats_data[f'frame_{fii+1}'] = result_thisframe
    
            if save_viz:
                img_layers, _ = densepose_results2bodypart_layers(result_thisframe, img.shape)
                if img_layers is not None:
                    img_layers_red = densepose_facehand_layers(img_layers)
    
                    img_layers_red = img_layers_red.astype(float)
                    img_layers_red[img_layers_red==0] = np.NaN
                    ax.imshow(img_layers_red,alpha=0.5)
    
        if save_viz:
            ax.set_xlim([0,frame_width])
            ax.set_ylim([0,frame_height])
            # make the origin top left corner. 
            ax.invert_yaxis()
            
            # Draw the figure first.
            canvas.draw()
                        
            canv_width, canv_height = canvas.get_width_height()
            img_rgb = np.frombuffer(canvas.tostring_rgb(), np.uint8).reshape((canv_height,canv_width, 3)) # RGB.
            
            if (canv_width, canv_height) != (frame_width, frame_height):
                raise SystemExit('Potential problem in cv2.VideoWriter!')
            
            output_vid_file.write(img_rgb[...,::-1])
            
    
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

    run_bodypart_detection(args.video, args.output_dir, args.save_viz)


if __name__ == "__main__":

    # ----- General settings ------
    # Video for feature extraction. 
    video_file = '../../sample_data/sample_input/office_sample_vid.mp4'
    
    # Folder to save output visualization videos. 
    output_dir = os.path.abspath('../../sample_data/sample_output')
    
    # Visualization output setting
    save_viz = True

    parser = argparse.ArgumentParser(description='Body parts detection using detectron2/densepose')
    parser.add_argument('-v', '--video', help='enter a video file to process', 
                        type=str, default=video_file)
    
    parser.add_argument('-o', '--output_dir', help='enter a directory to save outputs', 
                        type=str, default=output_dir)
    
    parser.add_argument('-s', '--save_viz', help='Boolean [True or False], whether to save the visualization of detected AOIs [default:True]',
                        type=str2bool, default=save_viz)
    
    args = parser.parse_args()

    main(args)

