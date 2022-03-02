#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""


import os
import cv2
import numpy as np

import pims
from tqdm import tqdm

from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

from .etutils import et_heatmap


def rgb2gray3ch(rgb):
    gray1ch = np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    w, h = gray1ch.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  gray1ch
    return ret


def convertMillis(millis):
    seconds=np.int((millis/1000)%60)
    minutes=np.int((millis/(1000*60))%60)
    return "{:02d}m {:02d}s".format(minutes, seconds)



def plot_gaze_basic(et_xy, video_file, merge_groups=False,
                    colors=['C3', 'C0', 'C1', 'C9'], zorder=[1,2,3,4],
                    save_viz=False, output_dir=None, show_viz=False, prep_viz=True):

    if save_viz and output_dir is None:
        raise SystemExit("Please provide an 'output_dir' to save visualization output!")
    
    if save_viz or show_viz:
        print("prep_viz set to False to scan all through the clip!")
        prep_viz = False
        
    if show_viz:
        print("\n\tPress q to stop script!\n")
        
    
    vr = pims.PyAVReaderTimed(video_file)
    frame_width, frame_height = vr.frame_shape[1], vr.frame_shape[0]
    nframes, vid_fps = len(vr), vr.frame_rate

    vid_basename = os.path.splitext(os.path.basename(video_file))[0]
    if save_viz:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # some setting for writing videos with openCV.        
        output_fname = os.path.join(output_dir,vid_basename) + "_ETgaze.avi"
        output_file = cv2.VideoWriter(filename=output_fname,
                    # if opencv does not support x264
                    # try other formats (e.g. MPEG, XVID).
                    # all formats work with a .avi format (alternative .mp4). 
                    fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
                    fps=float(vid_fps),
                    frameSize=(frame_width,frame_height), # (width, height),
                    isColor=True )
    
    
    print('Generating visualization...')
    dum_cnt = 0 # some dummy counter for prep_visualization case below.
    for fii, frame_ii in enumerate(tqdm( vr, total=nframes)):
        
        fig = plt.Figure(frameon=False)
        dpi = fig.get_dpi()
        fig.set_size_inches( (frame_width + 1e-2) / dpi,  (frame_height + 1e-2) / dpi )
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.imshow(frame_ii) # to show video frame in color. 
        # ax.imshow(rgb2gray3ch(frame_ii)) # to show video frame in gray-scale. 
        
        # plot data from each subject.
        if merge_groups:
            et_xy_plot = np.concatenate(et_xy,0)
            for sii in range(et_xy_plot.shape[0]):
                ax.plot(et_xy_plot[sii,fii,0],et_xy_plot[sii,fii,1],'o' ,markersize=10)
        else:
            for gii in range(len(et_xy)):
                et_xy_plot = et_xy[gii]
                for sii in range(et_xy_plot.shape[0]):
                    ax.plot(et_xy_plot[sii,fii,0],et_xy_plot[sii,fii,1],'o',color=colors[gii], alpha=1-(gii*0.12) ,markersize=10)
    
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
            
        if prep_viz:
            from PIL import Image
            im = Image.frombytes("RGB", (canv_width, canv_height), canvas.tostring_rgb()) # then: im.show()
            im.show()
            
            # a control againts opening too many image display windows.
            dum_cnt += 1
            if dum_cnt>2:
                raise SystemExit("It is better to set prep_visualization=False")
                
        if save_viz:    
            output_file.write(img_rgb[...,::-1])
    
        if show_viz:
            cv2.namedWindow(vid_basename, cv2.WINDOW_NORMAL)
            cv2.imshow(vid_basename, img_rgb[...,::-1])     
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break # q to quit. 
            if cv2.waitKey(1) == 27:
                break  # esc to quit.            
    
    if save_viz:
        # This video clip has finished. 
        output_file.release()
        
    if show_viz:
        cv2.destroyAllWindows()
        


# from skimage.filters import threshold_mean

def plot_compare_2groups(et_xy, video_file, sigma=21, plot_groups=[0,1],
                         colors=['C3', 'C0', 'C1', 'C9'], zorder=[1,2,3,4],
                         save_viz=False, output_dir=None, show_viz=False, prep_viz=True):

    # sigma is the standard deviation for Gaussian kernel. 
    # Corresponds to 1 degree visual angle. Equal for both axes. 43.0
    dpi = 140
    fig_w, fig_h = 10, 7
    # dpi = 120
    # fig_w, fig_h = 9, 6

    if save_viz and output_dir is None:
        raise SystemExit("Please provide an 'output_dir' to save visualization output!")
    
    if save_viz or show_viz:
        print("prep_viz set to False to scan all through the clip!")
        prep_viz = False
        
    if show_viz:
        print("\n\tPress q to stop script!\n")

    assert len(plot_groups) == 2, 'This function compares two groups!'
    
    grp_1, grp_2 = plot_groups
    
    vr = pims.PyAVReaderTimed(video_file)
    frame_width, frame_height = vr.frame_shape[1], vr.frame_shape[0]
    nframes, vid_fps = len(vr), vr.frame_rate
    framesize = [frame_width,frame_height] # required for heatmap.
    
    vid_basename = os.path.splitext(os.path.basename(video_file))[0]
    if save_viz:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # some setting for writing videos with openCV.        
        output_fname = os.path.join(output_dir,vid_basename) + "_compare_grps.avi"
        output_file = cv2.VideoWriter(filename=output_fname,
                    # if opencv does not support x264
                    # try other formats (e.g. MPEG, XVID).
                    # all formats work with a .avi format (alternative .mp4). 
                    fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
                    fps=float(vid_fps),
                    frameSize=(int(fig_w*dpi),int(fig_h*dpi)), # (width, height),
                    isColor=True )
    
    
    print('Generating visualization...')
    dum_cnt = 0 # some dummy counter for prep_visualization case below.
    for fii, frame_ii in enumerate(tqdm( vr, total=nframes)):
        
        fig = plt.Figure(frameon=None,dpi=dpi)
        fig.set_size_inches(fig_w, fig_h)
        canvas = FigureCanvasAgg(fig)
        
        gs = gridspec.GridSpec(2,6) # wspace=0.5
        ax1 = fig.add_subplot(gs[0, :3]) # fixations. 
        ax2 = fig.add_subplot(gs[0, 3:]) # zoom panel. 
        ax3 = fig.add_subplot(gs[1, :3]) # plot in the bottom.        
        ax4 = fig.add_subplot(gs[1, 3:]) # plot in the bottom.        
    
    
        for aii, (ax,grp) in enumerate([ (ax1,grp_1), (ax2,grp_2) ]):    
            
            ax.axis("off")
            # ax.imshow(frame_ii) # to show video frame in color. 
            ax.imshow(rgb2gray3ch(frame_ii)) # to show video frame in gray-scale. 
        
            et_xy_plot = et_xy[grp]
            for sii in range(et_xy_plot.shape[0]):
                ax.plot(et_xy_plot[sii,fii,0],et_xy_plot[sii,fii,1],'o',color=colors[aii] ,markersize=10)
        
            ax.set_xlim([0,frame_width])
            ax.set_ylim([0,frame_height])
            # make the origin top left corner. 
            ax.invert_yaxis()
        
        
        for aii, (ax,grp) in enumerate([ (ax3,grp_1), (ax4,grp_2) ]):    
            
            ax.axis("off")
            # ax.imshow(frame_ii) # to show video frame in color. 
            ax.imshow(rgb2gray3ch(frame_ii)) # to show video frame in gray-scale. 
        
            # get heatmap
            et_xy_plot = et_xy[grp]
            heatmap = et_heatmap(et_xy_plot[:,fii,:], framesize, sigma, None, get_full=True, get_down=False )
            
            # this thresholding is different than threshold_mean due to heatmap>0 condition.
            lowbound = np.mean(heatmap[heatmap>0])
            # lowbound = threshold_mean(heatmap)
            
            heatmap[heatmap<=lowbound] = np.nan
            ax.imshow(heatmap,alpha=0.75,zorder=20,cmap=plt.cm.jet,origin='upper')    
            
            ax.set_xlim([0,frame_width])
            ax.set_ylim([0,frame_height])
            # make the origin top left corner. 
            ax.invert_yaxis()
        
        
        plt.tight_layout()
        # Draw the figure first.
        canvas.draw()
                    
        canv_width, canv_height = canvas.get_width_height()
        img_rgb = np.frombuffer(canvas.tostring_rgb(), np.uint8).reshape((canv_height,canv_width, 3)) # RGB.
    
        if (canv_width, canv_height) != (int(fig_w*dpi),int(fig_h*dpi)):
                    raise SystemExit('Potential problem in cv2.VideoWriter!')        
            
            
        if prep_viz:
            from PIL import Image
            im = Image.frombytes("RGB", (canv_width, canv_height), canvas.tostring_rgb()) # then: im.show()
            im.show()
            
            # a control againts opening too many image display windows.
            dum_cnt += 1
            if dum_cnt>0:
                raise SystemExit("It is better to set prep_visualization=False")
                
        if save_viz:    
            output_file.write(img_rgb[...,::-1])
    
        if show_viz:
            cv2.namedWindow(vid_basename, cv2.WINDOW_NORMAL)
            cv2.imshow(vid_basename, img_rgb[...,::-1])     
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break # q to quit. 
            if cv2.waitKey(1) == 27:
                break  # esc to quit.            
    
    if save_viz:
        # This video clip has finished. 
        output_file.release()
        
    if show_viz:
        cv2.destroyAllWindows()
        
    
    
    
    
    
    
    
    