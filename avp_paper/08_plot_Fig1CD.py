#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Plots Fig. 1, bar plots in panels C and D. 

"""


import os
import numpy as np
import pandas as pd

import matplotlib.pylab as plt
plt.rcParams['svg.fonttype'] = 'none'

from statsmodels.sandbox.stats.multicomp import multipletests


#%% Main directory for experiment data
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'

# Load results generated by:
# 04_gazetime_bodyparts_postprocess.py, 05_gazetime_faceparts_postprocess.py, 06_heatmapcorrs_postprocess.py
results_dir = os.path.join(root_dir,'Results_v1','Cohend_pool')

gaze_results_vid1_file = os.path.join(results_dir,'FaceGaze_dvals_Ep1.csv')
bodypart_results_vid1_file = os.path.join(results_dir,'BodyPartGaze_dvals_Ep1.csv')

gaze_results_vid2_file = os.path.join(results_dir,'FaceGaze_dvals_Ep4_AQNR.csv')
bodypart_results_vid2_file = os.path.join(results_dir,'BodyPartGaze_dvals_Ep4_AQNR.csv')

gazecorr_vid1_file = os.path.join(results_dir,'GazeCorr_dvals_Ep1.csv')
gazecorr_vid1_asdref_file = os.path.join(results_dir,'GazeCorr_dvals_Ep1_asdref.csv')

gazecorr_vid2_file = os.path.join(results_dir,'GazeCorr_dvals_Ep4_AQNR.csv')
gazecorr_vid2_asdref_file = os.path.join(results_dir,'GazeCorr_dvals_Ep4_AQNR_asdref.csv')

#%%

gaze_results_vid1_pd = pd.read_csv(gaze_results_vid1_file,index_col='Unnamed: 0')
bodypart_results_vid1_pd = pd.read_csv(bodypart_results_vid1_file,index_col='Unnamed: 0')
bodypart_results_vid1_pd = bodypart_results_vid1_pd.drop(['On-screen']) # use from face file. Drop before concat. 

gazecorr_vid1_pd = pd.read_csv(gazecorr_vid1_file,index_col='Unnamed: 0')
gazecorr_vid1_asdref_pd = pd.read_csv(gazecorr_vid1_asdref_file,index_col='Unnamed: 0')
vid1_pd = pd.concat([gaze_results_vid1_pd, bodypart_results_vid1_pd, gazecorr_vid1_pd, gazecorr_vid1_asdref_pd ])

# -----------
gaze_results_vid2_pd = pd.read_csv(gaze_results_vid2_file,index_col='Unnamed: 0')
bodypart_results_vid2_pd = pd.read_csv(bodypart_results_vid2_file,index_col='Unnamed: 0')
bodypart_results_vid2_pd = bodypart_results_vid2_pd.drop(['On-screen'])

gazecorr_vid2_pd = pd.read_csv(gazecorr_vid2_file,index_col='Unnamed: 0')
gazecorr_vid2_asdref_pd = pd.read_csv(gazecorr_vid2_asdref_file,index_col='Unnamed: 0')
vid2_pd = pd.concat([gaze_results_vid2_pd, bodypart_results_vid2_pd, gazecorr_vid2_pd, gazecorr_vid2_asdref_pd ])


#%%
# variable names to get from dataframes.
AOIs = ['On-screen', 'Face', 'Non-social content', 
        'Non-head body', 'Hands', 'Eyes', 'Mouth', 'GazeCorrs', 'GazeCorrs_asdref'] # use all faces.

# labels to use in figures. 
AOI_labels = ['On-screen', 'Faces', 'Non-social\ncontent',  
              'Non-head\nbody',  'Hands', 'Eyes', 'Mouths', 'Heatmap corr.\nRef. TD', 'Heatmap corr.\nRef. ASD' ]

vid1_dvals = np.asarray([ vid1_pd.loc[aii]['d-bootstrap-mean'] for aii in AOIs ])
vid2_dvals = np.asarray([ vid2_pd.loc[aii]['d-bootstrap-mean'] for aii in AOIs ])

vid1_CI_l = np.asarray([ vid1_pd.loc[aii]['d CI lower'] for aii in AOIs ])
vid1_CI_l = vid1_dvals - vid1_CI_l

vid2_CI_l = np.asarray([ vid2_pd.loc[aii]['d CI lower'] for aii in AOIs ])
vid2_CI_l = vid2_dvals - vid2_CI_l

vid1_CI_h = np.asarray([ vid1_pd.loc[aii]['d CI upper'] for aii in AOIs ])
vid1_CI_h = vid1_CI_h - vid1_dvals

vid2_CI_h = np.asarray([ vid2_pd.loc[aii]['d CI upper'] for aii in AOIs ])
vid2_CI_h = vid2_CI_h - vid2_dvals

vid1_pvals = np.asarray([ vid1_pd.loc[aii]['d-pval'] for aii in AOIs ])
vid2_pvals = np.asarray([ vid2_pd.loc[aii]['d-pval'] for aii in AOIs ])

# correction for multiple comparison.
pval_comb = np.hstack([vid1_pvals,vid2_pvals])
fdr_TF = multipletests(pval_comb,alpha=0.05,method='fdr_bh')[0] # fdr_bh
fdr_pvals = multipletests(pval_comb,alpha=0.05,method='fdr_bh')[1] # fdr_bh

fdr_vid1_TF, fdr_vid2_TF = np.split(fdr_TF,2) 
fdr_vid1_pvals, fdr_vid2_pvals = np.split(fdr_pvals,2)

#%% Start plotting

def autolabel(ax, rects, pvals_in, dvals_in, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rii, rect in enumerate(rects):
        height = rect.get_height()

        if pvals_in[rii] < 0.05: 
    #        if pval < 0.0001:  text = '***'
    #        elif pval < 0.001: text = '**'
            if pvals_in[rii] < 0.01:  
                text = '*'
                if dvals_in[rii] > 0:
                    ax.annotate( text,
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(offset[xpos]*0.5, 0.5),  # use 3 points offset
                                textcoords="offset points",  # in both directions
                                ha=ha[xpos], va='bottom',size=8)
                else:
                    ax.annotate( text,
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(offset[xpos]*0.5, -5),  # use 3 points offset
                                textcoords="offset points",  # in both directions
                                ha=ha[xpos], va='top',size=8)
            else: 
                text = r'p$\approx$%1.2f'%pvals_in[rii]
                if dvals_in[rii] > 0:
                    ax.annotate( text,
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                # xytext=(offset[xpos]*3, 3),  # use 3 points offset
                                xytext=(offset['center']*1, 0),  # use 3 points offset
                                textcoords="offset points",  # in both directions
                                ha='center', va='bottom',rotation=45, size=6)
                else:
                    ax.annotate( text,
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                # xytext=(offset[xpos]*3, 3),  # use 3 points offset
                                xytext=(offset['center']*1, 0),  # use 3 points offset
                                textcoords="offset points",  # in both directions
                                ha='center', va='top',rotation=45, size=6)
        
#%%
from matplotlib.gridspec import GridSpec

fig = plt.figure(constrained_layout=True, figsize=(5,1.9))
gs = GridSpec(1, 9, figure=fig)

ax1 = fig.add_subplot(gs[0, :7])
ax2 = fig.add_subplot(gs[0, 7:])

n_aoi1, n_aoi2 = 7,2

signf_max = 1.01
scolors = [(0.859,0.224,0.149,0.5),(0.122,0.471,0.706,0.5),(0.60,0.60,0.60,0.5)] 
colors = ['#db3926','#1f78b4','#999999'] 

index = np.arange(n_aoi1)
bar_width = 0.43

error_config =dict(ecolor ='#989898', linewidth=1, capsize=2, capthick=1.)

colors = [(0.14, 0.30, 0.43, 1) if pvii else (0.14, 0.30, 0.43, 0.3) for pvii in fdr_vid1_TF[:n_aoi1] ]
rects1 = ax1.bar(index, vid1_dvals[:n_aoi1], bar_width,
                color=colors, #edgecolor='k',
                yerr=[ vid1_CI_l[:n_aoi1],  vid1_CI_h[:n_aoi1] ], error_kw=error_config,
                label='Episode A')

colors = [(0.00, 0.57, 0.61, 1) if pvii else (0.00, 0.57, 0.61, 0.3) for pvii in fdr_vid2_TF[:n_aoi1] ]
rects2 = ax1.bar(index + bar_width, vid2_dvals[:n_aoi1], bar_width,
                color=colors, #edgecolor='k',
                yerr=[ vid2_CI_l[:n_aoi1],  vid2_CI_h[:n_aoi1] ], error_kw=error_config,
                label='Episode B')


ax1.set_ylabel("Cohen's d\n(TD-ASD)",fontsize=7,fontweight='normal')
ax1.set_xticks(index + bar_width*1/2)
labels1 = AOI_labels[:n_aoi1]
ax1.set_xticklabels(labels1,fontsize=7,rotation=45)
ax1.set_ylim([-1.30,1.30])
yticklabels = [-1., -0.5, 0., 0.5 ,1.]
ax1.set_yticks(yticklabels)
ax1.set_yticklabels(yticklabels,fontsize=7)

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()
ax1.margins(0.025)


autolabel(ax1,rects1,fdr_vid1_pvals[:n_aoi1], vid1_dvals[:n_aoi1], 'left')
autolabel(ax1,rects2,fdr_vid2_pvals[:n_aoi1], vid2_dvals[:n_aoi1], 'left')

ax1.annotate(r'*p$<$0.01', xy=(index[-1]+bar_width-0.2,0.95), ha='center',
             zorder=10, fontsize=7, fontweight='normal')


index = np.arange(n_aoi2)

colors = [(0.14, 0.30, 0.43, 1) if pvii else (0.14, 0.30, 0.43, 0.3) for pvii in fdr_vid1_TF[n_aoi1:] ]
rects1 = ax2.bar(index, vid1_dvals[n_aoi1:], bar_width,
                color=colors, #edgecolor='k',
                yerr=[ vid1_CI_l[n_aoi1:],  vid1_CI_h[n_aoi1:] ], error_kw=error_config )

colors = [(0.00, 0.57, 0.61, 1) if pvii else (0.00, 0.57, 0.61, 0.3) for pvii in fdr_vid2_TF[n_aoi1:] ]
rects2 = ax2.bar(index + bar_width, vid2_dvals[n_aoi1:], bar_width,
                color=colors, #edgecolor='k',
                yerr=[ vid2_CI_l[n_aoi1:],  vid2_CI_h[n_aoi1:] ], error_kw=error_config )

ax2.set_xticks(index + bar_width*1/2)
labels2 = AOI_labels[n_aoi1:]
ax2.set_xticklabels(labels2,fontsize=7,rotation=45)

ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks(yticklabels)
ax2.set_yticklabels(yticklabels,fontsize=7)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()
ax2.margins(0.07)

autolabel(ax2,rects1,fdr_vid1_pvals[n_aoi1:], vid1_dvals[n_aoi1:], 'left')
autolabel(ax2,rects2,fdr_vid2_pvals[n_aoi1:], vid2_dvals[n_aoi1:], 'left')

ax1.legend(prop=dict(size=7,weight='normal'),frameon=False,
           handletextpad=0.3,handlelength=0.8) #,bbox_to_anchor=(0.32, 0.9))

leg = ax1.get_legend()
leg.legendHandles[0].set_color((0.14, 0.30, 0.43, 1))
leg.legendHandles[1].set_color((0.00, 0.57, 0.61, 1))

plt.savefig('Fig_Cohend_Compare.png', dpi=600)
plt.savefig('Fig_Cohend_Compare.svg', format='svg')
