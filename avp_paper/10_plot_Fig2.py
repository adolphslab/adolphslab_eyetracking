#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Different colors to individual subjects.

"""

import numpy as np
import os,sys
import pandas as pd
import seaborn as sns

import matplotlib.pylab as plt
#plt.rc('font', weight='semibold')
plt.rcParams['svg.fonttype'] = 'none'

from scipy.stats import pearsonr, spearmanr 

from alabeye.etutils import load_gazedata_episodes


#%% plotting functions
def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

def plot_scatters(ax, data_x, data_y, 
                  c=None,m=None,cline='k',
                  label_x=None, label_y=None, 
                  xy_min=None, xy_max=None, 
                  xticks=None, yticks=None,
                  add_text=None):
    
    data_pd_Ctrl = pd.DataFrame({'Clip1_data': data_x, 'Clip2_data': data_y })

    if xy_min is None:
        xy_min = np.min(np.r_[data_x, data_y])
        xy_min -= xy_min*0.1
    if xy_max is None:
        xy_max = np.max(np.r_[data_x, data_y])
        xy_max += xy_max*0.1
        
    ax.set_xlim([xy_min, xy_max])
    ax.set_ylim([xy_min, xy_max])
    
    ax = sns.regplot(x="Clip1_data", y="Clip2_data", data=data_pd_Ctrl, color= cline,
                     truncate=False , ax = ax, label='TD (N=%d)'%len(data_x), scatter_kws={'s':0, 'edgecolors':'none'}) 

    if add_text is not None:
        ax.text(xy_min+ (xy_min*0.05), xy_min+(xy_min*0.05), add_text, fontsize = 7)
    
    if label_x is not None:
        ax.set_xlabel(label_x, fontsize = 7)
    if label_y is not None:
        ax.set_ylabel(label_y, fontsize = 7)
    
    _ = mscatter(data_pd_Ctrl["Clip1_data"], data_pd_Ctrl["Clip2_data"], c=c, s=8, m=m, ax=ax)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
        
    ax.tick_params(axis='x', labelsize= 7)
    ax.tick_params(axis='y', labelsize= 7)
    
    # ax.legend(loc='upper left',frameon=True,edgecolor='none',handletextpad=0.3,handlelength=0.8, prop=dict(size=7), borderpad=0.05)
    ax.set_aspect(1./ax.get_data_ratio())
    
    return ax

#%% Settings

tbin = 600

# Main directory for experiment data
root_dir = '/home/umit/Documents/Research_ET/AutismVids/avp_data'
gazedata_dir = os.path.join(root_dir,'Results_v1','Cohend_pool')
behavioraldata_file = os.path.join(root_dir,'BehavioralData','participants_info.csv')

resampleddata_dir = os.path.join(root_dir, 'Results_v1', 'splitparts_data')


vidnames, asd_data_df, td_data_df, aa, bb = load_gazedata_episodes(gazedata_dir, behavioraldata_file)


vidclips_use = [ 'Episode A', 'Episode B']
colors_init = ['#db3926','#1f78b4'] 


columns_all = asd_data_df.columns.tolist()
assert np.array_equal(columns_all, td_data_df.columns.tolist())
featnames = np.unique([ cii.split('_')[0] for cii in columns_all ]).tolist()

asd_subjs = asd_data_df.index.to_numpy()
td_subjs = td_data_df.index.to_numpy()


asd_vid1_feat0 = asd_data_df['Onscreen_vid1'].to_numpy()*100
asd_vid2_feat0 = asd_data_df['Onscreen_vid2'].to_numpy()*100
td_vid1_feat0 = td_data_df['Onscreen_vid1'].to_numpy()*100
td_vid2_feat0 = td_data_df['Onscreen_vid2'].to_numpy()*100


asd_vid1_feat1 = asd_data_df['Face_vid1'].to_numpy()*100
asd_vid2_feat1 = asd_data_df['Face_vid2'].to_numpy()*100
td_vid1_feat1 = td_data_df['Face_vid1'].to_numpy()*100
td_vid2_feat1 = td_data_df['Face_vid2'].to_numpy()*100


asd_vid1_feat2 = asd_data_df['Eye_vid1'].to_numpy()*100
asd_vid2_feat2 = asd_data_df['Eye_vid2'].to_numpy()*100
td_vid1_feat2 = td_data_df['Eye_vid1'].to_numpy()*100
td_vid2_feat2 = td_data_df['Eye_vid2'].to_numpy()*100


asd_vid1_feat3 = asd_data_df['Heatcorr_vid1'].to_numpy()
asd_vid2_feat3 = asd_data_df['Heatcorr_vid2'].to_numpy()
td_vid1_feat3 = td_data_df['Heatcorr_vid1'].to_numpy()
td_vid2_feat3 = td_data_df['Heatcorr_vid2'].to_numpy()


#%%

# ------------- get colors for subjects based on gaze time to faces -------------
from matplotlib import colors
from scipy.stats import rankdata

cmap_asd = colors.LinearSegmentedColormap.from_list("", ['DarkRed', 'red', 'Crimson','IndianRed','Coral', 'Tomato', 
                                                         'OrangeRed', 'SandyBrown',"orange", 'Gold', 'Goldenrod'])

cmap_asd = colors.LinearSegmentedColormap.from_list("", ['DarkRed', '#db3926', 'SandyBrown', 'Gold',  'DarkOrange'])
# cmap_asd = cm.get_cmap('autumn')

cmap_td = colors.LinearSegmentedColormap.from_list("", ['MidnightBlue',"C0",'DeepSkyBlue','teal','Turquoise','cyan','C2'])
# cmap_td = cm.get_cmap('winter')

# # --- option 1 ---
# norm_asd = colors.Normalize(vmin=asd_vid1.min(), vmax=asd_vid1.max())
# asd_colors = [ cmap_asd(norm_asd(ii)) for ii in asd_vid1 ]

# norm_td = colors.Normalize(vmin=td_vid1.min(), vmax=td_vid1.max())
# td_colors = [ cmap_td(norm_td(ii)) for ii in td_vid1 ]

# --- option 2 ---
norm_asd = colors.Normalize(vmin=1, vmax=asd_vid1_feat1.size)
asd_vid1_rank = rankdata(asd_vid1_feat1)
asd_colors = [ cmap_asd(norm_asd(ii)) for ii in asd_vid1_rank ]

norm_td = colors.Normalize(vmin=1, vmax=td_vid1_feat1.size)
td_vid1_rank = rankdata(td_vid1_feat1)
td_colors = [ cmap_td(norm_td(ii)) for ii in td_vid1_rank ]
colorp = [ asd_colors, td_colors ]

# To bypass giving colors. 
# colorp = None

markerp_asd = [ 'v' if sii.startswith('RA') else 'o' for sii in asd_subjs ]
markerp_td = [ 'v' if sii.startswith('RA') else 'o' for sii in td_subjs ]
markerp = [ markerp_asd, markerp_td ]

# To bypass giving marker shapes. 
# markerp = None


#%% ----- Plotting starts here -----
fig, axes_all = plt.subplots(4, 3, sharey=False, sharex=False, figsize=(5.25,7.))
axes = axes_all[:,:2].ravel()

#%% Plot onscreen gaze time

feat0_corr_asd = pearsonr(asd_vid1_feat0, asd_vid2_feat0 )
feat0_corr_td = pearsonr(td_vid1_feat0, td_vid2_feat0 )

feat0_corr_asd_s = spearmanr(asd_vid1_feat0, asd_vid2_feat0 )
feat0_corr_td_s = spearmanr(td_vid1_feat0, td_vid2_feat0 )

add_text= "\nPearson's r=%.2f\n"%(feat0_corr_asd[0])+r"Spearman's $\rho$=%.2f"%(feat0_corr_asd_s[0])

xy_min, xy_max = [50,105]
xticks= [50,60,70,80,90,100]
yticks = xticks

plot_scatters(axes[0], asd_vid1_feat0, asd_vid2_feat0,
              c=colorp[0], m=markerp[0],cline=colors_init[0],
              label_x=f'On-screen gaze time\n{vidclips_use[0]} (%)', 
              label_y=f'On-screen gaze time\n{vidclips_use[1]} (%)',
              xy_min=xy_min, xy_max=xy_max,
              xticks=xticks,yticks=yticks,
              add_text=add_text)


from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='v', color='w', label='Cal', markerfacecolor='none', markeredgecolor='k', markersize=3,markeredgewidth=0.85),
                   Line2D([0], [0], marker='o', color='w', label='IU', markerfacecolor='none', markeredgecolor='k', markersize=3,markeredgewidth=0.85),]
axes[0].legend(handles=legend_elements,ncol=1, loc="upper left", borderaxespad=0.5,handletextpad=0.6,labelspacing=0.15,handlelength=0,columnspacing=1.5,
          frameon=False,prop={'size': 7})

add_text= "\nPearson's r=%.2f\n"%(feat0_corr_td[0])+r"Spearman's $\rho$=%.2f"%(feat0_corr_td_s[0])

plot_scatters(axes[1], td_vid1_feat0, td_vid2_feat0,
              c=colorp[1], m=markerp[1],cline=colors_init[1],
              label_x=f'On-screen gaze time\n{vidclips_use[0]} (%)', 
              label_y=f'On-screen gaze time\n{vidclips_use[1]} (%)',
              xy_min=xy_min, xy_max=xy_max,
              xticks=xticks,yticks=yticks,
              add_text=add_text)


#%%
feat1_corr_asd = pearsonr(asd_vid1_feat1, asd_vid2_feat1 )
feat1_corr_td = pearsonr(td_vid1_feat1, td_vid2_feat1 )

feat1_corr_asd_s = spearmanr(asd_vid1_feat1, asd_vid2_feat1 )
feat1_corr_td_s = spearmanr(td_vid1_feat1, td_vid2_feat1 )

add_text= "\nPearson's r=%.2f\n"%(feat1_corr_asd[0])+r"Spearman's $\rho$=%.2f"%(feat1_corr_asd_s[0])

xy_min, xy_max = [50,90]
xticks= [50,60,70,80,90]
yticks = xticks

plot_scatters(axes[2], asd_vid1_feat1, asd_vid2_feat1,
              c=colorp[0], m=markerp[0],cline=colors_init[0],
              label_x=f'Gaze time to faces\n{vidclips_use[0]} (%)', 
              label_y=f'Gaze time to faces\n{vidclips_use[1]} (%)',
              xy_min=xy_min, xy_max=xy_max,
              xticks=xticks,yticks=yticks,
              add_text=add_text)


add_text= "\nPearson's r=%.2f\n"%(feat1_corr_td[0])+r"Spearman's $\rho$=%.2f"%(feat1_corr_td_s[0])

plot_scatters(axes[3], td_vid1_feat1, td_vid2_feat1,
              c=colorp[1], m=markerp[1],cline=colors_init[1],
              label_x=f'Gaze time to faces\n{vidclips_use[0]} (%)', 
              label_y=f'Gaze time to faces\n{vidclips_use[1]} (%)',
              xy_min=xy_min, xy_max=xy_max,
              xticks=xticks,yticks=yticks,
              add_text=add_text)
    
    


#%% 
feat2_corr_asd = pearsonr(asd_vid1_feat2, asd_vid2_feat2 )
feat2_corr_td = pearsonr(td_vid1_feat2, td_vid2_feat2 )

feat2_corr_asd_s = spearmanr(asd_vid1_feat2, asd_vid2_feat2 )
feat2_corr_td_s = spearmanr(td_vid1_feat2, td_vid2_feat2 )    

xy_min, xy_max = [0,90]
xticks = [0,20,40,60,80]
yticks = xticks

add_text= "\nPearson's r=%.2f\n"%(feat2_corr_asd[0])+r"Spearman's $\rho$=%.2f"%(feat2_corr_asd_s[0])

plot_scatters(axes[4], asd_vid1_feat2, asd_vid2_feat2,
              c=colorp[0], m=markerp[0], cline=colors_init[0],
              label_x=f'Gaze time to eyes\n{vidclips_use[0]} (%)',
              label_y=f'Gaze time to eyes\n{vidclips_use[1]} (%)',
              xy_min=xy_min, xy_max=xy_max,
               xticks=xticks,yticks=yticks,
              add_text=add_text)


add_text= "\nPearson's r=%.2f\n"%(feat2_corr_td[0])+r"Spearman's $\rho$=%.2f"%(feat2_corr_td_s[0])

plot_scatters(axes[5], td_vid1_feat2, td_vid2_feat2,
              c=colorp[1], m=markerp[1], cline=colors_init[1],
              label_x=f'Gaze time to eyes\n{vidclips_use[0]} (%)',
              label_y=f'Gaze time to eyes\n{vidclips_use[1]} (%)',
              xy_min=xy_min, xy_max=xy_max,
               xticks=xticks,yticks=yticks,
              add_text=add_text)

    
#%% Heatmap correlation part
feat3_corr_asd = pearsonr(asd_vid1_feat3, asd_vid2_feat3 )
feat3_corr_td = pearsonr(td_vid1_feat3, td_vid2_feat3 )

feat3_corr_asd_s = spearmanr(asd_vid1_feat3, asd_vid2_feat3 )
feat3_corr_td_s = spearmanr(td_vid1_feat3, td_vid2_feat3 )    

xy_min, xy_max = [0.2,0.85]
xticks = [0.2,0.4,0.6,0.8]
yticks = xticks

add_text= "\nPearson's r=%.2f\n"%(feat3_corr_asd[0])+r"Spearman's $\rho$=%.2f"%(feat3_corr_asd_s[0])

plot_scatters(axes[6], asd_vid1_feat3, asd_vid2_feat3,
              c=colorp[0], m=markerp[0], cline=colors_init[0],
              label_x=f'Heatmap correlation\n{vidclips_use[0]}', 
              label_y=f'Heatmap correlation\n{vidclips_use[1]}',
              xy_min=xy_min, xy_max=xy_max,
              xticks=xticks,yticks=yticks,
              add_text=add_text)


add_text= "\nPearson's r=%.2f\n"%(feat3_corr_td[0])+r"Spearman's $\rho$=%.2f"%(feat3_corr_td_s[0])

plot_scatters(axes[7], td_vid1_feat3, td_vid2_feat3,
              c=colorp[1], m=markerp[1], cline=colors_init[1],
              label_x=f'Heatmap correlation\n{vidclips_use[0]}', 
              label_y=f'Heatmap correlation\n{vidclips_use[1]}',
              xy_min=xy_min, xy_max=xy_max,
              xticks=xticks,yticks=yticks,
              add_text=add_text)

#%%

feat0_corrs_asd_s = np.load(os.path.join(resampleddata_dir,f'feat0_corrs_asd_{tbin}_s.npy'))
feat0_corrs_td_s = np.load(os.path.join(resampleddata_dir,f'feat0_corrs_td_{tbin}_s.npy'))

feat1_corrs_asd_s = np.load(os.path.join(resampleddata_dir,f'feat1_corrs_asd_{tbin}_s.npy'))
feat1_corrs_td_s = np.load(os.path.join(resampleddata_dir,f'feat1_corrs_td_{tbin}_s.npy'))

feat2_corrs_asd_s = np.load(os.path.join(resampleddata_dir,f'feat2_corrs_asd_{tbin}_s.npy'))
feat2_corrs_td_s = np.load(os.path.join(resampleddata_dir,f'feat2_corrs_td_{tbin}_s.npy'))

feat3_corrs_asd_s = np.load(os.path.join(resampleddata_dir,f'feat3_corrs_asd_{tbin}_s.npy'))
feat3_corrs_td_s = np.load(os.path.join(resampleddata_dir,f'feat3_corrs_td_{tbin}_s.npy'))

feat0_corrs_diff_s = np.load(os.path.join(resampleddata_dir,f'feat0_corrs_diff_{tbin}_s.npy'))
feat1_corrs_diff_s = np.load(os.path.join(resampleddata_dir,f'feat1_corrs_diff_{tbin}_s.npy'))
feat2_corrs_diff_s = np.load(os.path.join(resampleddata_dir,f'feat2_corrs_diff_{tbin}_s.npy'))
feat3_corrs_diff_s = np.load(os.path.join(resampleddata_dir,f'feat3_corrs_diff_{tbin}_s.npy'))

#%% Plotting

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(3,5),sharex=False,sharey=True)
axes = axes_all[:,2].ravel()

# Add vertical histograms:
nbins = 80
data_sets_feat0 = [ feat0_corrs_asd_s, feat0_corrs_td_s, feat0_corrs_diff_s ]
data_sets_feat1 = [ feat1_corrs_asd_s, feat1_corrs_td_s, feat1_corrs_diff_s ]
data_sets_feat2 = [ feat2_corrs_asd_s, feat2_corrs_td_s, feat2_corrs_diff_s ]
data_sets_feat3 = [ feat3_corrs_asd_s, feat3_corrs_td_s, feat3_corrs_diff_s ]

hist_range = (np.min(data_sets_feat0+data_sets_feat1+data_sets_feat2+data_sets_feat3), 
              np.max(data_sets_feat0+data_sets_feat1+data_sets_feat2+data_sets_feat3))

binned_data_sets_feat0 = [ np.histogram(d,range=hist_range,bins=nbins)[0] for d in data_sets_feat0 ]
binned_data_sets_feat1 = [ np.histogram(d,range=hist_range,bins=nbins)[0] for d in data_sets_feat1 ]
binned_data_sets_feat2 = [ np.histogram(d,range=hist_range,bins=nbins)[0] for d in data_sets_feat2 ]
binned_data_sets_feat3 = [ np.histogram(d,range=hist_range,bins=nbins)[0] for d in data_sets_feat3 ]
binned_maxs = np.max(binned_data_sets_feat0+binned_data_sets_feat1+binned_data_sets_feat2+binned_data_sets_feat3, axis=1)

x_locs = [0,1,2]

colors = ['#db3926','#1f78b4', 'gray']
 
bin_edges = np.linspace(hist_range[0], hist_range[1], nbins + 1) # or get directly from np.histogram(...)[1]
centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
heights = np.diff(bin_edges)
for c_loc, x_loc in enumerate(x_locs):
    # axes[c_loc].barh(centers, binned_data_sets_face[c_loc],color=colors[c_loc], height=heights)#,left=x_loc)
    # axes[c_loc].hist(data_sets_face[c_loc], bins=nbins, range=hist_range, orientation="horizontal");
    
    for ds_ii, (data_set,data_set_bin) in enumerate(zip([data_sets_feat0, data_sets_feat1, data_sets_feat2, data_sets_feat3],\
                                    [binned_data_sets_feat0, binned_data_sets_feat1, binned_data_sets_feat2, binned_data_sets_feat3])): 

        dii = data_set[c_loc]
        dii_mean = dii.mean()
        ci_vals = np.asarray([ dii_mean - np.percentile(dii,2.5), np.percentile(dii,97.5) - dii_mean ]).reshape(2,1)

    
        if c_loc in [0,1,2]:
            axes[ds_ii].barh(centers, data_set_bin[c_loc]/binned_maxs.max()*1.,
                             color=colors[c_loc], height=heights,left=x_loc,alpha=0.85)
    
            axes[ds_ii].errorbar(x_loc, dii.mean(), yerr=ci_vals, marker='.',
                                 color='k',capsize=0,elinewidth=1.1,capthick=1)


for aii, ax in enumerate(axes):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_yaxis().tick_left()
    
    ax.axhline(0, color='0', linestyle='-', linewidth = 0.85, zorder=200) #2ecc71      
    ax.set_ylabel("Correlation\n"+r"(Spearman's $\rho$)", fontsize=7)
    ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.tick_params(axis='y', labelsize= 7)

    ax.get_xaxis().tick_bottom()
    x_lims = ax.get_xlim()
    ax.set_xlim([-0.2,x_lims[1]])

    y_lims = ax.get_ylim()
    ax.hlines(y_lims[0], 0-0.001, 2+0.01,'k')
    ax.set_ylim(y_lims)

    ax.set_xticks(x_locs)
    ax.set_xticklabels(['ASD', 'TD', 'Difference'],fontsize=7)


plt.subplots_adjust( hspace=0.5, wspace=0.6 )
# plt.tight_layout()

plt.savefig('CorrsStats_4feats_v1.png', dpi=300, bbox_inches='tight')
plt.savefig('CorrsStats_4feats_v1.svg', format='svg')


