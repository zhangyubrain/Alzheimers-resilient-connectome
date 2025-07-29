# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 00:07:12 2024

@author: 99488
"""
'''
result with age corrected subjects from adni, then cross validation in subject level were obtained, combat all subjects
'''

import os
import pandas as pd
import numpy as np
import scipy.io as sio
from pathlib import Path
from bids.layout import parse_file_entities
import sys
sys.path.append(r'F:\OPT\research\fMRI\utils_for_all')
from common_utils import read_singal_fmri_ts, get_fc, t_test, keep_triangle_half, sens_spec, setup_seed, vector_to_matrix, heatmap, get_net_net_connect, diverge_map_unipolar_colore
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import LinearSVC, SVC, OneClassSVM
from sklearn.metrics import accuracy_score, r2_score, auc, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import xgboost as xgb
from mrmr import mrmr_classif
from neuroCombat import neuroCombat
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import copy 
from scipy.stats import pearsonr, chi2_contingency, shapiro, kstest, iqr, ranksums, spearmanr
import matplotlib
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import namedtuple, OrderedDict
from matplotlib import colors as mcolors
from statsmodels.stats.multitest import multipletests
from scipy.spatial.distance import pdist
import random
from matplotlib.collections import PatchCollection
from statsmodels.formula.api import ols
import statsmodels.api as sm


font2 = {'family': 'Tahoma', 'weight': 'bold', 'size': 30}
matplotlib.rc('font', **font2)
setup_seed(6)

Color = namedtuple('RGB', 'red, green, blue')
colors = {}  # dict of colors
class RGB(Color):
    def hex_format(self):
        return '#{:02X}{:02X}{:02X}'.format(self.red, self.green, self.blue)

# Color Contants
red = RGB(0.941, 0.502, 0.498)
blue = RGB(0.388, 0.584, 0.933)

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def diverge_map(high=(0.565, 0.392, 0.173), low=(0.094, 0.310, 0.635)):
    '''
    low and high are colors that will be used for the two
    ends of the spectrum. they can be either color strings
    or rgb color tuples
    '''
    c = mcolors.ColorConverter().to_rgb
    if isinstance(low, str): low = c(low)
    if isinstance(high, str): high = c(high)
    return make_colormap([low, c('white'), 0.5, c('white'), high])

###########################dementia:1, SA: 0, CN:2, MCI: 3
info = sio.loadmat(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\superage_data_adni.mat')
sa_dem_fc_adni = info['fc_sa_dem']
sa_dem_dx_adni = np.squeeze(info['dx_sa_dem'])
sa_dem_age_adni = np.squeeze(info['age_sa_dem'])
sa_dem_sex_adni = np.squeeze(info['sex_sa_dem'])
sa_dem_sex_adni = np.array([2 if sex == 'Female'  else 1 for sex in sa_dem_sex_adni])
sa_dem_sub_adni = np.squeeze(info['sub_id_sa_dem'])
sa_dem_sess_adni = np.squeeze(info['sess_sa_dem'])
df=pd.DataFrame(np.c_[sa_dem_sub_adni, sa_dem_sess_adni],columns=['sub','sess'])
df['sub_sess']=df['sub']+'_'+df['sess']
sa_dem_sub_sess_adni = df['sub_sess'].values

half_fc_adni = keep_triangle_half(sa_dem_fc_adni.shape[1] * (sa_dem_fc_adni.shape[1]-1)//2, sa_dem_fc_adni.shape[0], sa_dem_fc_adni)

all_fc_adni = info['adni_fc_all']
all_dx_adni = np.squeeze(info['adni_dx_all'])
all_age_adni = np.squeeze(info['adni_age_all'])
all_sex_adni = np.squeeze(info['adni_sex_all'])
# all_sex_adni = np.array([2 if sex == 'Female' else 1 for sex in all_sex_adni])
all_sub_adni = np.squeeze(info['adni_sub_all'])
all_sess_adni = np.squeeze(info['adni_sess_all'])

other_adni_sub = np.setdiff1d(all_sub_adni, sa_dem_sub_adni)
other_fc_adni = []
other_dx_adni = []
other_age_adni = []
other_sex_adni = []
other_sess_adni = []
other_sub_adni = []
for i,sub in enumerate(other_adni_sub):
    other_fc_adni.extend(list(all_fc_adni[all_sub_adni==sub]))
    other_dx_adni.extend(list(all_dx_adni[all_sub_adni==sub]))
    other_age_adni.extend(list(all_age_adni[all_sub_adni==sub]))
    other_sex_adni.extend(list(all_sex_adni[all_sub_adni==sub]))
    other_sess_adni.extend(list(all_sess_adni[all_sub_adni==sub]))
    other_sub_adni.extend(list(all_sub_adni[all_sub_adni==sub]))
other_fc_adni = np.array(other_fc_adni)    
other_dx_adni = np.array(other_dx_adni)    
other_age_adni = np.array(other_age_adni)    
other_sex_adni = np.array(other_sex_adni)    
other_sess_adni = np.array(other_sess_adni)    
other_sub_adni = np.array(other_sub_adni) 
   
low_qua_fc_idx = np.where(np.isnan(np.sum(np.sum(other_fc_adni,-1),-1)))[0]
other_fc_adni = np.delete(other_fc_adni, low_qua_fc_idx, 0)
other_dx_adni = np.delete(other_dx_adni, low_qua_fc_idx, 0)
other_age_adni = np.delete(other_age_adni, low_qua_fc_idx, 0)
other_sex_adni = np.delete(other_sex_adni, low_qua_fc_idx, 0)
other_sess_adni = np.delete(other_sess_adni, low_qua_fc_idx, 0)
other_sub_adni = np.delete(other_sub_adni, low_qua_fc_idx, 0)
df=pd.DataFrame(np.c_[other_sub_adni, other_sess_adni],columns=['sub','sess'])
df['sub_sess']=df['sub']+'_'+df['sess']
other_sub_sess_adni = df['sub_sess'].values

for i in range(len(other_sex_adni)):
    if 'F' in other_sex_adni[i]:
        other_sex_adni[i] = 2
    elif 'M' in other_sex_adni[i]:
        other_sex_adni[i] = 1
    if 'CN' in other_dx_adni[i]:
        other_dx_adni[i] = 2
    elif 'MCI' in other_dx_adni[i]:
        other_dx_adni[i] = 3
    elif 'Dementia' in other_dx_adni[i]:
        other_dx_adni[i] = 1
other_dx_adni = other_dx_adni.astype(float)        
other_sex_adni = other_sex_adni.astype(float)   

info_oas = sio.loadmat(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\superage_data_oas_dem.mat')
sa_dem_fc_oas = info_oas['multi_run_fcs_sa_dem']
sa_dem_dx_oas = np.squeeze(info_oas['multi_run_dx_sa_dem'])
sa_dem_sub_oas_ = np.squeeze(info_oas['multi_run_subs_sa_dem'])
sa_dem_sess_oas = np.squeeze(info_oas['multi_run_sess_sa_dem'])
all_pcd_oas = np.squeeze(info_oas['multi_run_pcd_all'])
sa_dem_sub_oas = pd.Series(sa_dem_sub_oas_).str.cat(pd.Series(sa_dem_sess_oas), sep = '_').values
sex_oas = np.squeeze(info_oas['sex_sa_dem'])
pcd_oas = info_oas['pcd_sa_dem']
sess_oas = info_oas['sess_sa_dem']
age_oas = pcd_oas[:,2].astype(float).squeeze()
sub_oas = pcd_oas[:,0]
sub_oas = np.array([sub[0].split('_')[0] for sub in sub_oas])
oasis_path = r'E:\PHD\learning\research\AD\data\OASIS3'
oasis_demo = pd.read_csv(os.path.join(oasis_path, 'UDS-A2-InformatDemos.csv'))
oasis_demo2 = pd.read_csv(os.path.join(oasis_path, 'OASIS3_UDSa2_cs_demo.csv'))
oasis_demo_sub = oasis_demo['Subject']
oasis_demo_sub2 = oasis_demo2['OASISID']
sa_dem_sex_oas = []
sa_dem_age_oas = []
for i, sub in enumerate(sa_dem_sub_oas_):
    # sa_dem_sex_oas.append(sex_oas[(sub_oas==sub)&(sess_oas == sa_dem_sess_oas[i])])
    if sum(oasis_demo_sub == sub) >= 1:
        sa_dem_sex_oas.append(oasis_demo['INSEX'][oasis_demo_sub == sub].values[0])
    else:
        if sum(oasis_demo_sub2 == sub) >= 1:
            sa_dem_sex_oas.append(oasis_demo2['INSEX'][oasis_demo_sub2 == sub].values[0])
    sa_dem_age_oas.append(age_oas[(sub_oas==sub)&(sess_oas == sa_dem_sess_oas[i])])
sa_dem_sex_oas = np.array(sa_dem_sex_oas).squeeze() # female: 2; male: 1
sa_dem_age_oas = np.array(sa_dem_age_oas).squeeze()

all_fc_oas = info_oas['multi_run_fcs_all']
all_sub_oas = np.squeeze(info_oas['multi_run_subs_all'])
all_sess_oas = np.squeeze(info_oas['multi_run_sess_all'])
all_pcd_oas = np.squeeze(info_oas['multi_run_pcd_all'])
all_sex_oas = np.squeeze(info_oas['multi_run_sex_all'])
all_age_oas = all_pcd_oas[:,2].astype(float).squeeze()
all_dx_oas = all_pcd_oas[:,1]

other_oas_sub = np.setdiff1d(all_sub_oas, sa_dem_sub_oas)
other_fc_oas = []
other_dx_oas = []
other_age_oas = []
other_sex_oas = []
other_sess_oas = []
other_sub_oas_multises = []
for i,sub in enumerate(other_oas_sub):
    other_fc_oas.extend(list(all_fc_oas[all_sub_oas==sub]))
    other_dx_oas.extend([s[0][0] if isinstance(s[0], np.ndarray) else s[0] for s in all_dx_oas[all_sub_oas==sub]])
    other_age_oas.extend(list(all_age_oas[all_sub_oas==sub]))
    other_sex_oas.extend(list(all_sex_oas[all_sub_oas==sub]))
    other_sess_oas.extend(list(all_sess_oas[all_sub_oas==sub]))
    other_sub_oas_multises.extend(list(all_sub_oas[all_sub_oas==sub]))
other_fc_oas = np.array(other_fc_oas)    
other_dx_oas = np.array(other_dx_oas)    
other_age_oas = np.array(other_age_oas)    
other_sex_oas = np.array(other_sex_oas)    
other_sess_oas = np.array(other_sess_oas)   
other_sub_oas_multises = np.array(other_sub_oas_multises)   
mask_nc = pd.Series(other_dx_oas).str.contains('normal', regex=False)
mask_ad = (pd.Series(other_dx_oas).str.contains('AD', regex=False)) |(pd.Series(other_dx_oas).str.contains('DAT', regex=False)) | \
    (pd.Series(other_dx_oas)=='Frontotemporal demt. prim') | (pd.Series(other_dx_oas)=='DLBD, primary') | (pd.Series(other_dx_oas).str.contains('primary', regex=False)) | \
        (pd.Series(other_dx_oas)=='Vascular Demt, secondary') | (pd.Series(other_dx_oas)=='uncertain dementia') | \
            (pd.Series(other_dx_oas)=='uncertain, possible NON AD dem') | (pd.Series(other_dx_oas)=='DLBD, primary')
mask_mci = pd.Series(other_dx_oas).str.contains('0.5 in memory', regex=False) | (pd.Series(other_dx_oas)=='Unc: ques. Impairment')
other_dx_oas_encode = np.zeros((len(other_dx_oas)))
other_dx_oas_encode[mask_nc] = 2
other_dx_oas_encode[mask_mci] = 3
other_dx_oas_encode[mask_ad] = 1
 
low_qua_fc_idx = np.where(np.isnan(np.sum(np.sum(other_fc_oas,-1),-1)))[0]
other_fc_oas = np.delete(other_fc_oas, low_qua_fc_idx, 0)
other_dx_oas_encode_raw = np.delete(other_dx_oas_encode, low_qua_fc_idx, 0)
other_age_oas_raw = np.delete(other_age_oas, low_qua_fc_idx, 0)
other_sex_oas_raw = np.delete(other_sex_oas, low_qua_fc_idx, 0)
other_sess_oas_raw = np.delete(other_sess_oas, low_qua_fc_idx, 0)
other_sub_oas_multises = np.delete(other_sub_oas_multises, low_qua_fc_idx, 0)

half_fc_oas_sa_dem = keep_triangle_half(sa_dem_fc_oas.shape[1] * (sa_dem_fc_oas.shape[1]-1)//2, sa_dem_fc_oas.shape[0], sa_dem_fc_oas)
half_fc_oas_other = keep_triangle_half(other_fc_oas.shape[1] * (other_fc_oas.shape[1]-1)//2, other_fc_oas.shape[0], other_fc_oas)
half_fc_adni_other = keep_triangle_half(other_fc_adni.shape[1] * (other_fc_adni.shape[1]-1)//2, other_fc_adni.shape[0], other_fc_adni)

info_habs = sio.loadmat(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\superage_data_habs.mat')
sa_pcd_habs = info_habs['multi_run_pcd_sa'].squeeze()
sa_fc_habs = info_habs['multi_run_fcs_sa'][:,:100,:100]
sa_sub_sess_habs = np.array([sub[0] for sub in sa_pcd_habs[:,-1].squeeze()])
sa_sex_habs = np.array([2 if sex[0] == 'F'  else 1 for sex in sa_pcd_habs[:,59]])
sa_age_habs = np.array([sub[0][0] for sub in sa_pcd_habs[:,58].squeeze()]) 
sa_sex_habs = sa_sex_habs.astype(float)
sa_dx_habs = np.zeros((len(sa_sex_habs)))
half_fc_habs = keep_triangle_half(sa_fc_habs.shape[1] * (sa_fc_habs.shape[1]-1)//2, sa_fc_habs.shape[0], sa_fc_habs)
sa_sub_habs = pd.Series(sa_sub_sess_habs).str.split('_', expand = True)[0]

habs_pcd_all = info_habs['multi_run_pcd_all'].squeeze()
habs_fc_all = info_habs['multi_run_fc_all'][:,:100,:100]
habs_sub_sess_all = np.array([sub[0] for sub in habs_pcd_all[:,-1].squeeze()])
habs_sex_all = np.array([2 if sex[0] == 'F'  else 1 for sex in habs_pcd_all[:,59]])
habs_age_all = np.array([sub[0][0] for sub in habs_pcd_all[:,58].squeeze()]) 
habs_sex_all = habs_sex_all.astype(float)
habs_dx_all = np.array([sub[0] for sub in habs_pcd_all[:,67].squeeze()]) 
habs_dx_all[habs_dx_all == 'CN'] = 2
habs_dx_all[habs_dx_all == 'MCI'] = 3
habs_dx_all[habs_dx_all == 'Dementia'] = 1
habs_dx_all = habs_dx_all.astype(float)
half_habs_fc_all = keep_triangle_half(habs_fc_all.shape[1] * (habs_fc_all.shape[1]-1)//2, habs_fc_all.shape[0], habs_fc_all)
habs_sub_all = pd.Series(habs_sub_sess_all).str.split('_', expand = True)[0]

other_habs_sub = np.setdiff1d(habs_sub_sess_all, sa_sub_sess_habs)
other_fc_habs = []
other_dx_habs = []
other_age_habs = []
other_sex_habs = []
for i,sub in enumerate(other_habs_sub):
    other_fc_habs.extend(list(half_habs_fc_all[habs_sub_sess_all==sub]))
    other_dx_habs.extend(list(habs_dx_all[habs_sub_sess_all==sub]))
    other_age_habs.extend(list(habs_age_all[habs_sub_sess_all==sub]))
    other_sex_habs.extend(list(habs_sex_all[habs_sub_sess_all==sub]))
other_fc_habs = np.array(other_fc_habs)    
other_dx_habs = np.array(other_dx_habs)    
other_age_habs = np.array(other_age_habs)    
other_sex_habs = np.array(other_sex_habs)    

half_fc_ind = np.r_[half_fc_oas_sa_dem, half_fc_habs]
age_ind = np.r_[sa_dem_age_oas, sa_age_habs]
sex_ind = np.r_[sa_dem_sex_oas, sa_sex_habs]
dx_ind = np.r_[sa_dem_dx_oas, sa_dx_habs]
sub_ind = np.r_[sa_dem_sub_oas, sa_sub_sess_habs]
sub_ind_ = np.r_[sa_dem_sub_oas_, sa_sub_habs]

site_ind = np.r_[np.zeros(len(sa_dem_sub_oas)), sa_dx_habs+20]

subject_used, uni_id = np.unique(sub_ind, return_index=True)

print('Counts of raw subjects: {}, superager: {}, dementia: {}'.format(len(set(sub_ind)), sum(dx_ind[uni_id]==0), sum(dx_ind[uni_id]==1)))
print('Counts of used runs: {}, superager: {}, dementia: {}'.format(len(sub_ind), sum(dx_ind==0), sum(dx_ind==1)))

scaler = StandardScaler()
half_fc_adni = scaler.fit_transform(half_fc_adni.T).T
half_fc_ind = scaler.fit_transform(half_fc_ind.T).T
half_fc_oas_other = scaler.fit_transform(half_fc_oas_other.T).T
half_fc_adni_other = scaler.fit_transform(half_fc_adni_other.T).T
other_fc_habs = scaler.fit_transform(other_fc_habs.T).T

df=pd.DataFrame(np.c_[other_sub_oas_multises, other_sess_oas_raw],columns=['sub','sess'])
df['sub_sess']=df['sub']+'_'+df['sess']
sub_sess_oas = df['sub_sess'].values
sub_combine_uni_oas, uni_id_oas = np.unique(df['sub_sess'], return_index=True)
#################averaging runs
half_fc_oas_other_uni = []
for i, sub in enumerate(sub_combine_uni_oas):
    mask = df['sub_sess'].values == sub
    half_fc_oas_other_uni.append(half_fc_oas_other[mask].mean(0))
half_fc_oas_other_uni = np.array(half_fc_oas_other_uni)
other_dx_oas_encode = other_dx_oas_encode_raw[uni_id_oas]
other_age_oas = other_age_oas_raw[uni_id_oas]
other_sex_oas = other_sex_oas_raw[uni_id_oas]

#####################redefine MCI based on ADNI criteria
oasis_sa_sub_sess_uniq, oasis_sa_sub_sess_idx, oasis_sa_sub_sess_idx_reverse = np.unique(sub_combine_uni_oas, return_index=True, return_inverse=True)

oas_sess_info = sio.loadmat(r'F:\PHD\learning\project\super_age\sess_match_oas_all.mat')
day_all = oas_sess_info['day']
sess_all = oas_sess_info['sess']
idx = np.argsort(sess_all)
sess_all = sess_all[idx]
day_all = pd.Series(day_all[idx]).str.split('d', expand = True).iloc[:,-1].astype(int)

oas_pcd_info = sio.loadmat(r'E:\PHD\learning\research\AD\data\OASIS3\fMRIdata_OASIS3_struct.mat')
tvar = oas_pcd_info['OASIS3_Phenotypic_struct']
PCD_data = []
pcd_keys = ['subjectID_Date', 'apoe', 'NPIQINF', 'NPIQINFX', 'DEL',
              'DELSEV', 'HALL', 'HALLSEV', 'AGIT', 'AGITSEV', 'DEPD', 'DEPDSEV', 'ANX', 
              'ANXSEV', 'ELAT', 'ELATSEV', 'APA', 'APASEV', 'DISN', 'DISNSEV', 'IRR', 
              'IRRSEV', 'MOT', 'MOTSEV', 'NITE', 'NITESEV', 'APP', 'APPSEV', 'mmse',
              'cdr', 'commun','homehobb', 'judgment', 'memory', 'orient', 'perscare', 'sumbox', 
              'DIGIF', 'DIGIB', 'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB',
              'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON']
for idx in range(len(tvar)):
    sub_data = []
    for name in pcd_keys:
        value = tvar[idx][0][name]
        for i in range(3):
            if isinstance(value, np.ndarray):
                try:
                    value = value[0]
                except IndexError:
                    if name == 'DX':
                        value = 'nan'
                    else:
                        value = "error"
            else:
                if name == 'subjectID_Date' or name == 'dx1':
                    value = str(value)
                sub_data.append(value)
                break
    PCD_data.append(sub_data)
PCD_data = np.array(PCD_data)
PCD_data[:,1] = pd.Series(PCD_data[:,1]).str.count('4')
#update NPI 0 score
for i in [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]:
    mask =  PCD_data[:, i] == 'nan'
    PCD_data[np.isfinite(PCD_data[:,2].astype(float))&mask, i] = 0
PCD_data[np.isfinite(PCD_data[:,2].astype(float))][:,[5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27]] = 0

PCD_sub = np.array([sub[:8] for sub in PCD_data[:,0]])
PCD_day = np.array([sub[10:] for sub in PCD_data[:,0]]).astype(int)
PCD_sess = []
day_min = []
for da in PCD_day:
    mask = day_all==da
    if sum(mask) > 0:
        PCD_sess.append(sess_all[day_all==da][0])
    else:
        diff = day_all-da
        diff_min_id = np.argmin(abs(diff))
        diff_min = diff[diff_min_id]
        day_min.append(diff_min)
        PCD_sess.append(sess_all[diff_min_id])
PCD_sess = np.array(PCD_sess)
PCD_interest = []
for sub_sess in oasis_sa_sub_sess_uniq:
    sub = sub_sess[:8]
    sess = sub_sess[9:]
    if sum((PCD_sub == sub) & (PCD_sess == sess))== 1:
        PCD_interest.append(PCD_data[:,np.r_[[1, 2, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25], np.arange(27,50)]][
            (PCD_sub == sub) & (PCD_sess == sess)].squeeze())
    elif sum((PCD_sub == sub) & (PCD_sess == sess))> 1:
        pcd = PCD_data[:,np.r_[[1, 2, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25], np.arange(27,50)]][
            (PCD_sub == sub) & (PCD_sess == sess)].squeeze()
        pcd_new = []
        for i in range(pcd.shape[-1]):
            info = pcd[:,i]
            if np.isnan(info.astype(float)).sum() == len(info):
                pcd_new.append(np.nan)
            else:
                pcd_new.append(info[~np.isnan(info.astype(float))][0].squeeze())
        PCD_interest.append(np.array(pcd_new))
    else:
        PCD_interest.append(np.array([np.nan for i in range(len(['apoe', 'NPIQINF', 'DELSEV', 'HALLSEV', 'AGITSEV', 'DEPDSEV', 
                                              'ANXSEV', 'ELATSEV', 'APASEV', 'DISNSEV', 'IRRSEV', 'MOTSEV', 'NITESEV',
                                              'APPSEV', 'mmse', 'cdr', 'commun','homehobb', 'judgment', 'memory', 'orient', 'perscare', 'sumbox', 
                                              'DIGIF', 'DIGIB', 'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB',
                                              'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON']))]))
PCD_interest = np.array(PCD_interest)

df=pd.DataFrame(np.c_[other_sub_adni, other_sess_adni],columns=['sub','sess'])
df['sub_sess']=df['sub']+'_'+df['sess']
sub_combine_uni, uni_id = np.unique(df['sub_sess'], return_index=True)
half_fc_adni_other_uni = []
for i, sub in enumerate(sub_combine_uni):
    mask = df['sub_sess'].values == sub
    half_fc_adni_other_uni.append(half_fc_adni_other[mask].mean(0))
half_fc_adni_other_uni = np.array(half_fc_adni_other_uni)
other_dx_adni = other_dx_adni[uni_id]
other_sub_sess_adni = other_sub_sess_adni[uni_id]
other_age_adni = other_age_adni[uni_id]
other_age_adni[other_age_adni == 0] = np.nan
other_sex_adni = other_sex_adni[uni_id]
other_sub_adni = other_sub_adni[uni_id]
other_sess_adni[other_sess_adni == 'bl                              '] = 'm000'
other_sess_adni = other_sess_adni[uni_id]

harmonized_fc = np.r_[half_fc_ind, half_fc_adni, half_fc_adni_other_uni, half_fc_oas_other_uni, other_fc_habs]
harmonized_dx = np.r_[dx_ind, sa_dem_dx_adni, other_dx_adni, other_dx_oas_encode, other_dx_habs]
harmonized_age = np.r_[age_ind, sa_dem_age_adni, other_age_adni, other_age_oas, other_age_habs]
harmonized_age[np.isnan(harmonized_age)] = np.nanmean(harmonized_age)
harmonized_sex = np.r_[sex_ind, sa_dem_sex_adni, other_sex_adni, other_sex_oas, other_sex_habs]
harmonized_site = np.r_[site_ind, np.zeros(len(sa_dem_sex_adni))+2, np.zeros(len(other_sex_adni))+2, np.zeros(len(other_sex_oas)), np.zeros((len(other_sex_habs)))+20]
harmonized_dx[harmonized_dx==1]=4
covars = {'Age': harmonized_age, 'Sex':harmonized_sex, 'Site': harmonized_site, 'DX': harmonized_dx}
covars = pd.DataFrame(covars)

categorical_cols = ['DX']
continuous_cols = ['Age']
batch_col = 'Site'

##############demo, you should determine whether the covariate should be taked into consideration, in our case
##############kept age, DX information will be the best
combat = neuroCombat(dat=harmonized_fc.T,
    covars=covars,
    batch_col=batch_col,
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols)

data_combat = combat["data"].T
half_fc_ind = data_combat[:len(site_ind)]
half_fc_adni_correct = data_combat[len(site_ind):len(sa_dem_sex_adni)+len(site_ind)]
half_fc_adni_other_correct = data_combat[len(sa_dem_sex_adni)+len(site_ind): len(sa_dem_sex_adni)+len(site_ind)+len(other_sex_adni)]
half_fc_oas_other_correct = data_combat[len(sa_dem_sex_adni)+len(site_ind)+len(other_sex_adni): len(sa_dem_sex_adni)+len(site_ind)+len(other_sex_adni)+len(other_sex_oas)]
half_fc_habs_other_correct = data_combat[len(sa_dem_sex_adni)+len(site_ind)+len(other_sex_adni)+len(other_sex_oas):
                                         len(sa_dem_sex_adni)+len(site_ind)+len(other_sex_adni)+len(other_sex_oas)+len(other_sex_habs)]
    
half_fc_adni_correct_all = np.r_[half_fc_adni_correct, half_fc_adni_other_correct]
half_fc_adni_dx_all = np.r_[sa_dem_dx_adni, other_dx_adni]
half_fc_adni_age_all = np.r_[sa_dem_age_adni, other_age_adni]
half_fc_adni_sex_all = np.r_[sa_dem_sex_adni, other_sex_adni]
half_fc_adni_sub_sess_all = np.r_[sa_dem_sub_sess_adni, other_sub_sess_adni]

info_other_sa_define = sio.loadmat(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\superage_data_adni_logicmem_based.mat')
sa_dem_sub_adni_new_define = np.squeeze(info_other_sa_define['sub_id_sa_dem'])
sa_dem_sess_adni_new_define = np.squeeze(info_other_sa_define['sess_sa_dem'])
sa_dem_dx_adni_new_define = np.squeeze(info_other_sa_define['dx_sa_dem'])
df=pd.DataFrame(np.c_[sa_dem_sub_adni_new_define, sa_dem_sess_adni_new_define],columns=['sub','sess'])
df['sub_sess']=df['sub']+'_'+df['sess']
sa_dem_sub_sess_adni_new_define = df['sub_sess'].values

_, sa_adni_new_define_idx, sa_adni_idx = np.intersect1d(sa_dem_sub_sess_adni_new_define[sa_dem_dx_adni_new_define == 0], half_fc_adni_sub_sess_all[(half_fc_adni_dx_all == 0)|(half_fc_adni_dx_all==2)], return_indices=True)
sa_fc_adni_new_define = half_fc_adni_correct_all[(half_fc_adni_dx_all == 0)|(half_fc_adni_dx_all==2)][sa_adni_idx]
sa_sex_adni_new_define = half_fc_adni_sex_all[(half_fc_adni_dx_all == 0)|(half_fc_adni_dx_all==2)][sa_adni_idx]
sa_age_adni_new_define = half_fc_adni_age_all[(half_fc_adni_dx_all == 0)|(half_fc_adni_dx_all==2)][sa_adni_idx]
sa_sub_sess_adni_new_define = half_fc_adni_sub_sess_all[(half_fc_adni_dx_all == 0)|(half_fc_adni_dx_all==2)][sa_adni_idx]

dem_fc_adni_new_define = half_fc_adni_correct[sa_dem_dx_adni == 1]
dem_sex_adni_new_define = sa_dem_sex_adni[sa_dem_dx_adni == 1]
dem_age_adni_new_define = sa_dem_age_adni[sa_dem_dx_adni == 1]
dem_sub_sess_adni_new_define = sa_dem_sub_sess_adni[sa_dem_dx_adni == 1]

sa_dem_fc_adni_new_define = np.r_[dem_fc_adni_new_define, sa_fc_adni_new_define]
sa_dem_sex_adni_new_define = np.r_[dem_sex_adni_new_define, sa_sex_adni_new_define]
sa_dem_age_adni_new_define = np.r_[dem_age_adni_new_define, sa_age_adni_new_define]
sa_dem_sub_sess_adni_new_define = np.r_[dem_sub_sess_adni_new_define, sa_sub_sess_adni_new_define]
sa_dem_sub_adni_new_define = pd.Series(sa_dem_sub_sess_adni_new_define).str.split(' ', expand = True).iloc[:,0]
sa_dem_dx_adni_new_define = np.r_[np.ones(len(dem_fc_adni_new_define)), np.zeros(len(sa_fc_adni_new_define))]

subject_used, uni_id = np.unique(sub_ind, return_index=True)

dx_ind_uniq = dx_ind[uni_id]
site_ind_uniq = site_ind[uni_id]
site_ind_uniq[site_ind_uniq == 1] = 20
# half_fc_adni_correct = copy.deepcopy(half_fc_adni)

t_p_vector = np.zeros((4950, 2))
for i in range(4950):
    t_p_vector[i, 0],  t_p_vector[i, 1] = t_test(half_fc_ind[dx_ind == 1, i], half_fc_ind[dx_ind == 0, i], 0.05, False)
    
weight_sys, _ = vector_to_matrix(t_p_vector[:,0])
label_idx = np.arange(0,100,1)
cbarlabel = 'Weights'
plt.figure(figsize =(15,15))
ax = plt.gca()
im, cbar = heatmap(weight_sys.squeeze(), label_idx, ax=ax, cmap='RdBu_r', connect_type='roi',
                    cbarlabel=cbarlabel, half_or_full = 'half', with_diag = True)
folds = 10
kf = StratifiedKFold(n_splits=folds, random_state=9, shuffle=True)
perform = []
perform_sub = []
perform_oas = []
perform_habs = []

weights = []

t_p_vector_all = []
indep_pred_prob = []
indep_performance_all = []
indep_performance_logic = []
indep_pred_prob_logic = []

half_fc_cn_dem_pair = np.r_[half_fc_adni_other_correct[other_dx_adni==2], half_fc_oas_other_correct[other_dx_oas_encode==2], half_fc_ind[dx_ind==1]]
dx_cn_dem_pair = np.r_[other_dx_adni[other_dx_adni==2], other_dx_oas_encode[other_dx_oas_encode==2], dx_ind[dx_ind==1]]
dx_cn_dem_pair[dx_cn_dem_pair==2] = 0

half_fc_cn_sa_pair = np.r_[half_fc_adni_other_correct[other_dx_adni==2], half_fc_oas_other_correct[other_dx_oas_encode==2], half_fc_ind[dx_ind==0]]
dx_cn_sa_pair = np.r_[other_dx_adni[other_dx_adni==2], other_dx_oas_encode[other_dx_oas_encode==2], dx_ind[dx_ind==0]]
dx_cn_sa_pair[dx_cn_sa_pair==2] = 1

indep_cn_dem_prob = []
indep_cn_sa_prob = []

for train_index, test_index in kf.split(dx_ind_uniq, dx_ind_uniq):
    tr_all_id = []
    for sub in subject_used[train_index]:
        tr_all_id.extend(list(np.where(sub_ind==sub)[0]))
    tr_all_id = np.array(tr_all_id)
    te_all_id = []
    for sub in subject_used[test_index]:
        te_all_id.extend(list(np.where(sub_ind==sub)[0]))
    te_all_id = np.array(te_all_id)
    te_subs = sub_ind[te_all_id]
    
    te_sub_unqiu, te_id = np.unique(te_subs, return_index=True)
    te_sub = pd.Series(te_sub_unqiu).str.split('_', expand = True).iloc[:,0]
    _, te_sub_id = np.unique(te_sub, return_index=True)

    X_train = half_fc_ind[tr_all_id]
    X_test = half_fc_ind[te_all_id]
    y_train = dx_ind[tr_all_id]
    y_test = dx_ind[te_all_id]
    site_test = site_ind[te_all_id]
    site_train = site_ind[tr_all_id]
    site_test_te = site_test[te_id]
    
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train = X_train[idx]
    y_train = y_train[idx]
    p_thre = 0.01

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    t_p_vector = np.zeros((4950, 2))
    for i in range(4950):
        t_p_vector[i, 0],  t_p_vector[i, 1] = t_test(X_train[y_train==1, i], X_train[y_train==0, i], 0.05, False)
    t_p_vector_all.append(t_p_vector)
    X_train = X_train[:,t_p_vector[:,1]<=p_thre]
    
    # sample_weights = compute_sample_weight(class_weight={0: 1, 1: 200000}, y=y_train) # svm
    sample_weights = compute_sample_weight(class_weight={0: 1, 1: 100000}, y=y_train) # svm

    clf = LinearSVC(penalty = 'l1', C = 0.1, max_iter=5000, dual=False).fit(X_train, y_train, sample_weight=sample_weights) # 
    # clf = LinearSVC(penalty = 'l1', C = 0.7, max_iter=5000, dual=False).fit(X_train, y_train, sample_weight=sample_weights) # 

    pred_tr = np.expand_dims(clf.predict(X_train), -1)
    weight = np.zeros((len(t_p_vector)))
    weight[t_p_vector[:,1]<=p_thre] = clf.coef_.squeeze()
    # weight[t_p_vector[:,1]<=p_thre] = clf.feature_importances_.squeeze()
    weights.append(weight.squeeze())
    
    X_test = scaler.transform(X_test)
    half_fc_adni_correct_ = scaler.transform(half_fc_adni_correct)
    half_fc_adni_correct_logic_define_ = scaler.transform(sa_dem_fc_adni_new_define)
    X_test = X_test[:,t_p_vector[:,1]<=p_thre]
    predPROB_te = clf.decision_function(X_test)
    pred_test_uni = []
    predPROB_test_uni = []
    for te in te_sub_unqiu:
        predPROB = np.mean(predPROB_te[te_subs == te],0)
        predPROB_test_uni.append(predPROB)
        if predPROB>0:
            pred_test_uni.append(1)
        else:
            pred_test_uni.append(0)

    pred_test_uni = np.array(pred_test_uni)
    predPROB_test_uni = np.array(predPROB_test_uni)
    
    sen, spe = sens_spec(pred_test_uni, y_test[te_id])
    acc = accuracy_score(y_test[te_id],pred_test_uni)
    AUC = roc_auc_score(y_test[te_id], predPROB_test_uni)
    print('training acc: {}'.format(accuracy_score(y_train,pred_tr)))
    perform.append([acc, sen, spe, AUC])
    
    sen, spe = sens_spec(pred_test_uni[te_sub_id], y_test[te_id][te_sub_id])
    acc = accuracy_score(y_test[te_id][te_sub_id],pred_test_uni[te_sub_id])
    AUC = roc_auc_score(y_test[te_id][te_sub_id], predPROB_test_uni[te_sub_id])
    perform_sub.append([acc, sen, spe, AUC])
    
    sen, spe = sens_spec(pred_test_uni[site_test_te==0], y_test[te_id][site_test_te==0])
    acc = accuracy_score(y_test[te_id][site_test_te==0],pred_test_uni[site_test_te==0])
    AUC = roc_auc_score(y_test[te_id][site_test_te==0], predPROB_test_uni[site_test_te==0])
    print('training acc oas: {}'.format(accuracy_score(y_train[site_train==0],pred_tr[site_train==0])))
    perform_oas.append([acc, sen, spe, AUC])
    
    acc = accuracy_score(y_test[te_id][site_test_te==20],pred_test_uni[site_test_te==20])
    print('training acc hasb: {}'.format(accuracy_score(y_train[site_train==20],pred_tr[site_train==20])))
    perform_habs.append(acc)
    #################independent test
    pred_ind = np.expand_dims(clf.predict(half_fc_adni_correct_[:,t_p_vector[:,1]<=p_thre]), -1)
    predPROB_ind = clf.decision_function(half_fc_adni_correct_[:,t_p_vector[:,1]<=p_thre])
    indep_pred_prob.append(predPROB_ind)
    sen, spe = sens_spec(pred_ind, sa_dem_dx_adni)
    acc = accuracy_score(sa_dem_dx_adni,pred_ind)
    AUC = roc_auc_score(sa_dem_dx_adni, np.array(predPROB_ind))
    indep_performance_all.append([acc, sen, spe, AUC])

    #################independent test logic mem
    pred_ind = np.expand_dims(clf.predict(half_fc_adni_correct_logic_define_[:,t_p_vector[:,1]<=p_thre]), -1)
    predPROB_ind = clf.decision_function(half_fc_adni_correct_logic_define_[:,t_p_vector[:,1]<=p_thre])
    indep_pred_prob_logic.append(predPROB_ind)
    sen, spe = sens_spec(pred_ind, sa_dem_dx_adni_new_define)
    acc = accuracy_score(sa_dem_dx_adni_new_define,pred_ind)
    AUC = roc_auc_score(sa_dem_dx_adni_new_define, np.array(predPROB_ind))
    indep_performance_logic.append([acc, sen, spe, AUC])
    
    #################independent test cn and dem
    half_fc_cn_dem_pair_ = scaler.transform(half_fc_cn_dem_pair)
    half_fc_cn_sa_pair_ = scaler.transform(half_fc_cn_sa_pair)
    predPROB_ind = clf.decision_function(half_fc_cn_dem_pair_[:,t_p_vector[:,1]<=p_thre])
    indep_cn_dem_prob.append(predPROB_ind)
    
    predPROB_ind = clf.decision_function(half_fc_cn_sa_pair_[:,t_p_vector[:,1]<=p_thre])
    indep_cn_sa_prob.append(predPROB_ind)
    
perform_oas = np.array(perform_oas)
print(perform_oas.mean(0))
perform_habs = np.array(perform_habs)
print(perform_habs.mean(0))
perform = np.array(perform)
print(perform.mean(0))
perform_sub = np.array(perform_sub)
print(perform_sub.mean(0))

indep_pred_prob = np.array(indep_pred_prob)
indep_pred_prob_mean = np.mean(indep_pred_prob, 0).squeeze()
indep_pred_all_mean = np.zeros((len(indep_pred_prob_mean)))
indep_pred_all_mean[indep_pred_prob_mean>0] = 1
indep_pred_all_mean[indep_pred_prob_mean<0] = 0
acc = accuracy_score(sa_dem_dx_adni,indep_pred_all_mean)
AUC = roc_auc_score(sa_dem_dx_adni, np.array(indep_pred_prob_mean))
sen, spe = sens_spec(indep_pred_all_mean, sa_dem_dx_adni)
adni_perf = [acc, sen, spe, AUC]
print('ensemble acc:{} sen:{} spe:{} auc:{}'.format(acc, sen, spe, AUC))

label_diff = (sa_dem_dx_adni - indep_pred_all_mean)
sa_dem_sub_adni_unq, sa_dem_sub_adni_unqi = np.unique(sa_dem_sub_adni, return_index=True)
acc = accuracy_score(sa_dem_dx_adni[sa_dem_sub_adni_unqi],indep_pred_all_mean[sa_dem_sub_adni_unqi])
AUC = roc_auc_score(sa_dem_dx_adni[sa_dem_sub_adni_unqi], np.array(indep_pred_prob_mean)[sa_dem_sub_adni_unqi])
sen, spe = sens_spec(indep_pred_all_mean[sa_dem_sub_adni_unqi], sa_dem_dx_adni[sa_dem_sub_adni_unqi])
adni_perf_sublevel = [acc, sen, spe, AUC]
print('ensemble unique sub acc:{} sen:{} spe:{} auc:{}'.format(acc, sen, spe, AUC))
adni_perf_uni = [acc, sen, spe, AUC]

indep_pred_prob_logic = np.array(indep_pred_prob_logic)
indep_pred_prob_mean = np.mean(indep_pred_prob_logic, 0).squeeze()
indep_pred_all_mean = np.zeros((len(indep_pred_prob_mean)))
indep_pred_all_mean[indep_pred_prob_mean>0] = 1
indep_pred_all_mean[indep_pred_prob_mean<0] = 0
acc = accuracy_score(sa_dem_dx_adni_new_define,indep_pred_all_mean)
AUC = roc_auc_score(sa_dem_dx_adni_new_define, np.array(indep_pred_prob_mean))
sen, spe = sens_spec(indep_pred_all_mean, sa_dem_dx_adni_new_define)
adni_perf = [acc, sen, spe, AUC]
print('ensemble acc:{} sen:{} spe:{} auc:{}'.format(acc, sen, spe, AUC))

sa_previous = sub_ind_ = pd.Series(sa_dem_sub_adni_unq[sa_dem_dx_adni[sa_dem_sub_adni_unqi] == 0]).str.split(' ', expand = True).iloc[:,0]
# sa_new_define = sa_dem_sub_adni_unq_newdefine[sa_dem_dx_adni_new_define[sa_dem_sub_adni_unqi_new_define] == 0]
# sa_intersect = np.intersect1d(sa_previous, sa_new_define)


weights = np.array(weights)
weights_mean = np.mean(weights, 0)
newcmp = diverge_map(high=(0.89, 0.3, 0.3), low=(0.3, 0.3, 1))
# weight_sys, _ = vector_to_matrix(weights_mean)
# label_idx = np.arange(0,100,1)
# cbarlabel = 'Weights'
# plt.figure(figsize =(15,15))
# ax = plt.gca()
# im, cbar = heatmap(weight_sys.squeeze(), label_idx, ax=ax, cmap=newcmp, connect_type='roi',
#                     cbarlabel=cbarlabel, half_or_full = 'half', with_diag = True)
# plt.savefig(os.path.join(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\classification\Classification_adni_indep_weights.svg'), bbox_inches = 'tight')
# sio.savemat(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\classification\Classification_adni_indep_weights.mat', {'weight': weight_sys})

# plt.figure(figsize =(15,15))
# edge = get_net_net_connect(np.expand_dims(abs(weight_sys), 0), 1, method = 'mean').squeeze()
# label_idx = np.arange(0,28,1)
# ax = plt.gca()
# im, cbar = heatmap(edge, label_idx, ax=ax, cmap='Greens', connect_type='net', with_diag = True, half_or_full='half', dash_line=False) 

# plt.figure(figsize =(15,15))
# edge = get_net_net_connect(np.expand_dims(abs(weight_sys), 0), 1, method = 'sum').squeeze()
# label_idx = np.arange(0,28,1)
# ax = plt.gca()
# im, cbar = heatmap(edge, label_idx, ax=ax, cmap='Greens', connect_type='net', with_diag = True, half_or_full='half', dash_line=False) 

# # all_dx_adni_ = np.delete(sa_dem_dx_adni, idx_del_ind[:5])
# # indep_pred_all_mean_ = np.delete(indep_pred_all_mean, idx_del_ind[:5])
# # indep_pred_prob_mean_ = np.delete(indep_pred_prob_mean, idx_del_ind[:5])
# # acc = accuracy_score(all_dx_adni_,indep_pred_all_mean_)
# # AUC = roc_auc_score(all_dx_adni_, np.array(indep_pred_prob_mean_))
# # sen, spe = sens_spec(indep_pred_all_mean_, all_dx_adni_)
# # print('ensemble acc:{} sen:{} spe:{} auc:{}'.format(acc, sen, spe, AUC))

sub_ind_ = pd.Series(sub_ind).str.split('_', expand = True).iloc[:,0]

# sub_ind_uni, sub_idx = np.unique(sub_ind, return_index=True)      
sub_ind_uni, sub_idx = np.unique(sub_ind_, return_index=True)      
dx_ind_uni = dx_ind[sub_idx]
sex_ind_uni = sex_ind[sub_idx]
age_ind_uni = age_ind[sub_idx]
site_ind_uni = site_ind[sub_idx]
chi2_contingency([[np.sum(sex_ind_uni[(dx_ind_uni==0)]==1),np.sum(sex_ind_uni[(dx_ind_uni==1)]==1)],
                  [np.sum(sex_ind_uni[(dx_ind_uni==0)]==2),np.sum(sex_ind_uni[(dx_ind_uni==1)]==2)]])
chi2_contingency([[np.sum(sex_ind_uni[(sex_ind_uni==0)&(site_ind_uni==1)]==1),np.sum(sex_ind_uni[(sex_ind_uni==1)&(site_ind_uni==1)]==1)],
                  [np.sum(sex_ind_uni[(sex_ind_uni==0)&(site_ind_uni==1)]==2),np.sum(sex_ind_uni[(sex_ind_uni==1)&(site_ind_uni==1)]==2)]])
Q1 = np.percentile(age_ind_uni[dx_ind_uni==0], 25)
Q3 = np.percentile(age_ind_uni[dx_ind_uni==0], 75)
np.median(age_ind_uni[dx_ind_uni==0])
ranksums(age_ind_uni[dx_ind_uni==0], age_ind_uni[dx_ind_uni==1])

site_indx = []
for i,sub in enumerate(sub_ind_uni):
    if 'HAB' in sub:
        site_indx.append(i)
site_indx = np.array(site_indx) 

# ##########################

sub_adni_uni, sub_idx = np.unique(sa_dem_sub_adni, return_index=True)
sa_dem_dx_adni_uni = sa_dem_dx_adni[sub_idx]
sa_dem_sex_adni_uni = sa_dem_sex_adni[sub_idx]
sa_dem_age_adni_uni = sa_dem_age_adni[sub_idx]
print(kstest(sa_dem_age_adni_uni, cdf = "norm"))
print(shapiro(sa_dem_age_adni_uni))
chi2_contingency([[np.sum(sa_dem_sex_adni_uni[sa_dem_dx_adni_uni==0]==1),np.sum(sa_dem_sex_adni_uni[sa_dem_dx_adni_uni==1]==1)],
                  [np.sum(sa_dem_sex_adni_uni[sa_dem_dx_adni_uni==0]==2),np.sum(sa_dem_sex_adni_uni[sa_dem_dx_adni_uni==1]==2)]])
# chi2_contingency([[np.sum(sa_dem_sex_adni[sa_dem_dx_adni==0]==1),np.sum(sa_dem_sex_adni[sa_dem_dx_adni==1]==1)],
#                   [np.sum(sa_dem_sex_adni[sa_dem_dx_adni==0]==2),np.sum(sa_dem_sex_adni[sa_dem_dx_adni==1]==2)]])
Q1 = np.percentile(sa_dem_age_adni_uni[sa_dem_dx_adni_uni==0], 25)
Q3 = np.percentile(sa_dem_age_adni_uni[sa_dem_dx_adni_uni==0], 75)
age_median = np.median(sa_dem_age_adni_uni[sa_dem_dx_adni_uni==0])
ranksums(sa_dem_age_adni_uni[sa_dem_dx_adni_uni==0], sa_dem_age_adni_uni[sa_dem_dx_adni_uni==1])

# ###################sample-based performance plot
# font2 = {'family': 'Tahoma', 'weight': 'bold', 'size': 30}
# matplotlib.rc('font', **font2)
# fig = plt.figure(figsize=(10,10),)#设置画布的尺寸
# grid = plt.GridSpec(1, 2, hspace=0.0, wspace=0.5)
# # Define the axes
# ax1 = fig.add_subplot(grid[:, 0])
# ax2 = fig.add_subplot(grid[:, 1])
# ax1.bar([0, 1 ,2, 3], perform.mean(0), width = 0.75, yerr = perform.std(0), alpha=.6, color = 'r')
# ax1.get_children()[0].set_color((0.941, 0.502, 0.498)) 
# ax1.get_children()[1].set_color((0.388, 0.584, 0.933)) 
# ax1.get_children()[2].set_color((0.96, 0.65, 0.1)) 
# ax1.get_children()[3].set_color((0.75, 0.75, 0.75)) 
# ax1.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax1.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# ax1.set_ylabel('Performance', fontproperties=font2)
# sns.stripplot(perform, ax = ax1, size = 13, color='black', alpha=0.4)
# ax1.set_xticklabels(['Accuracy', 'Sensitivity', 'Specificity', 'AUC'])
# plt.setp(ax1.get_xticklabels(), rotation=30) 
# ax1.set_yticks(np.arange(0, 1.3, 0.3))

# ax2.bar([0, 1 ,2, 3], adni_perf, width = 0.75, alpha=.6)
# ax2.set_xticks([0, 1 ,2, 3])
# ax2.get_children()[0].set_color((0.941, 0.502, 0.498)) 
# ax2.get_children()[1].set_color((0.388, 0.584, 0.933)) 
# ax2.get_children()[2].set_color((0.96, 0.65, 0.1)) 
# ax2.get_children()[3].set_color((0.75, 0.75, 0.75)) 
# ax2.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax2.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# ax2.set_yticks(np.arange(0, 1.3, 0.3))
# ax2.set_ylabel('Performance', fontproperties=font2)
# ax2.set_xticklabels(['Accuracy', 'Sensitivity', 'Specificity', 'AUC'])
# plt.setp(ax2.get_xticklabels(), rotation=30) 
# plt.savefig(os.path.join(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\classification', '{}.svg'.format('performance_classification_sample_based')), bbox_inches = 'tight')

# ####################subject-based performance plot
# fig = plt.figure(figsize=(10,10),)#设置画布的尺寸
# grid = plt.GridSpec(1, 2, hspace=0.0, wspace=0.5)
# # Define the axes
# ax1 = fig.add_subplot(grid[:, 0])
# ax2 = fig.add_subplot(grid[:, 1])
# ax1.bar([0, 1 ,2, 3], perform_sub.mean(0), width = 0.75, yerr = perform_sub.std(0), alpha=.6, color = 'r')
# ax1.get_children()[0].set_color((0.941, 0.502, 0.498)) 
# ax1.get_children()[1].set_color((0.388, 0.584, 0.933)) 
# ax1.get_children()[2].set_color((0.96, 0.65, 0.1)) 
# ax1.get_children()[3].set_color((0.75, 0.75, 0.75)) 
# ax1.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax1.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# ax1.set_ylabel('Performance', fontproperties=font2)
# sns.stripplot(perform, ax = ax1, size = 13, color='black', alpha=0.4)
# ax1.set_xticklabels(['Accuracy', 'Sensitivity', 'Specificity', 'AUC'])
# plt.setp(ax1.get_xticklabels(), rotation=30) 
# ax1.set_yticks(np.arange(0, 1.3, 0.3))

# ax2.bar([0, 1 ,2, 3], adni_perf_sublevel, width = 0.75, alpha=.6)
# ax2.set_xticks([0, 1 ,2, 3])
# ax2.get_children()[0].set_color((0.941, 0.502, 0.498)) 
# ax2.get_children()[1].set_color((0.388, 0.584, 0.933)) 
# ax2.get_children()[2].set_color((0.96, 0.65, 0.1)) 
# ax2.get_children()[3].set_color((0.75, 0.75, 0.75)) 
# ax2.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax2.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# ax2.set_yticks(np.arange(0, 1.3, 0.3))
# ax2.set_ylabel('Performance', fontproperties=font2)
# ax2.set_xticklabels(['Accuracy', 'Sensitivity', 'Specificity', 'AUC'])
# plt.setp(ax2.get_xticklabels(), rotation=30) 
# plt.savefig(os.path.join(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\classification', '{}.svg'.format('performance_classification_subject_based')), bbox_inches = 'tight')

# #########logic mem adni
# fig = plt.figure(figsize=(10,10),)#设置画布的尺寸
# # Define the axes
# ax = plt.gca()
# ax.bar([0, 1 ,2, 3], adni_perf_sublevel, width = 0.75, alpha=.6)
# ax.set_xticks([0, 1 ,2, 3])
# ax.get_children()[0].set_color((0.941, 0.502, 0.498)) 
# ax.get_children()[1].set_color((0.388, 0.584, 0.933)) 
# ax.get_children()[2].set_color((0.96, 0.65, 0.1)) 
# ax.get_children()[3].set_color((0.75, 0.75, 0.75)) 
# ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# ax.set_yticks(np.arange(0, 1.3, 0.3))
# ax.set_ylabel('Performance', fontproperties=font2)
# ax.set_xticklabels(['Accuracy', 'Sensitivity', 'Specificity', 'AUC'])
# plt.setp(ax.get_xticklabels(), rotation=30) 
# # plt.savefig(os.path.join(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\classification', '{}.svg'.format('performance_classification_subject_based')), bbox_inches = 'tight')
# plt.savefig(os.path.join(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\classification', '{}.svg'.format('performance_classification_subject_based_logicmem')), bbox_inches = 'tight')


# ####################site-based performance plot
# fig = plt.figure(figsize=(10,10),)#设置画布的尺寸
# grid = plt.GridSpec(1, 2, hspace=0.0, wspace=0.8, width_ratios=[2,1])
# # Define the axes
# ax1 = fig.add_subplot(grid[:, 0])
# ax2 = fig.add_subplot(grid[:, 1])
# ax1.bar([0, 1 ,2, 3], perform_oas.mean(0), width = 0.75, yerr = perform_oas.std(0), alpha=.6, color = 'r')
# ax1.get_children()[0].set_color((0.941, 0.502, 0.498)) 
# ax1.get_children()[1].set_color((0.388, 0.584, 0.933)) 
# ax1.get_children()[2].set_color((0.96, 0.65, 0.1)) 
# ax1.get_children()[3].set_color((0.75, 0.75, 0.75)) 
# ax1.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax1.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# ax1.set_ylabel('Performance', fontproperties=font2)
# sns.stripplot(perform_oas, ax = ax1, size = 13, color='black', alpha=0.4)
# ax1.set_xticklabels(['Accuracy', 'Sensitivity', 'Specificity', 'AUC'])
# plt.setp(ax1.get_xticklabels(), rotation=30) 
# ax1.set_yticks(np.arange(0, 1.3, 0.3))

# ax2.bar([0], perform_habs.mean(0), yerr = perform_habs.std(0), width = 0.4, alpha=.6)
# sns.stripplot(perform_habs, ax = ax2, size = 13, color='black', alpha=0.4)
# ax2.set_xticks([0])
# ax2.get_children()[0].set_color((0.96, 0.65, 0.1)) 
# ax2.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax2.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# ax2.set_yticks(np.arange(0, 1.3, 0.3))
# ax2.set_ylabel('Performance', fontproperties=font2)
# ax2.set_xticklabels(['Accuracy'])
# plt.setp(ax2.get_xticklabels(), rotation=30) 
# # plt.savefig(os.path.join(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\classification', '{}.svg'.format('performance_classification_site_based')), bbox_inches = 'tight')

