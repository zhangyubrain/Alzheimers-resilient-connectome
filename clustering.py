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
from common_utils import read_singal_fmri_ts, get_fc, t_test, keep_triangle_half, sens_spec, setup_seed, \
    vector_to_matrix, heatmap, get_net_net_connect, diverge_map_unipolar_colore, cohen_d
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
from scipy.stats import pearsonr, chi2_contingency, shapiro, kstest, iqr, ranksums, kendalltau, f_oneway, kruskal, kendalltau, spearmanr
import matplotlib
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import namedtuple, OrderedDict
from matplotlib import colors as mcolors
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols

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
other_sub_habs = []

for i,sub in enumerate(other_habs_sub):
    other_fc_habs.extend(list(half_habs_fc_all[habs_sub_sess_all==sub]))
    other_dx_habs.extend(list(habs_dx_all[habs_sub_sess_all==sub]))
    other_age_habs.extend(list(habs_age_all[habs_sub_sess_all==sub]))
    other_sex_habs.extend(list(habs_sex_all[habs_sub_sess_all==sub]))
    other_sub_habs.extend(list(habs_sub_sess_all[habs_sub_sess_all==sub]))

other_fc_habs = np.array(other_fc_habs)    
other_dx_habs = np.array(other_dx_habs)    
other_age_habs = np.array(other_age_habs)    
other_sex_habs = np.array(other_sex_habs)    
other_sub_habs = np.array(other_sub_habs)     

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
# other_sess_adni[other_sess_adni == 'bl                              '] = 'm000'
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

combat = neuroCombat(dat=harmonized_fc.T,
    covars=covars,
    batch_col=batch_col,
    # continuous_cols=continuous_cols,
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

_, sa_adni_new_define_idx, sa_adni_idx = np.intersect1d(sa_dem_sub_sess_adni_new_define[sa_dem_dx_adni_new_define == 0], half_fc_adni_sub_sess_all[half_fc_adni_dx_all == 0], return_indices=True)
sa_fc_adni_new_define = half_fc_adni_correct_all[half_fc_adni_dx_all == 0][sa_adni_idx]
sa_sex_adni_new_define = half_fc_adni_sex_all[half_fc_adni_dx_all == 0][sa_adni_idx]
sa_age_adni_new_define = half_fc_adni_age_all[half_fc_adni_dx_all == 0][sa_adni_idx]
sa_sub_sess_adni_new_define = half_fc_adni_sub_sess_all[half_fc_adni_dx_all == 0][sa_adni_idx]

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

########################

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
other_adni_pred_label = []
other_oas_pred_label = []
other_habs_pred_label = []

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

    # clf = LinearSVC(penalty = 'l1', C = 0.1, max_iter=5000, dual=False).fit(X_train, y_train, sample_weight=sample_weights) # 
    clf = LinearSVC(penalty = 'l1', C = 0.7, max_iter=5000, dual=False).fit(X_train, y_train, sample_weight=sample_weights) # 

    pred_tr = np.expand_dims(clf.predict(X_train), -1)
    weight = np.zeros((len(t_p_vector)))
    weight[t_p_vector[:,1]<=p_thre] = clf.coef_.squeeze()
    # weight[t_p_vector[:,1]<=p_thre] = clf.feature_importances_.squeeze()
    weights.append(weight.squeeze())
    
    X_test = scaler.transform(X_test)
    half_fc_adni_correct_ = scaler.transform(half_fc_adni_correct)
    half_fc_adni_correct_logic_define_ = scaler.transform(sa_dem_fc_adni_new_define)
    
    half_fc_adni_other_correct_ = scaler.transform(half_fc_adni_other_correct)
    half_fc_adni_other_correct_ = half_fc_adni_other_correct_[:,t_p_vector[:,1]<=p_thre]
    pred_adni_other = clf.decision_function(half_fc_adni_other_correct_)
    other_adni_pred_label.append(pred_adni_other)
    half_fc_oas_other_correct_ = scaler.transform(half_fc_oas_other_correct)
    half_fc_oas_other_correct_ = half_fc_oas_other_correct_[:,t_p_vector[:,1]<=p_thre]
    pred_oas_other = clf.decision_function(half_fc_oas_other_correct_)
    other_oas_pred_label.append(pred_oas_other)
    half_fc_habs_other_correct_ = scaler.transform(half_fc_habs_other_correct)
    half_fc_habs_other_correct_ = half_fc_habs_other_correct_[:,t_p_vector[:,1]<=p_thre]
    pred_habs_other = clf.decision_function(half_fc_habs_other_correct_)
    other_habs_pred_label.append(pred_habs_other)

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
    
other_adni_pred_label = np.array(other_adni_pred_label)
other_oas_pred_label = np.array(other_oas_pred_label)
other_habs_pred_label = np.array(other_habs_pred_label)

#################averaging runs
other_oas_pred_label_uni = []
half_fc_oas_other_correct_uni = []
idx_oas_uni = []
for i, sub in enumerate(oasis_sa_sub_sess_uniq):
    other_oas_pred_label_uni.append(other_oas_pred_label[:,oasis_sa_sub_sess_uniq == sub].mean(-1))
    half_fc_oas_other_correct_uni.append(half_fc_oas_other_correct[oasis_sa_sub_sess_uniq == sub,:].mean(0))
    idx_oas_uni.append(i)
other_oas_pred_label_uni = np.array(other_oas_pred_label_uni)
half_fc_oas_other_correct_uni = np.array(half_fc_oas_other_correct_uni)
idx_oas_uni = np.array(idx_oas_uni)

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

_, sa_dem_sub_adni_unqi = np.unique(sa_dem_sub_adni, return_index=True)
acc = accuracy_score(sa_dem_dx_adni[sa_dem_sub_adni_unqi],indep_pred_all_mean[sa_dem_sub_adni_unqi])
AUC = roc_auc_score(sa_dem_dx_adni[sa_dem_sub_adni_unqi], np.array(indep_pred_prob_mean)[sa_dem_sub_adni_unqi])
sen, spe = sens_spec(indep_pred_all_mean[sa_dem_sub_adni_unqi], sa_dem_dx_adni[sa_dem_sub_adni_unqi])
adni_perf_sublevel = [acc, sen, spe, AUC]
print('ensemble unique sub acc:{} sen:{} spe:{} auc:{}'.format(acc, sen, spe, AUC))
adni_perf_uni = [acc, sen, spe, AUC]

indep_pred_prob = np.array(indep_cn_dem_prob)
indep_pred_prob_mean = np.mean(indep_pred_prob, 0).squeeze()
indep_pred_all_mean = np.zeros((len(indep_pred_prob_mean)))
indep_pred_all_mean[indep_pred_prob_mean>0] = 1
indep_pred_all_mean[indep_pred_prob_mean<0] = 0
acc = accuracy_score(dx_cn_dem_pair,indep_pred_all_mean)
AUC = roc_auc_score(dx_cn_dem_pair, np.array(indep_pred_prob_mean))
sen, spe = sens_spec(indep_pred_all_mean, dx_cn_dem_pair)
adni_perf = [acc, sen, spe, AUC]
print('ensemble acc cn vs dem:{} sen:{} spe:{} auc:{}'.format(acc, sen, spe, AUC))

indep_pred_prob = np.array(indep_cn_sa_prob)
indep_pred_prob_mean = np.mean(indep_pred_prob, 0).squeeze()
indep_pred_all_mean = np.zeros((len(indep_pred_prob_mean)))
indep_pred_all_mean[indep_pred_prob_mean>0] = 1
indep_pred_all_mean[indep_pred_prob_mean<0] = 0
acc = accuracy_score(dx_cn_sa_pair,indep_pred_all_mean)
AUC = roc_auc_score(dx_cn_sa_pair, np.array(indep_pred_prob_mean))
sen, spe = sens_spec(indep_pred_all_mean, dx_cn_sa_pair)
adni_perf = [acc, sen, spe, AUC]
print('ensemble acc cn vs sa:{} sen:{} spe:{} auc:{}'.format(acc, sen, spe, AUC))

# # #################association of PCD in ADNI
other_sub_adni_uni, other_idx_adni_uni = np.unique(other_sub_adni, return_index=True)

other_sub_adni_unique = []
other_sess_adni_unique = []
other_idx_adni_unique = []
for sub in other_sub_adni_uni:
    # sub = '002_S_20.50                      '
    mask = other_sub_adni == sub
    sess_sub = other_sess_adni[mask]
    idx = np.where(mask)[0]
    if sum(mask) == 1:
        other_sub_adni_unique.append(sub)
        other_sess_adni_unique.append(other_sess_adni[mask][0])
        other_idx_adni_unique.append(idx[0])
    else:      
        sess_num = []
        for i,sess in enumerate(sess_sub):
            if 'bl' in sess:
                other_sub_adni_unique.append(sub)
                other_sess_adni_unique.append(other_sess_adni[mask][i])
                other_idx_adni_unique.append(idx[i])
                skip = True
                break
            else:
                sess_num.append(float(sess.split('m')[-1]))
                skip = False
        if skip == False:
            sess_num = np.array(sess_num)
            other_sub_adni_unique.append(sub)
            other_sess_adni_unique.append(other_sess_adni[mask][np.argmin(sess_num)])
            other_idx_adni_unique.append(idx[np.argmin(sess_num)])
            skip = True

other_sub_adni_unique = np.array(other_sub_adni_unique)
other_sess_adni_unique = np.array(other_sess_adni_unique)
other_idx_adni_unique = np.array(other_idx_adni_unique)
    
# mask_cn_tem = other_dx_adni[other_idx_adni_unique]
# other_sub_adni_unique_cn = other_sub_adni_unique[mask_cn_tem==2]
# other_sess_adni_unique_cn = other_sess_adni_unique[mask_cn_tem==2]
# other_adni_pred_label_avg_unique_cn = other_adni_pred_label_avg[mask_cn_tem==2]
# other_adni_pred_label_avg_all = other_adni_pred_label.mean(0)
# other_sess_adni_unique_cn = other_sess_adni_unique[mask_cn_tem==2]

PCD_adni_interest = {}
adni_path = r'E:\PHD\learning\research\AD\data\ADNI'
adni_demo = pd.read_csv(os.path.join(adni_path, 'ADNIMERGE_01May2024.csv'))
adni_demo_sub = adni_demo['PTID']
adni_demo_sess = adni_demo['VISCODE']
adni_demo_interest = []
for sub, sess in zip(other_sub_adni_unique, other_sess_adni_unique):
    if sum((adni_demo_sub == sub[:10])&(adni_demo_sess == sess.split(' ')[0])) > 0:
        adni_demo_interest.append(adni_demo[['PTEDUCAT', 'PTRACCAT', 'APOE4', 'FDG', 'PIB', 'AV45', 'FBB', 
                                              'ABETA', 'TAU', 'PTAU', 'CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 
                                              'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 
                                              'TRABSCOR', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 
                                              'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 
                                              'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal']][
            (adni_demo_sub == sub[:10])&(adni_demo_sess == sess.split(' ')[0])].values[0,:])
    else:
        adni_demo_interest.append([np.nan for i in range(len(['PTEDUCAT', 'PTRACCAT', 'APOE4', 'FDG', 'PIB', 'AV45', 'FBB', 
                                              'ABETA', 'TAU', 'PTAU', 'CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 
                                              'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 
                                              'TRABSCOR', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 
                                              'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 
                                              'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal']))])
adni_demo_interest = np.array(adni_demo_interest)
adni_demo_interest[adni_demo_interest[:,1]!='White',1] = 0
adni_demo_interest[adni_demo_interest[:,1]=='White',1] = 1
adni_demo_interest[adni_demo_interest[:,7]=='>1700',7] = 2000
adni_demo_interest[adni_demo_interest[:,9]=='<8',9] = 8
adni_demo_interest[adni_demo_interest[:,17]<-300,17] = np.nan
PCD_adni_interest.update({"AGE": other_age_adni[other_idx_adni_unique], 'GENDER': other_sex_adni[other_idx_adni_unique], 'dx': other_dx_adni[other_idx_adni_unique], 
                          'PTEDUCAT': adni_demo_interest[:,0], 
                          'PTRACCAT': adni_demo_interest[:,1], 'APOE4': adni_demo_interest[:,2], 'FDG': adni_demo_interest[:,3],
                          'AV45': adni_demo_interest[:,5], 'FBB': adni_demo_interest[:,6], 'ABETA': adni_demo_interest[:,7],
                          'TAU': adni_demo_interest[:,8], 'PTAU': adni_demo_interest[:,9], 'CDRSB': adni_demo_interest[:,10], 
                          'ADAS11': adni_demo_interest[:,11], 'ADAS13': adni_demo_interest[:,12], 'MMSE': adni_demo_interest[:,13], 
                          'RAVLT_immediate': adni_demo_interest[:,14], 'RAVLT_learning': adni_demo_interest[:,15], 'RAVLT_forgetting': adni_demo_interest[:,16], 
                          'RAVLT_perc_forgetting': adni_demo_interest[:,17], 'LDELTOTAL': adni_demo_interest[:,18], 
                          'TRABSCOR': adni_demo_interest[:,20], 'FAQ': adni_demo_interest[:,21], 'MOCA': adni_demo_interest[:,22], 
                          'EcogPtMem': adni_demo_interest[:,23], 'EcogPtLang': adni_demo_interest[:,24], 'EcogPtVisspat': adni_demo_interest[:,25], 
                          'EcogPtPlan': adni_demo_interest[:,26], 'EcogPtOrgan': adni_demo_interest[:,27], 'EcogPtDivatt': adni_demo_interest[:,28], 
                          'EcogPtTotal': adni_demo_interest[:,29], 'EcogSPMem': adni_demo_interest[:,30], 'EcogSPLang': adni_demo_interest[:,31], 
                          'EcogSPVisspat': adni_demo_interest[:,32], 'EcogSPPlan': adni_demo_interest[:,33], 'EcogSPOrgan': adni_demo_interest[:,34], 
                          'EcogSPDivatt': adni_demo_interest[:,35], 'EcogSPTotal': adni_demo_interest[:,36]}) 

adni_cdr = pd.read_csv(os.path.join(adni_path, 'CDR_07May2024.csv'))
adni_cdr_sub = adni_cdr['PTID']
adni_cdr_sess = adni_cdr['VISCODE2']
adni_cdr_sess[adni_cdr_sess == 'sc'] = 'bl'
adni_cdr_sess[adni_cdr_sess == 'f'] = 'bl'
adni_cdr_interest = []
for sub, sess in zip(other_sub_adni_unique, other_sess_adni_unique):
    if sum((adni_cdr_sub == sub[:10])&(adni_cdr_sess == sess.split(' ')[0])) > 0:
        adni_cdr_interest.append(adni_cdr[['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE', 'CDGLOBAL']][
            (adni_cdr_sub == sub[:10])&(adni_cdr_sess == sess.split(' ')[0])].values[0,:])
    else:
        adni_cdr_interest.append([np.nan for i in range(len(['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE', 'CDGLOBAL']))])
adni_cdr_interest = np.array(adni_cdr_interest)
PCD_adni_interest.update({'CDMEMORY': adni_cdr_interest[:,0], 'CDORIENT': adni_cdr_interest[:,1], 'CDJUDGE': adni_cdr_interest[:,2],
                          'CDCOMMUN': adni_cdr_interest[:,3], 'CDHOME': adni_cdr_interest[:,4], 'CDCARE': adni_cdr_interest[:,5], 
                          'CDGLOBAL': adni_cdr_interest[:,6]}) 

adni_faq = pd.read_csv(os.path.join(adni_path, 'FAQ_08May2024.csv'))
adni_faq_sub = adni_faq['PTID']
adni_faq_sess = adni_faq['VISCODE2']
adni_faq_interest = []
for sub, sess in zip(other_sub_adni_unique, other_sess_adni_unique):
    if sum((adni_faq_sub == sub[:10])&(adni_faq_sess == sess.split(' ')[0])) > 0:
        adni_faq_interest.append(adni_faq[['FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG', 'FAQMEAL',
                                            'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL']][
            (adni_faq_sub == sub[:10])&(adni_faq_sess == sess.split(' ')[0])].values[0,:])
    else:
        adni_faq_interest.append([np.nan for i in range(len(['FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG', 'FAQMEAL',
                                            'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL']))])
adni_faq_interest = np.array(adni_faq_interest)
adni_faq_interest[(adni_faq_interest==1)|(adni_faq_interest==2)|(adni_faq_interest==3)] = 1
adni_faq_interest[(adni_faq_interest==4)] = 2
adni_faq_interest[(adni_faq_interest==5)] = 3
PCD_adni_interest.update({'FAQFINAN': adni_faq_interest[:,0], 'FAQFORM': adni_faq_interest[:,1], 'FAQSHOP': adni_faq_interest[:,2],
                          'FAQGAME': adni_faq_interest[:,3], 'FAQBEVG': adni_faq_interest[:,4], 'FAQMEAL': adni_faq_interest[:,5], 
                          'FAQEVENT': adni_faq_interest[:,6], 'FAQTV': adni_faq_interest[:,7], 'FAQREM': adni_faq_interest[:,8], 
                          'FAQTRAVL': adni_faq_interest[:,9]}) 


adni_neuroBIO = pd.read_csv(os.path.join(adni_path, 'UPENNBIOMK_ROCHE_ELECSYS_16May2024.csv'))
adni_neuroBIO_sub = adni_neuroBIO['RID']
adni_neuroBIO_sess = adni_neuroBIO['VISCODE2']
adni_neuroBIO_interest = []
for sub, sess in zip(other_sub_adni_unique, other_sess_adni_unique):
    if sum((adni_neuroBIO_sub == int(sub[6:10]))&(adni_neuroBIO_sess == sess.split(' ')[0])) > 0:
        adni_neuroBIO_interest.append(adni_neuroBIO[['ABETA42', 'TAU', 'PTAU']][
            (adni_neuroBIO_sub == int(sub[6:10]))&(adni_neuroBIO_sess == sess.split(' ')[0])].values[0,:])
    else:
        adni_neuroBIO_interest.append([np.nan for i in range(len(['ABETA42', 'TAU', 'PTAU']))])
adni_neuroBIO_interest = np.array(adni_neuroBIO_interest)
PCD_adni_interest.update({'ABETA': adni_neuroBIO_interest[:,0], 'TAU': adni_neuroBIO_interest[:,1], 'PTAU': adni_neuroBIO_interest[:,2]}) 

adni_gds = pd.read_csv(os.path.join(adni_path, 'GDSCALE_08May2024.csv'))
adni_gds_sub = adni_gds['PTID']
adni_gds_sess = adni_gds['VISCODE2']
adni_gds_interest = []
for sub, sess in zip(other_sub_adni_unique, other_sess_adni_unique):
    if sum((adni_gds_sub == sub[:10])&(adni_gds_sess == sess.split(' ')[0])) > 0:
        adni_gds_interest.append(adni_gds['GDTOTAL'][
            (adni_gds_sub == sub[:10])&(adni_gds_sess == sess.split(' ')[0])].values[0])
    else:
        adni_gds_interest.append(np.nan)
adni_gds_interest = np.array(adni_gds_interest)
PCD_adni_interest.update({'GDTOTAL': adni_gds_interest}) 

adni_his = pd.read_csv(os.path.join(adni_path, 'MODHACH_08May2024.csv'))
adni_his_sub = adni_his['PTID']
adni_his_sess = adni_his['VISCODE2']
adni_his_interest = []
for sub, sess in zip(other_sub_adni_unique, other_sess_adni_unique):
    if sum((adni_his_sub == sub[:10])) > 0:
        adni_his_interest.append(adni_his[['HMSTEPWS', 'HMSOMATC', 'HMEMOTIO', 'HMHYPERT', 'HMSTROKE', 'HMNEURSM',
                                            'HMNEURSG', 'HMSCORE', 'HMONSET']][
            (adni_his_sub == sub[:10])].values[0,:])
    else:
        adni_his_interest.append([np.nan for i in range(len(['HMSTEPWS', 'HMSOMATC', 'HMEMOTIO', 'HMHYPERT', 'HMSTROKE', 'HMNEURSM',
                                            'HMNEURSG', 'HMSCORE', 'HMONSET']))])
adni_his_interest = np.array(adni_his_interest)
PCD_adni_interest.update({'HMSTEPWS': adni_his_interest[:,0], 'HMSOMATC': adni_his_interest[:,1], 'HMEMOTIO': adni_his_interest[:,2],
                          'HMHYPERT': adni_his_interest[:,3], 'HMSTROKE': adni_his_interest[:,4], 'HMNEURSM': adni_his_interest[:,5],
                          'HMNEURSG': adni_his_interest[:,6], 'HMSCORE': adni_his_interest[:,7], 'HMONSET': adni_his_interest[:,8]}) 

adni_moca = pd.read_csv(os.path.join(adni_path, 'MOCA_08May2024.csv'))
adni_moca_sub = adni_moca['PTID']
adni_moca_sess = adni_moca['VISCODE2']
adni_moca_interest = []
for sub, sess in zip(other_sub_adni_unique, other_sess_adni_unique):
    if sum((adni_moca_sub == sub[:10])&(adni_moca_sess == sess.split(' ')[0])) > 0:
        adni_moca_interest.append(adni_moca[['TRAILS', 'CUBE', 'CLOCKCON', 'CLOCKNO', 'CLOCKHAN', 'LION', 'RHINO', 'CAMEL',
                                              'IMMT1W1', 'IMMT1W2', 'IMMT1W3', 'IMMT1W4', 'IMMT1W5', 'IMMT2W1', 'IMMT2W2', 'IMMT2W3', 
                                              'IMMT2W4', 'IMMT2W5', 'DIGFOR', 'DIGBACK', 'LETTERS', 'SERIAL1', 'SERIAL2', 'SERIAL3', 'SERIAL4', 
                                              'SERIAL5', 'REPEAT1', 'REPEAT2', 'FFLUENCY', 'ABSTRAN', 'ABSMEAS', 'DELW1', 'DELW2', 'DELW3', 
                                              'DELW4', 'DELW5', 'DATE', 'MONTH', 'YEAR', 'DAY', 'PLACE', 'CITY']][
            (adni_moca_sub == sub[:10])&(adni_moca_sess == sess.split(' ')[0])].values[0,:])
    else:
        adni_moca_interest.append([np.nan for i in range(len(['TRAILS', 'CUBE', 'CLOCKCON', 'CLOCKNO', 'CLOCKHAN', 'LION', 'RHINO', 'CAMEL',
                                              'IMMT1W1', 'IMMT1W2', 'IMMT1W3', 'IMMT1W4', 'IMMT1W5', 'IMMT2W1', 'IMMT2W2', 'IMMT2W3', 
                                              'IMMT2W4', 'IMMT2W5', 'DIGFOR', 'DIGBACK', 'LETTERS', 'SERIAL1', 'SERIAL2', 'SERIAL3', 'SERIAL4', 
                                              'SERIAL5', 'REPEAT1', 'REPEAT2', 'FFLUENCY', 'ABSTRAN', 'ABSMEAS', 'DELW1', 'DELW2', 'DELW3', 
                                              'DELW4', 'DELW5', 'DATE', 'MONTH', 'YEAR', 'DAY', 'PLACE', 'CITY']))])
adni_moca_interest = np.array(adni_moca_interest)
adni_moca_subtract = adni_moca_interest[:,21:26].sum(-1)
adni_moca_subtract[(adni_moca_subtract>=2) & (adni_moca_subtract<=3)] = 2
adni_moca_subtract[adni_moca_subtract>=4] = 3
adni_moca_frq = adni_moca_interest[:,28]
adni_moca_frq[adni_moca_frq<11] = 0
adni_moca_frq[adni_moca_frq>=11] = 1

PCD_adni_interest.update({'visuospatial_executive': adni_moca_interest[:,:5].sum(-1), 'naming': adni_moca_interest[:,5:8].sum(-1),
                          'abstraction': adni_moca_interest[:,29:31].sum(-1), 'delayed_recall': adni_moca_interest[:,31:36].sum(-1)/3, 
                          'orientation': adni_moca_interest[:,36:].sum(-1), 'attention': adni_moca_interest[:,18:20].sum(-1) + adni_moca_interest[:,20] + adni_moca_subtract, 
                          'language': adni_moca_interest[:,26:28].sum(-1) + adni_moca_frq}) 

adni_nps = pd.read_csv(os.path.join(adni_path, 'NPIQ_08May2024.csv'))
adni_nps_sub = adni_nps['PTID']
adni_nps_sess = adni_nps['VISCODE2']
adni_nps_interest = []
for sub, sess in zip(other_sub_adni_unique, other_sess_adni_unique):
    if sum((adni_nps_sub == sub[:10])&(adni_nps_sess == sess.split(' ')[0])) > 0:
        adni_nps_interest.append(adni_nps[['NPIASEV', 'NPIBSEV', 'NPICSEV', 'NPIDSEV', 'NPIESEV', 'NPIFSEV', 'NPIGSEV', 
                                            'NPIHSEV', 'NPIISEV', 'NPIJSEV', 'NPIKSEV', 'NPILSEV','NPISCORE']][
            (adni_nps_sub == sub[:10])&(adni_nps_sess == sess.split(' ')[0])].values[0,:])
    else:
        adni_nps_interest.append([np.nan for i in range(len(['NPIASEV', 'NPIBSEV', 'NPICSEV', 'NPIDSEV', 'NPIESEV', 'NPIFSEV', 'NPIGSEV', 
                                            'NPIHSEV', 'NPIISEV', 'NPIJSEV', 'NPIKSEV', 'NPILSEV','NPISCORE']))])
adni_nps_interest = np.array(adni_nps_interest)
adni_nps_interest[adni_nps_interest == -4] = 0
adni_nps_interest[np.isnan(adni_nps_interest)] = 0
PCD_adni_interest.update({'NPIASEV': adni_nps_interest[:,0], 'NPIBSEV': adni_nps_interest[:,1], 'NPICSEV': adni_nps_interest[:,2],
                          'NPIDSEV': adni_nps_interest[:,3], 'NPIESEV': adni_nps_interest[:,4], 'NPIFSEV': adni_nps_interest[:,5], 
                          'NPIGSEV': adni_nps_interest[:,6], 'NPIHSEV': adni_nps_interest[:,7], 'NPIISEV': adni_nps_interest[:,8], 
                          'NPIJSEV': adni_nps_interest[:,9], 'NPIKSEV': adni_nps_interest[:,10], 'NPILSEV': adni_nps_interest[:,11], 
                          'NPISCORE': adni_nps_interest[:,12]}) 

adni_neu = pd.read_csv(os.path.join(adni_path, 'NEUROBAT_08May2024.csv'))
adni_neu_sub = adni_neu['PTID']
adni_neu_sess = adni_neu['VISCODE2']
adni_neu_interest = []
for sub, sess in zip(other_sub_adni_unique, other_sess_adni_unique):
    if sum((adni_neu_sub == sub[:10])&(adni_neu_sess == sess.split(' ')[0])) > 0:
        adni_neu_interest.append(adni_neu[['LIMMTOTAL', 'DSPANFOR', 'DSPANFLTH', 'DSPANBAC', 'DSPANBLTH', 'CATANIMSC', 'CATANPERS', 
                                            'CATVEGESC', 'CATVGPERS', 'TRAASCOR', 'TRABSCOR', 'DIGITSCOR','LDELTOTAL', 'BNTTOTAL']][
            (adni_neu_sub == sub[:10])&(adni_neu_sess == sess.split(' ')[0])].values[0,:])
    else:
        adni_neu_interest.append([np.nan for i in range(len(['LIMMTOTAL', 'DSPANFOR', 'DSPANFLTH', 'DSPANBAC', 'DSPANBLTH', 'CATANIMSC',
                                                              'CATANPERS', 'CATVEGESC', 'CATVGPERS', 'TRAASCOR', 'TRABSCOR', 'DIGITSCOR','LDELTOTAL',
                                                              'BNTTOTAL']))])
adni_neu_interest = np.array(adni_neu_interest)
adni_neu_interest[adni_neu_interest == -4] = 0
PCD_adni_interest.update({'LIMMTOTAL': adni_neu_interest[:,0], 'CATANIMSC': adni_neu_interest[:,5], 
                          'CATANPERS': adni_neu_interest[:,6], 'CATVEGESC': adni_neu_interest[:,7], 'DSPANFOR': adni_neu_interest[:,1], 
                          'TRAASCOR': adni_neu_interest[:,9], 'DSPANBAC': adni_neu_interest[:,3], 'DIGITSCOR': adni_neu_interest[:,11],
                          'BNTTOTAL': adni_neu_interest[:,13]})

##############correlation analysis
PCD_keys = ['AGE', 'GENDER', 'PTEDUCAT', 'PTRACCAT', 'APOE4', 'FDG', 'AV45', 'FBB', 'ABETA', 'TAU', 'PTAU', 'CDRSB',
            'ADAS11', 'ADAS13', 'MMSE', 'RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 
            'RAVLT_perc_forgetting', 'LDELTOTAL', 'TRABSCOR', 'FAQ', 'MOCA', 'CDMEMORY',
            'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE', 'CDGLOBAL', 'FAQFINAN', 'FAQFORM', 
            'FAQSHOP', 'FAQGAME', 'FAQBEVG', 'FAQMEAL', 'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL', 'GDTOTAL',
            'visuospatial_executive', 'naming', 'abstraction', 'delayed_recall', 'orientation', 'attention',
            'language', 'NPIASEV', 'NPIBSEV', 'NPICSEV',
            'NPIDSEV', 'NPIESEV', 'NPIFSEV', 'NPIGSEV', 'NPIHSEV', 'NPIISEV', 'NPIJSEV', 'NPIKSEV', 'NPILSEV',
            'NPISCORE', 'LIMMTOTAL', 'DSPANFOR', 'DSPANBAC', 'CATANIMSC', 'CATANPERS', 'CATVEGESC', 'TRAASCOR', 'DIGITSCOR', 'BNTTOTAL']

df_adni = pd.DataFrame(PCD_adni_interest).astype(float)
df_adni['A+'] = (df_adni['ABETA'] > 977) * 1
df_adni['A+'][df_adni['ABETA'].isnull()] = np.nan
df_adni['T+'] = (df_adni['PTAU'] < 22) * 1
df_adni['T+'][df_adni['PTAU'].isnull()] = np.nan
PCD_keys.append('A+')
PCD_keys.append('T+')
stat_all_adni_cn = []
stat_keys_adni = []
stat_test = []
mask_cn = (other_dx_adni[other_idx_adni_unique]==2)
mask_MCI = (other_dx_adni[other_idx_adni_unique]==3)
# other_adni_pred_label_avg = other_adni_pred_label.mean(0)[other_idx_adni_unique]

other_adni_pred_label_mask = other_adni_pred_label>0
other_adni_pred_label_avg = other_adni_pred_label_mask.mean(0)[other_idx_adni_unique]

# #################association of PCD in OASIS
oasis_sa_sub = pd.Series(oasis_sa_sub_sess_uniq).str.split('_', expand = True).iloc[:,0]
oasis_sa_sub_unique, unique_id = np.unique(oasis_sa_sub, return_index=True)
oasis_sa_sub_sess_unique = oasis_sa_sub_sess_uniq[unique_id]

PCD_all_interest = {}
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

oasis_path = r'E:\PHD\learning\research\AD\data\OASIS3'
oasis_demo = pd.read_csv(os.path.join(oasis_path, 'UDS-A2-InformatDemos.csv'))
oasis_demo2 = pd.read_csv(os.path.join(oasis_path, 'OASIS3_UDSa2_cs_demo.csv'))
oasis_demo_sub = oasis_demo['Subject']
oasis_demo_sub2 = oasis_demo2['OASISID']
oasis_demo_interest = []
for sub in oasis_sa_sub_sess_unique:
    sub = sub_sess[:8]
    if sum(oasis_demo_sub == sub) >= 1:
        oasis_demo_interest.append(oasis_demo[['INRACE', 'INEDUC', 'INSEX']][oasis_demo_sub == sub].values[0,:])
    else:
        if sum(oasis_demo_sub2 == sub) >= 1:
            oasis_demo_interest.append(oasis_demo2[['INRACE', 'INEDUC', 'INSEX']][oasis_demo_sub2 == sub].values[0,:])
        else:
            oasis_demo_interest.append([np.nan, np.nan, np.nan])
oasis_demo_interest = np.array(oasis_demo_interest)
PCD_all_interest.update({"INRACE": oasis_demo_interest[:,0], 'INEDUC': oasis_demo_interest[:,1], 'age': other_age_oas[unique_id], 
                          'sex': oasis_demo_interest[:,2], 'dx': other_dx_oas_encode[unique_id]}) 

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
for sub_sess in oasis_sa_sub_sess_unique:
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

PCD_all_interest.update({"apoe": PCD_interest[:,0], 'NPIQINF': PCD_interest[:,1], 'DELSEV': PCD_interest[:,2], 
                          'HALLSEV': PCD_interest[:,3], 'AGITSEV': PCD_interest[:,4], 'DEPDSEV': PCD_interest[:,5], 
                          'ANXSEV': PCD_interest[:,6], 'ELATSEV': PCD_interest[:,7], 'APASEV': PCD_interest[:,8], 
                          'DISNSEV': PCD_interest[:,9], 'IRRSEV': PCD_interest[:,10], 'MOTSEV': PCD_interest[:,11], 
                          'NITESEV': PCD_interest[:,12], 'APPSEV': PCD_interest[:,13], 'mmse': PCD_interest[:,14],
                          'cdr': PCD_interest[:,15], 'commun': PCD_interest[:,16], 'homehobb': PCD_interest[:,17], 
                          'judgment': PCD_interest[:,18], 'memory': PCD_interest[:,19], 'orient': PCD_interest[:,20], 'perscare': PCD_interest[:,21],
                          'sumbox': PCD_interest[:,22], 'DIGIF': PCD_interest[:,23], 'DIGIB': PCD_interest[:,24], 
                          'ANIMALS': PCD_interest[:,25], 'VEG': PCD_interest[:,26], 'TRAILA': PCD_interest[:,27], 
                          'TRAILALI': PCD_interest[:,28], 'TRAILB': PCD_interest[:,29], 'TRAILBLI': PCD_interest[:,30], 'WAIS': PCD_interest[:,31], 
                          'LOGIMEM': PCD_interest[:,32], 'MEMUNITS': PCD_interest[:,33], 'MEMTIME': PCD_interest[:,34], 'BOSTON': PCD_interest[:,35]}) 

oasis_faq = pd.read_csv(os.path.join(oasis_path, 'UDS-B7-FAQ.csv'))
oasis_faq_sub_day = oasis_faq['UDS_B7FAQDATA ID']
oasis_faq_sub = oasis_faq_sub_day.str.split('_', expand = True).iloc[:,0]
oasis_faq_day = oasis_faq_sub_day.str.split('_d', expand = True).iloc[:,-1].astype(int)
oasis_faq_sess = []
day_min = []
for da in oasis_faq_day:
    mask = day_all==da
    if sum(mask) > 0:
        oasis_faq_sess.append(sess_all[day_all==da][0])
    else:
        diff = day_all-da
        diff_min_id = np.argmin(abs(diff))
        diff_min = diff[diff_min_id]
        day_min.append(diff_min)
        oasis_faq_sess.append(sess_all[diff_min_id])
oasis_faq_sess = np.array(oasis_faq_sess)

oasis_faq_interest = []
for sub_sess in oasis_sa_sub_sess_unique:
    sub = sub_sess[:8]
    sess = sub_sess[9:]
    if sum((oasis_faq_sub == sub) & (oasis_faq_sess == sess))> 0:
        oasis_faq_interest.append(oasis_faq[['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS',
                                              'PAYATTN', 'REMDATES', 'TRAVEL']][
            (oasis_faq_sub == sub) & (oasis_faq_sess == sess)].values[0,:])
    else:
        oasis_faq_interest.append([np.nan for i in range(len(['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS',
                                              'PAYATTN', 'REMDATES', 'TRAVEL']))])
oasis_faq_interest = np.array(oasis_faq_interest)
oasis_faq_interest[(oasis_faq_interest==8)|(oasis_faq_interest==9)] = np.nan
PCD_all_interest.update({"BILLS": oasis_faq_interest[:,0], 'TAXES': oasis_faq_interest[:,1], 'SHOPPING': oasis_faq_interest[:,2], 
                          'GAMES': oasis_faq_interest[:,3], 'STOVE': oasis_faq_interest[:,4], 'MEALPREP': oasis_faq_interest[:,5], 
                          'EVENTS': oasis_faq_interest[:,6], 'PAYATTN': oasis_faq_interest[:,7], 'REMDATES': oasis_faq_interest[:,8], 
                          'TRAVEL': oasis_faq_interest[:,9]}) 

oasis_pet = pd.read_csv(os.path.join(oasis_path, 'OASIS3_PUP.csv'))
oasis_pet['Centil_fBP_TOT_CORTMEAN_av45'] = np.zeros((oasis_pet.shape[0]))
oasis_pet['Centil_fBP_TOT_CORTMEAN_av45'] = oasis_pet['Centil_fBP_TOT_CORTMEAN'][oasis_pet['tracer'] == 'AV45']
oasis_pet['Centil_fSUVR_TOT_CORTMEAN_av45'] = np.zeros((oasis_pet.shape[0]))
oasis_pet['Centil_fSUVR_TOT_CORTMEAN_av45'] = oasis_pet['Centil_fSUVR_TOT_CORTMEAN'][oasis_pet['tracer'] == 'AV45']
oasis_pet['Centil_fBP_rsf_TOT_CORTMEAN_av45'] = np.zeros((oasis_pet.shape[0]))
oasis_pet['Centil_fBP_rsf_TOT_CORTMEAN_av45'] = oasis_pet['Centil_fBP_rsf_TOT_CORTMEAN'][oasis_pet['tracer'] == 'AV45']
oasis_pet['Centil_fSUVR_rsf_TOT_CORTMEAN_av45'] = np.zeros((oasis_pet.shape[0]))
oasis_pet['Centil_fSUVR_rsf_TOT_CORTMEAN_av45'] = oasis_pet['Centil_fSUVR_rsf_TOT_CORTMEAN'][oasis_pet['tracer'] == 'AV45']
oasis_pet['Centil_fBP_TOT_CORTMEAN_pib'] = np.zeros((oasis_pet.shape[0]))
oasis_pet['Centil_fBP_TOT_CORTMEAN_pib'] = oasis_pet['Centil_fBP_TOT_CORTMEAN'][oasis_pet['tracer'] == 'PIB']
oasis_pet['Centil_fSUVR_TOT_CORTMEAN_pib'] = np.zeros((oasis_pet.shape[0]))
oasis_pet['Centil_fSUVR_TOT_CORTMEAN_pib'] = oasis_pet['Centil_fSUVR_TOT_CORTMEAN'][oasis_pet['tracer'] == 'PIB']
oasis_pet['Centil_fBP_rsf_TOT_CORTMEAN_pib'] = np.zeros((oasis_pet.shape[0]))
oasis_pet['Centil_fBP_rsf_TOT_CORTMEAN_pib'] = oasis_pet['Centil_fBP_rsf_TOT_CORTMEAN'][oasis_pet['tracer'] == 'PIB']
oasis_pet['Centil_fSUVR_rsf_TOT_CORTMEAN_pib'] = np.zeros((oasis_pet.shape[0]))
oasis_pet['Centil_fSUVR_rsf_TOT_CORTMEAN_pib'] = oasis_pet['Centil_fSUVR_rsf_TOT_CORTMEAN'][oasis_pet['tracer'] == 'PIB']
oasis_pet_sub_day = oasis_pet['PUP_PUPTIMECOURSEDATA ID']
oasis_pet_sub = oasis_pet_sub_day.str.split('_', expand = True).iloc[:,0]
oasis_pet_day = oasis_pet_sub_day.str.split('_d', expand = True).iloc[:,-1].astype(int)
oasis_pet_sess = []
day_min = []
for da in oasis_pet_day:
    mask = day_all==da
    if sum(mask) > 0:
        oasis_pet_sess.append(sess_all[day_all==da][0])
    else:
        diff = day_all-da
        diff_min_id = np.argmin(abs(diff))
        diff_min = diff[diff_min_id]
        day_min.append(diff_min)
        oasis_pet_sess.append(sess_all[diff_min_id])
oasis_pet_sess = np.array(oasis_pet_sess)

oasis_pet_interest = []
for sub_sess in oasis_sa_sub_sess_unique:
    sub = sub_sess[:8]
    sess = sub_sess[9:]
    if sum((oasis_pet_sub == sub) & (oasis_pet_sess == sess))> 0:
        oasis_pet_interest.append(oasis_pet[['Centil_fBP_TOT_CORTMEAN_av45', 'Centil_fSUVR_TOT_CORTMEAN_av45', 
                                                    'Centil_fBP_rsf_TOT_CORTMEAN_av45', 'Centil_fSUVR_rsf_TOT_CORTMEAN_av45',
                                                    'Centil_fBP_TOT_CORTMEAN_pib', 'Centil_fSUVR_TOT_CORTMEAN_pib', 
                                                    'Centil_fBP_rsf_TOT_CORTMEAN_pib', 'Centil_fSUVR_rsf_TOT_CORTMEAN_pib']][
            (oasis_pet_sub == sub) & (oasis_pet_sess == sess)].values[0,:])
    else:
        oasis_pet_interest.append([np.nan for i in range(len(['Centil_fBP_TOT_CORTMEAN_av45', 'Centil_fSUVR_TOT_CORTMEAN_av45', 
                                                    'Centil_fBP_rsf_TOT_CORTMEAN_av45', 'Centil_fSUVR_rsf_TOT_CORTMEAN_av45',
                                                    'Centil_fBP_TOT_CORTMEAN_pib', 'Centil_fSUVR_TOT_CORTMEAN_pib', 
                                                    'Centil_fBP_rsf_TOT_CORTMEAN_pib', 'Centil_fSUVR_rsf_TOT_CORTMEAN_pib']))])
oasis_pet_interest = np.array(oasis_pet_interest)
PCD_all_interest.update({"Centil_fBP_TOT_CORTMEAN_av45": oasis_pet_interest[:,0], 'Centil_fSUVR_TOT_CORTMEAN_av45': oasis_pet_interest[:,1], 
                          'Centil_fBP_rsf_TOT_CORTMEAN_av45': oasis_pet_interest[:,2], 'Centil_fSUVR_rsf_TOT_CORTMEAN_av45': oasis_pet_interest[:,3],
                          "Centil_fBP_TOT_CORTMEAN_pib": oasis_pet_interest[:,4], 'Centil_fSUVR_TOT_CORTMEAN_pib': oasis_pet_interest[:,5], 
                          'Centil_fBP_rsf_TOT_CORTMEAN_pib': oasis_pet_interest[:,6], 'Centil_fSUVR_rsf_TOT_CORTMEAN_pib': oasis_pet_interest[:,7]}) 
PCD_all_interest['TRAILB'] = PCD_all_interest['TRAILB'].astype(float)
PCD_all_interest['TRAILB'][PCD_all_interest['TRAILB']>400] = np.nan
df_oas = pd.DataFrame(PCD_all_interest).astype(float)

PCD_keys = ['INRACE', 'INEDUC', 'sex', 'age', 'apoe', 'DELSEV', 'HALLSEV', 'AGITSEV',
                                        'DEPDSEV', 'ANXSEV', 'ELATSEV', 'APASEV', 'DISNSEV', 'IRRSEV', 'MOTSEV', 'NITESEV',
                                        'APPSEV', 'mmse', 'cdr', 'commun','homehobb', 'judgment', 'memory', 'orient', 'perscare', 'sumbox', 'VEG',
                                        'DIGIF', 'DIGIB', 'ANIMALS', 'TRAILA', 'TRAILALI', 'TRAILB', 'TRAILBLI', 'WAIS', 'LOGIMEM', 
                                        'MEMUNITS', 'MEMTIME', 'BOSTON', 'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP',
                                        'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL', 'Centil_fBP_TOT_CORTMEAN_av45', 'Centil_fSUVR_TOT_CORTMEAN_av45', 
                                        'Centil_fBP_rsf_TOT_CORTMEAN_av45', 'Centil_fSUVR_rsf_TOT_CORTMEAN_av45', 'Centil_fBP_TOT_CORTMEAN_pib', 'Centil_fSUVR_TOT_CORTMEAN_pib', 
                                        'Centil_fBP_rsf_TOT_CORTMEAN_pib', 'Centil_fSUVR_rsf_TOT_CORTMEAN_pib']

stat_all_oas_cn = []
stat_keys_oas = []
stat_test = []
mask_MCI = (other_dx_oas_encode[unique_id]==3)
mask_cn = (other_dx_oas_encode[unique_id]==2)

other_oas_pred_label_mask = other_oas_pred_label>0
other_oas_pred_label_avg = other_oas_pred_label_mask.mean(0)[unique_id]

# other_oas_pred_label_avg = other_oas_pred_label.mean(0)[unique_id]
oasis_sa_sub_sess_uniq_ = pd.Series(oasis_sa_sub_sess_unique).str.split('_', expand = True).iloc[:,0]

# #################association of PCD in HABS sa
half_fc_habs_aging_correct_uniq = []
habs_sub_uni, uni_idx = np.unique(other_sub_habs, return_index=True)
age_habs_aging_correct_uniq = []
sex_habs_aging_correct_uniq = []

for i in range(len(habs_sub_uni)):
    half_fc_habs_aging_correct_uniq.append(half_fc_habs_other_correct[other_sub_habs == habs_sub_uni[i], :].mean(0))
    age_habs_aging_correct_uniq.append(other_age_habs[other_sub_habs == habs_sub_uni[i]].mean(0))
    sex_habs_aging_correct_uniq.append(other_sex_habs[other_sub_habs == habs_sub_uni[i]][0])
half_fc_habs_aging_correct_uniq = np.array(half_fc_habs_aging_correct_uniq)
age_habs_aging_correct_uniq = np.array(age_habs_aging_correct_uniq)
sex_habs_aging_correct_uniq = np.array(sex_habs_aging_correct_uniq)

habs_aging_behavior_path = r'I:\data\habs'
cog_path = os.path.join(habs_aging_behavior_path, 'Cognition_HABS_DataRelease_2.0.csv')
cog_info = pd.read_csv(cog_path)
demo_path = os.path.join(habs_aging_behavior_path, 'Demographics_HABS_DataRelease_2.0.csv')
demo_info = pd.read_csv(demo_path)
demo_info = demo_info.rename(columns={'SubjID': 'SubjIDshort'})
clinic_path = os.path.join(habs_aging_behavior_path, 'ClinicalMeasures_HABS_DataRelease_2.0.csv')
clinic_info = pd.read_csv(clinic_path)
FDG_path = os.path.join(habs_aging_behavior_path, 'FDG_FS6_SUVR_HABS_DataRelease_2.0.csv')
FDG_info = pd.read_csv(FDG_path)
FDG_info['fdg_mean'] = FDG_info.iloc[:,4:].mean(1)
FTP_path = os.path.join(habs_aging_behavior_path, 'FTP_FS6_SUVR_HABS_DataRelease_2.0.csv')
FTP_info = pd.read_csv(FTP_path)
FTP_info['ftp_mean'] = FTP_info.iloc[:,4:].mean(1)
PIB_path = os.path.join(habs_aging_behavior_path, 'PIB_FS6_DVR_HABS_DataRelease_2.0.csv')
PIB_info = pd.read_csv(PIB_path)
PIB_info['StudyArc'][PIB_info['StudyArc'] == 'HAB_18m_PET'] = 'HAB_3.0' 
FDG_info['StudyArc'][FDG_info['StudyArc'] == 'HAB_18m_PET'] = 'HAB_3.0' 
FDG_info['StudyArc'][FDG_info['StudyArc'] == 'HAB_AI'] = 'HAB_6.0' 
pcd_all_habs = pd.merge(cog_info, demo_info, on=['SubjIDshort', 'StudyArc', 'NP_SessionDate'], how='outer')
pcd_all_habs = pd.merge(pcd_all_habs, clinic_info, on=['SubjIDshort', 'StudyArc', 'NP_SessionDate'], how='outer')
pcd_all_habs = pd.merge(pcd_all_habs, FDG_info, on=['SubjIDshort', 'StudyArc'], how='outer')
pcd_all_habs = pd.merge(pcd_all_habs, FTP_info, on=['SubjIDshort', 'StudyArc'], how='outer')
pcd_all_habs = pd.merge(pcd_all_habs, PIB_info, on=['SubjIDshort', 'StudyArc'], how='outer')
pcd_all_habs['sub_sess'] = pcd_all_habs['SubjIDshort'].str.split('_', expand = True).iloc[:,-1] + '_' +pcd_all_habs['StudyArc']
pcd_all_habs['Race'][pcd_all_habs['Race'] == 'W'] = 0
pcd_all_habs['Race'][pcd_all_habs['Race'] != 'W'] = 1

_, hpc_fc_idx, pcd_idx = np.intersect1d(habs_sub_uni, pcd_all_habs['sub_sess'], return_indices=True)
PCD_habs_interest = {}
PCD_habs_interest['age']= other_age_habs[uni_idx][hpc_fc_idx]
PCD_habs_interest['sex']= other_sex_habs[uni_idx][hpc_fc_idx]
PCD_habs_interest['YrsOfEd']= pcd_all_habs['YrsOfEd'][pcd_idx]
PCD_habs_interest['APOE_haplotype']= pcd_all_habs['APOE_haplotype'][pcd_idx].astype(str).str.count('4')
PCD_habs_interest['BNT_30']= pcd_all_habs['BNT_30'][pcd_idx]
PCD_habs_interest['CAT3']= pcd_all_habs['CAT3'][pcd_idx]
PCD_habs_interest['CAT_Animal_Total']= pcd_all_habs['CAT_Animal_Total'][pcd_idx]
PCD_habs_interest['CAT_Vegetable_Total']= pcd_all_habs['CAT_Vegetable_Total'][pcd_idx]
PCD_habs_interest['CAT_Fruit_Total']= pcd_all_habs['CAT_Fruit_Total'][pcd_idx]
PCD_habs_interest['Digits_Forward']= pcd_all_habs['Digits_Forward'][pcd_idx]
PCD_habs_interest['Digits_Backwards']= pcd_all_habs['Digits_Backwards'][pcd_idx]
PCD_habs_interest['DigitSym']= pcd_all_habs['DigitSym'][pcd_idx]
PCD_habs_interest['FAS_Total']= pcd_all_habs['FAS_Total'][pcd_idx]
PCD_habs_interest['FAS_F_Total']= pcd_all_habs['FAS_F_Total'][pcd_idx]
PCD_habs_interest['FAS_A_Total']= pcd_all_habs['FAS_A_Total'][pcd_idx]
PCD_habs_interest['FAS_S_Total']= pcd_all_habs['FAS_S_Total'][pcd_idx]
PCD_habs_interest['FCsrt_FNC']= pcd_all_habs['FCsrt_FNC'][pcd_idx]
PCD_habs_interest['FCsrt_Free']= pcd_all_habs['FCsrt_Free'][pcd_idx]
PCD_habs_interest['LetterNum_Total']= pcd_all_habs['LetterNum_Total'][pcd_idx]
PCD_habs_interest['LogicMem_IL']= pcd_all_habs['LogicMem_IL'][pcd_idx]
PCD_habs_interest['LogicMem_DR']= pcd_all_habs['LogicMem_DR'][pcd_idx]
PCD_habs_interest['SRT_dr']= pcd_all_habs['SRT_dr'][pcd_idx]
PCD_habs_interest['SRT_cltr']= pcd_all_habs['SRT_cltr'][pcd_idx]
PCD_habs_interest['SRT_cr']= pcd_all_habs['SRT_cr'][pcd_idx]
PCD_habs_interest['SRT_ltr']= pcd_all_habs['SRT_ltr'][pcd_idx]
PCD_habs_interest['SRT_lts']= pcd_all_habs['SRT_lts'][pcd_idx]
PCD_habs_interest['SRT_mc']= pcd_all_habs['SRT_mc'][pcd_idx]
PCD_habs_interest['SRT_str']= pcd_all_habs['SRT_str'][pcd_idx]
PCD_habs_interest['SRT_tr']= pcd_all_habs['SRT_tr'][pcd_idx]
PCD_habs_interest['TMT_A']= pcd_all_habs['TMT_A'][pcd_idx]
PCD_habs_interest['TMT_B']= pcd_all_habs['TMT_B'][pcd_idx]
PCD_habs_interest['VFDT']= pcd_all_habs['VFDT'][pcd_idx]
PCD_habs_interest['CDR_Global']= pcd_all_habs['CDR_Global'][pcd_idx]
PCD_habs_interest['CDR_SB']= pcd_all_habs['CDR_SB'][pcd_idx]
PCD_habs_interest['CDR_Memory']= pcd_all_habs['CDR_Memory'][pcd_idx]
PCD_habs_interest['MMSE_Total']= pcd_all_habs['MMSE_Total'][pcd_idx]
PCD_habs_interest['MMSE_Orientation']= pcd_all_habs['MMSE_Orientation'][pcd_idx]
PCD_habs_interest['MMSE_ImmRecall']= pcd_all_habs['MMSE_ImmRecall'][pcd_idx]
PCD_habs_interest['MMSE_AttnCalc']= pcd_all_habs['MMSE_AttnCalc'][pcd_idx]
PCD_habs_interest['MMSE_DelRecall']= pcd_all_habs['MMSE_DelRecall'][pcd_idx]
PCD_habs_interest['MMSE_Language']= pcd_all_habs['MMSE_Language'][pcd_idx]
PCD_habs_interest['MMSE_Pentagons']= pcd_all_habs['MMSE_Pentagons'][pcd_idx]
PCD_habs_interest['GDS_Total']= pcd_all_habs['GDS_Total'][pcd_idx]
PCD_habs_interest['Hachinski']= pcd_all_habs['Hachinski'][pcd_idx]
PCD_habs_interest['Race']= pcd_all_habs['Race'][pcd_idx]

PCD_keys = ['Race', 'YrsOfEd', 'sex', 'age', 'APOE_haplotype', 'BNT_30', 'CAT3', 'CAT_Animal_Total', 'CAT_Vegetable_Total', 'CAT_Fruit_Total', 
            'Digits_Forward', 'Digits_Backwards', 'DigitSym', 'FAS_Total', 'FAS_F_Total', 'FAS_A_Total', 'FAS_S_Total', 'FCsrt_FNC', 'FCsrt_Free', 'LetterNum_Total',
            'LogicMem_IL', 'LogicMem_DR', 'SRT_dr', 'SRT_cltr', 'SRT_cr', 'SRT_ltr', 'SRT_lts', 'SRT_mc', 'SRT_str', 'SRT_tr', 'TMT_A', 'TMT_B', 'VFDT', 'CDR_Global', 
            'CDR_SB', 'CDR_Memory', 'MMSE_Total', 'MMSE_Orientation', 'MMSE_ImmRecall', 'MMSE_AttnCalc', 'MMSE_DelRecall', 'MMSE_Language', 'MMSE_Pentagons', 
            'GDS_Total', 'Hachinski']

df_pcd_habs = pd.DataFrame(PCD_habs_interest).astype(float)
##############correlation analysis
stat_all_habs_cn = []
stat_keys_habs = []
stat_test = []
# other_habs_pred_label_avg = other_habs_pred_label.mean(0)[uni_idx][hpc_fc_idx]
other_habs_pred_label_mask = other_habs_pred_label>0
other_habs_pred_label_avg = other_habs_pred_label_mask.mean(0)[hpc_fc_idx]

df_all = {'age': np.r_[df_oas['age'], df_pcd_habs['age']], 
            # 'sub': np.r_[np.delete(HPC_sa_sub_uni,idx), oasis_sa_sub_sess_uniq_, habs_sub_uni[hpc_fc_idx]], 
          'sub': np.r_[oasis_sa_sub_sess_uniq_, habs_sub_uni[hpc_fc_idx]], 
          'sex': np.r_[df_oas['sex'], df_pcd_habs['sex']],
          'cluster_label': np.r_[other_oas_pred_label_avg, other_habs_pred_label_avg],
          'race': np.r_[df_oas['INRACE'], df_pcd_habs['Race']], 
          'educ': np.r_[df_oas['INEDUC'], df_pcd_habs['YrsOfEd']], 
          'dx': np.r_[df_oas['dx'], [2 for i in range(len(df_pcd_habs['sex']))]],  
          # 'moca_total': np.r_[df_pcd_hcp['moca_total'], [np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          # 'visuospatial_executive': np.r_[df_pcd_hcp['visuospatial_executive'], [np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          # 'naming': np.r_[df_pcd_hcp['naming'], [np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          # 'abstraction': np.r_[df_pcd_hcp['abstraction'], [np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          # 'delayed_recall': np.r_[df_pcd_hcp['delayed_recall'], [np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          # 'orientation': np.r_[df_pcd_hcp['orientation'], [np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          # 'attention': np.r_[df_pcd_hcp['attention'], [np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          # 'language': np.r_[df_pcd_hcp['language'], [np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'tmt_a': np.r_[df_oas['TRAILA'], df_pcd_habs['TMT_A']], 
          'tmt_b': np.r_[df_oas['TRAILB'], df_pcd_habs['TMT_B']], 
          'apoe': np.r_[df_oas['apoe'].astype(str).str.count('4'), df_pcd_habs['APOE_haplotype']], 
           'DELSEV': np.r_[df_oas['DELSEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'HALLSEV': np.r_[df_oas['HALLSEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'AGITSEV': np.r_[df_oas['AGITSEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'DEPDSEV': np.r_[df_oas['DEPDSEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'ANXSEV': np.r_[df_oas['ANXSEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'ELATSEV': np.r_[df_oas['ELATSEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'APASEV': np.r_[df_oas['APASEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'DISNSEV': np.r_[df_oas['DISNSEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'IRRSEV': np.r_[df_oas['IRRSEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'MOTSEV': np.r_[df_oas['MOTSEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'NITESEV': np.r_[df_oas['NITESEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
           'APPSEV': np.r_[df_oas['APPSEV'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          'mmse': np.r_[df_oas['mmse'], df_pcd_habs['MMSE_Total']], 
          'cdr': np.r_[df_oas['cdr'], df_pcd_habs['CDR_Global']], 
          'commun': np.r_[df_oas['commun'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          'homehobb': np.r_[df_oas['homehobb'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          'perscare': np.r_[df_oas['perscare'], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          'judgment': np.r_[df_oas['judgment'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'memory': np.r_[df_oas['memory'], df_pcd_habs['CDR_Memory']],
          'orient': np.r_[df_oas['orient'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'sumbox': np.r_[df_oas['sumbox'], df_pcd_habs['CDR_SB']],
          'RAVLT_immediate': np.r_[[np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          'RAVLT_learning': np.r_[[np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          'RAVLT_forgetting': np.r_[[np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          'RAVLT_perc_forgetting': np.r_[[np.nan for i in range(len(df_oas['sex']))], [np.nan for i in range(len(df_pcd_habs['sex']))]], 
          'LogMem_Delay': np.r_[df_oas['MEMUNITS'], df_pcd_habs['LogicMem_DR']],
          'LogMem_immediate': np.r_[df_oas['LOGIMEM'], df_pcd_habs['LogicMem_IL']],
          'CFT_ANIMALS': np.r_[df_oas['ANIMALS'], df_pcd_habs['CAT_Animal_Total']],
          'CFT_VEG': np.r_[df_oas['VEG'], df_pcd_habs['CAT_Vegetable_Total']],
          'DIGIF': np.r_[df_oas['DIGIF'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'DIGIB': np.r_[df_oas['DIGIB'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'BOSTON': np.r_[df_oas['BOSTON'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'BILLS': np.r_[df_oas['BILLS'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'TAXES': np.r_[df_oas['TAXES'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'SHOPPING': np.r_[df_oas['SHOPPING'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'GAMES': np.r_[df_oas['GAMES'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'STOVE': np.r_[df_oas['STOVE'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'MEALPREP': np.r_[df_oas['MEALPREP'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'EVENTS': np.r_[df_oas['EVENTS'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'PAYATTN': np.r_[df_oas['PAYATTN'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'REMDATES': np.r_[df_oas['REMDATES'], [np.nan for i in range(len(df_pcd_habs['sex']))]],
          'TRAVEL': np.r_[df_oas['TRAVEL'], [np.nan for i in range(len(df_pcd_habs['sex']))]]}

#################################################
#################################################
####################brain pattern visualization
scaler = StandardScaler()
half_fc_ind_zscore = scaler.fit_transform(half_fc_ind)
half_fc_adni_other_correct_zsocre = scaler.transform(half_fc_adni_other_correct)
half_fc_oas_other_correct_uni_zscore = scaler.transform(half_fc_oas_other_correct_uni)
half_fc_habs_aging_correct_uniq_zsocre = scaler.transform(half_fc_habs_aging_correct_uniq)
half_fc_adni_correct_score = scaler.transform(half_fc_adni_correct)

dem_fc_discovery = half_fc_ind[dx_ind == 1]
dem_sub_discovery = sub_ind[dx_ind == 1]
dem_age_discovery = age_ind[dx_ind == 1]
dem_sex_discovery = sex_ind[dx_ind == 1]

sa_fc_discovery = half_fc_ind[dx_ind == 0]
sa_sub_discovery = sub_ind[dx_ind == 0]
sa_age_discovery = age_ind[dx_ind == 0]
sa_sex_discovery = sex_ind[dx_ind == 0]
###discovery pattern
mask_cn_oas = (other_dx_oas_encode[unique_id]==2)
mask_ad_oas = (other_dx_oas_encode[unique_id]==1)
cn_fc_discovery = half_fc_ind[dx_ind == 2]
dem_fc_discovery2 = half_fc_oas_other_correct_uni[unique_id][mask_ad_oas]
dem_sub_discovery2 = oasis_sa_sub_sess_uniq[unique_id][mask_ad_oas]
dem_age_discovery2 = other_age_oas[unique_id][mask_ad_oas]
dem_sex_discovery2 = other_sex_oas[unique_id][mask_ad_oas]

dem_fc_discovery = np.r_[dem_fc_discovery, dem_fc_discovery2]
dem_age_discovery = np.r_[dem_age_discovery, dem_age_discovery2]
dem_sex_discovery = np.r_[dem_sex_discovery, dem_sex_discovery2]
dem_sub_discovery = np.r_[dem_sub_discovery, dem_sub_discovery2]

# dem_fc_discovery = dem_fc_discovery2
# dem_age_discovery = dem_age_discovery2
# dem_sex_discovery = dem_sex_discovery2
# dem_sub_discovery = dem_sub_discovery2
###discovery pattern
age_oas_other_correct_uni_salike_cn = (other_age_oas[unique_id][(other_oas_pred_label_avg<0.5)&mask_cn_oas])
sex_oas_other_correct_uni_salike_cn = (other_sex_oas[unique_id][(other_oas_pred_label_avg<0.5)&mask_cn_oas])
age_oas_other_correct_uni_demlike_cn = (other_age_oas[unique_id][(other_oas_pred_label_avg>0.5)&mask_cn_oas])
sex_oas_other_correct_uni_demlike_cn = (other_sex_oas[unique_id][(other_oas_pred_label_avg>0.5)&mask_cn_oas])
mask_cn_oas = (other_dx_oas_encode[unique_id]==2)
half_fc_oas_other_correct_uni_salike_cn = (half_fc_oas_other_correct_uni[unique_id][(other_oas_pred_label_avg<0.5)&mask_cn_oas])
half_fc_oas_other_correct_uni_demlike_cn = (half_fc_oas_other_correct_uni[unique_id][(other_oas_pred_label_avg>0.5)&mask_cn_oas])

# habs_idx_bl = []
# for i in range(len(habs_sub_uni)):
#     if '1.0' in habs_sub_uni[i]:
#         habs_idx_bl.append(i)
# habs_idx_bl = np.array(habs_idx_bl)        
     
half_fc_habs_other_correct_uni_salike_cn = (half_fc_habs_aging_correct_uniq[(other_habs_pred_label_avg<0.5)])
half_fc_habs_other_correct_uni_demlike_cn = (half_fc_habs_aging_correct_uniq[(other_habs_pred_label_avg>0.5)])
age_habs_correct_uni_salike_cn = (age_habs_aging_correct_uniq[(other_habs_pred_label_avg<0.5)])
sex_habs_correct_uni_salike_cn = (sex_habs_aging_correct_uniq[(other_habs_pred_label_avg<0.5)])
age_habs_correct_uni_demlike_cn = (age_habs_aging_correct_uniq[(other_habs_pred_label_avg>0.5)])
sex_habs_correct_uni_demlike_cn = (sex_habs_aging_correct_uniq[(other_habs_pred_label_avg>0.5)])

half_fc_discovery_uni_all_cn = np.r_[half_fc_oas_other_correct_uni, half_fc_habs_aging_correct_uniq]
half_fc_discovery_uni_salike_cn = np.r_[half_fc_oas_other_correct_uni_salike_cn, half_fc_habs_other_correct_uni_salike_cn]
half_fc_discovery_uni_demlike_cn = np.r_[half_fc_oas_other_correct_uni_demlike_cn, half_fc_habs_other_correct_uni_demlike_cn]
age_discovery_uni_salike_cn = np.r_[age_oas_other_correct_uni_salike_cn, age_habs_correct_uni_salike_cn]
sex_discovery_uni_salike_cn = np.r_[sex_oas_other_correct_uni_salike_cn, sex_habs_correct_uni_salike_cn]
age_discovery_uni_demlike_cn = np.r_[age_oas_other_correct_uni_demlike_cn, age_habs_correct_uni_demlike_cn]
sex_discovery_uni_demlike_cn = np.r_[sex_oas_other_correct_uni_demlike_cn, sex_habs_correct_uni_demlike_cn]

############################################
################longitidual data preprare, using adni2.mat
sub_adni_lmm = []
sess_adni_lmm = []
pcd_adni_lmm = []
age_adni_lmm = []
sex_adni_lmm = []
cluster_label_adni_lmm = []
dx_adni_lmm = []
sess_used = adni_demo_sess.unique()
sess_used = np.delete(sess_used, 33)
i = 0
for sub in other_sub_adni_unique:
    print(i)

    for sess in sess_used:
        pcd_adni_lmm_one = []
        sub_adni_lmm.append(str(sub))
        cluster_label_adni_lmm.append(other_adni_pred_label_avg[i])
        dx_adni_lmm.append(other_dx_adni[other_idx_adni_unique][i])
        if 'bl' in sess:
            sess_adni_lmm.append(0)
        else:
            sess_adni_lmm.append(int(sess.split('m')[-1].split(' ')[0]))
        adni_demo_sess[adni_demo_sess == 'm0'] = 'bl'
        if sum((adni_demo_sub == sub[:10])&(adni_demo_sess == sess.split(' ')[0])) == 1:
            pcd_adni_lmm_one.extend(list(adni_demo[['PTEDUCAT', 'PTRACCAT', 'APOE4', 'FDG', 'PIB', 'AV45', 'FBB', 
                                                  'CDRSB', 'RAVLT_immediate', 
                                                  'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 
                                                  'TRABSCOR', 'FAQ', 'MOCA']][
                (adni_demo_sub == sub[:10])&(adni_demo_sess == sess.split(' ')[0])].values[0,:]))
            age = adni_demo['AGE'][(adni_demo_sub == sub[:10])&(adni_demo_sess == sess.split(' ')[0])].values[0]           
            sex = adni_demo['PTGENDER'][(adni_demo_sub == sub[:10])&(adni_demo_sess == sess.split(' ')[0])].values[0]                      
            if sess != 'bl':
                age =  age + float(sess.split('m')[-1])/12
        elif sum((adni_demo_sub == sub[:10])&(adni_demo_sess == sess.split(' ')[0])) > 1:
            pcd = adni_demo[['PTEDUCAT', 'PTRACCAT', 'APOE4', 'FDG', 'PIB', 'AV45', 'FBB', 
                                                  'CDRSB', 'RAVLT_immediate', 
                                                  'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 
                                                  'TRABSCOR', 'FAQ', 'MOCA']][
                (adni_demo_sub == sub[:10])&(adni_demo_sess == sess.split(' ')[0])].values
            selected_id = np.isnan(pcd[:,3:].astype(float)).sum(-1)
            pcd_adni_lmm_one.extend(list(pcd[np.argmin(selected_id)]))
            age = adni_demo['AGE'][(adni_demo_sub == sub[:10])&(adni_demo_sess == sess.split(' ')[0])].values[0]                      
            sex = adni_demo['PTGENDER'][(adni_demo_sub == sub[:10])&(adni_demo_sess == sess.split(' ')[0])].values[0]                      
            if sess != 'bl':
                age =  age + float(sess.split('m')[-1])/12            
        else:
            pcd_adni_lmm_one.extend(list([np.nan for i in range(len(['PTEDUCAT', 'PTRACCAT', 'APOE4', 'FDG', 'PIB', 'AV45', 'FBB', 
                                                  'CDRSB', 'RAVLT_immediate', 
                                                  'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 
                                                  'TRABSCOR', 'FAQ', 'MOCA']))]))
            age = np.nan                   
            sex = np.nan            
        adni_cdr_sess[adni_cdr_sess == 'sc'] = 'bl'
        age_adni_lmm.append(age)
        sex_adni_lmm.append(sex)        
        if sum((adni_cdr_sub == sub[:10])&(adni_cdr_sess == sess.split(' ')[0])) == 1:
            pcd_adni_lmm_one.extend(list(adni_cdr[['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE', 'CDGLOBAL']][
                (adni_cdr_sub == sub[:10])&(adni_cdr_sess == sess.split(' ')[0])].values[0,:]))
        elif sum((adni_cdr_sub == sub[:10])&(adni_cdr_sess == sess.split(' ')[0])) > 1:
            pcd = adni_cdr[['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE', 'CDGLOBAL']][
                (adni_cdr_sub == sub[:10])&(adni_cdr_sess == sess.split(' ')[0])].values
            selected_id = np.isnan(pcd.astype(float)).sum(-1)
            pcd_adni_lmm_one.extend(list(pcd[np.argmin(selected_id)]))
        else:
            pcd_adni_lmm_one.extend(list([np.nan for i in range(len(['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE', 'CDGLOBAL']))]))

        if sum((adni_faq_sub == sub[:10])&(adni_faq_sess == sess.split(' ')[0])) == 1:
            pcd = adni_faq[['FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG', 'FAQMEAL',
                                                'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL']][
                (adni_faq_sub == sub[:10])&(adni_faq_sess == sess.split(' ')[0])].values[0,:]
            pcd[(pcd==1)|(pcd==2)|(pcd==3)] = 1
            pcd[(pcd==4)] = 2
            pcd[(pcd==5)] = 3
            pcd_adni_lmm_one.extend(list(pcd))
        elif sum((adni_faq_sub == sub[:10])&(adni_faq_sess == sess.split(' ')[0])) > 1:
            pcd = adni_faq[['FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG', 'FAQMEAL',
                                                'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL']][
                (adni_faq_sub == sub[:10])&(adni_faq_sess == sess.split(' ')[0])].values
            pcd[(pcd==1)|(pcd==2)|(pcd==3)] = 1
            pcd[(pcd==4)] = 2
            pcd[(pcd==5)] = 3
            selected_id = np.isnan(pcd.astype(float)).sum(-1)
            pcd_adni_lmm_one.extend(list(pcd[np.argmin(selected_id)]))
        else:
            pcd_adni_lmm_one.extend(list([np.nan for i in range(len(['FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG', 'FAQMEAL',
                                                'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL']))]))
        if sum((adni_neuroBIO_sub == int(sub[6:10]))&(adni_neuroBIO_sess == sess.split(' ')[0])) == 1:
            pcd_adni_lmm_one.extend(list(adni_neuroBIO[['ABETA42', 'TAU', 'PTAU']][
                (adni_neuroBIO_sub == int(sub[6:10]))&(adni_neuroBIO_sess == sess.split(' ')[0])].values[0,:]))
        elif sum((adni_neuroBIO_sub == int(sub[6:10]))&(adni_neuroBIO_sess == sess.split(' ')[0])) > 1:
            pcd = adni_neuroBIO[['ABETA42', 'TAU', 'PTAU']][
                (adni_neuroBIO_sub == int(sub[6:10]))&(adni_neuroBIO_sess == sess.split(' ')[0])].values
            selected_id = np.isnan(pcd.astype(float)).sum(-1)
            pcd_adni_lmm_one.extend(list(pcd[np.argmin(selected_id)]))
        else:
            pcd_adni_lmm_one.extend(list([np.nan for i in range(len(['ABETA42', 'TAU', 'PTAU']))]))
        if sum((adni_moca_sub == sub[:10])&(adni_moca_sess == sess.split(' ')[0])) == 1:
            moca_tem = adni_moca[['TRAILS', 'CUBE', 'CLOCKCON', 'CLOCKNO', 'CLOCKHAN', 'LION', 'RHINO', 'CAMEL',
                                                  'IMMT1W1', 'IMMT1W2', 'IMMT1W3', 'IMMT1W4', 'IMMT1W5', 'IMMT2W1', 'IMMT2W2', 'IMMT2W3', 
                                                  'IMMT2W4', 'IMMT2W5', 'DIGFOR', 'DIGBACK', 'LETTERS', 'SERIAL1', 'SERIAL2', 'SERIAL3', 'SERIAL4', 
                                                  'SERIAL5', 'REPEAT1', 'REPEAT2', 'FFLUENCY', 'ABSTRAN', 'ABSMEAS', 'DELW1', 'DELW2', 'DELW3', 
                                                  'DELW4', 'DELW5', 'DATE', 'MONTH', 'YEAR', 'DAY', 'PLACE', 'CITY']][
                (adni_moca_sub == sub[:10])&(adni_moca_sess == sess.split(' ')[0])].values[0,:]
            adni_moca_subtract = moca_tem[21:26].sum(-1)
            if (adni_moca_subtract>=2) & (adni_moca_subtract<=3):
                adni_moca_subtract = 2
            elif adni_moca_subtract>=4:
                adni_moca_subtract = 3
            if moca_tem[28] < 11:
                adni_moca_frq = 0
            else:
                adni_moca_frq = 1

            pcd_adni_lmm_one.extend(list([moca_tem[:5].sum(), moca_tem[5:8].sum(-1), moca_tem[29:31].sum(-1), 
                                      moca_tem[31:36].sum(-1)/3, moca_tem[36:].sum(-1), moca_tem[18:20].sum(-1) + moca_tem[20] + adni_moca_subtract, 
                                      moca_tem[26:28].sum(-1) + adni_moca_frq]))
        else:
            pcd_adni_lmm_one.extend(list([np.nan for i in range(7)]))
        if sum((adni_nps_sub == sub[:10])&(adni_nps_sess == sess.split(' ')[0])) == 1:
            pcd = adni_nps[['NPIASEV', 'NPIBSEV', 'NPICSEV', 'NPIDSEV', 'NPIESEV', 'NPIFSEV', 'NPIGSEV', 
                                                'NPIHSEV', 'NPIISEV', 'NPIJSEV', 'NPIKSEV', 'NPILSEV','NPISCORE']][
                (adni_nps_sub == sub[:10])&(adni_nps_sess == sess.split(' ')[0])].values[0,:]
            pcd[pcd == -4] = 0
            if np.isfinite(pcd[-1]):
                pcd[np.isnan(pcd)] = 0
            pcd_adni_lmm_one.extend(list(pcd))
        elif sum((adni_nps_sub == sub[:10])&(adni_nps_sess == sess.split(' ')[0])) > 1:
            pcd = adni_nps[['NPIASEV', 'NPIBSEV', 'NPICSEV', 'NPIDSEV', 'NPIESEV', 'NPIFSEV', 'NPIGSEV', 
                                                'NPIHSEV', 'NPIISEV', 'NPIJSEV', 'NPIKSEV', 'NPILSEV','NPISCORE']][
                (adni_nps_sub == sub[:10])&(adni_nps_sess == sess.split(' ')[0])].values
            pcd[pcd == -4] = 0
            for k in range(pcd.shape[0]):
                if np.isfinite(pcd[k,-1]):
                    pcd[k, np.isnan(pcd[k,:])] = 0
            selected_id = np.isnan(pcd.astype(float)).sum(-1)
            pcd_adni_lmm_one.extend(list(pcd[np.argmin(selected_id)]))
        else:
            pcd_adni_lmm_one.extend(list([np.nan for i in range(len(['NPIASEV', 'NPIBSEV', 'NPICSEV', 'NPIDSEV', 'NPIESEV', 'NPIFSEV', 'NPIGSEV', 
                                                'NPIHSEV', 'NPIISEV', 'NPIJSEV', 'NPIKSEV', 'NPILSEV','NPISCORE']))]))
        adni_cdr_sess[adni_cdr_sess == 'sc'] = 'bl'
        if sum((adni_neu_sub == sub[:10])&(adni_neu_sess == sess.split(' ')[0])) == 1:
            pcd = adni_neu[['LIMMTOTAL', 'DSPANFOR', 'DSPANFLTH', 'DSPANBAC', 'DSPANBLTH', 'CATANIMSC', 'CATANPERS', 
                                                'CATVEGESC', 'CATVGPERS', 'TRAASCOR', 'TRABSCOR', 'DIGITSCOR','LDELTOTAL', 'BNTTOTAL']][
                (adni_neu_sub == sub[:10])&(adni_neu_sess == sess.split(' ')[0])].values[0,:]
            pcd[pcd==-4] = 0
            pcd_adni_lmm_one.extend(list(pcd))
        elif sum((adni_neu_sub == sub[:10])&(adni_neu_sess == sess.split(' ')[0])) > 1:
            pcd = adni_neu[['LIMMTOTAL', 'DSPANFOR', 'DSPANFLTH', 'DSPANBAC', 'DSPANBLTH', 'CATANIMSC', 'CATANPERS', 
                                                'CATVEGESC', 'CATVGPERS', 'TRAASCOR', 'TRABSCOR', 'DIGITSCOR','LDELTOTAL', 'BNTTOTAL']][
                (adni_neu_sub == sub[:10])&(adni_neu_sess == sess.split(' ')[0])].values
            pcd[pcd==-4] = 0
            selected_id = np.isnan(pcd.astype(float)).sum(-1)
            pcd_adni_lmm_one.extend(list(pcd[np.argmin(selected_id)]))
        else:
            pcd_adni_lmm_one.extend(list([np.nan for i in range(len(['LIMMTOTAL', 'DSPANFOR', 'DSPANFLTH', 'DSPANBAC', 'DSPANBLTH', 'CATANIMSC',
                                                                  'CATANPERS', 'CATVEGESC', 'CATVGPERS', 'TRAASCOR', 'TRABSCOR', 'DIGITSCOR','LDELTOTAL',
                                                                  'BNTTOTAL']))]))
        pcd_adni_lmm.append(pcd_adni_lmm_one)
    i = i + 1
    
    
sub_adni_lmm = np.array(sub_adni_lmm)
sess_adni_lmm = np.array(sess_adni_lmm)
cluster_label_adni_lmm = np.array(cluster_label_adni_lmm)
dx_adni_lmm = np.array(dx_adni_lmm)
age_adni_lmm = np.array(age_adni_lmm)
sex_adni_lmm = np.array(sex_adni_lmm)

pcd_adni_lmm = np.array(pcd_adni_lmm)
pcd_adni_lmm[pcd_adni_lmm[:,1]!='White',1] = 0
pcd_adni_lmm[pcd_adni_lmm[:,1]=='White',1] = 1
pcd_adni_lmm = pcd_adni_lmm.astype(float)

sub_oas_lmm = []
sess_oas_lmm = []
pcd_oas_lmm = []
cluster_label_oas_lmm = []
dx_oas_lmm = []
age_oas_lmm = []
sex_oas_lmm = []
oasis_sub = pd.Series(oasis_sa_sub_sess_unique).str.split('_', expand = True).iloc[:,0]
oasis_sess = pd.Series(oasis_sa_sub_sess_unique).str.split('_', expand = True).iloc[:,-1]

j = 0
for sub in oasis_sub:
    print(j)
    sess_used = list(set(oasis_sess))
    for sess in sess_used:
        pcd_oas_lmm_one = []
        sub_oas_lmm.append(str(sub))
        cluster_label_oas_lmm.append(other_oas_pred_label_avg[j])
        dx_oas_lmm.append(other_dx_oas_encode[unique_id][j])
        age = other_age_oas[unique_id][(oasis_sub == sub)][0]
        age_sess = oasis_sess[(oasis_sub == sub)].iloc[0]
        age = age + float(sess.split('M')[-1])/12 - float(age_sess.split('M')[-1])/12
        age_oas_lmm.append(age)
        sex_oas_lmm.append(other_sex_oas[unique_id][(oasis_sub == sub)][0])
        if 'bl' in sess:
            sess_oas_lmm.append(0)
        else:
            sess_oas_lmm.append(int(sess.split('M')[-1].split(' ')[0]))
        if sum((PCD_sub == sub) & (PCD_sess == sess))== 1:
            pcd_oas_lmm_one.extend(list(PCD_data[:,np.r_[[1, 2, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25], np.arange(27,50)]][
                (PCD_sub == sub) & (PCD_sess == sess)].squeeze()))
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
            pcd_oas_lmm_one.extend(pcd_new)
        else:
            pcd_oas_lmm_one.extend([np.nan for i in range(len(['apoe', 'NPIQINF', 'DELSEV', 'HALLSEV', 'AGITSEV', 'DEPDSEV', 
                                                  'ANXSEV', 'ELATSEV', 'APASEV', 'DISNSEV', 'IRRSEV', 'MOTSEV', 'NITESEV',
                                                  'APPSEV', 'mmse', 'cdr', 'commun','homehobb', 'judgment', 'memory', 'orient', 'perscare', 'sumbox', 
                                                  'DIGIF', 'DIGIB', 'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB',
                                                  'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON']))])
        if sum((oasis_faq_sub == sub) & (oasis_faq_sess == sess))> 0:
            faq_ = oasis_faq[['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS',
                                                  'PAYATTN', 'REMDATES', 'TRAVEL']][
                (oasis_faq_sub == sub) & (oasis_faq_sess == sess)].values[0,:]
            faq_[(faq_==8)|(faq_==9)] = np.nan # recode the 8, 9

            pcd_oas_lmm_one.extend(list(faq_))
        else:
            pcd_oas_lmm_one.extend(list([np.nan for i in range(len(['BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS',
                                                  'PAYATTN', 'REMDATES', 'TRAVEL']))]))
        if sum((oasis_pet_sub == sub) & (oasis_pet_sess == sess))> 0:
            pcd_oas_lmm_one.extend(list(oasis_pet[['Centil_fBP_TOT_CORTMEAN_av45', 'Centil_fSUVR_TOT_CORTMEAN_av45', 
            'Centil_fBP_rsf_TOT_CORTMEAN_av45', 'Centil_fSUVR_rsf_TOT_CORTMEAN_av45', 'Centil_fBP_TOT_CORTMEAN_pib', 'Centil_fSUVR_TOT_CORTMEAN_pib', 
            'Centil_fBP_rsf_TOT_CORTMEAN_pib', 'Centil_fSUVR_rsf_TOT_CORTMEAN_pib']][
                (oasis_pet_sub == sub) & (oasis_pet_sess == sess)].values[0,:]))
        else:
            pcd_oas_lmm_one.extend(list([np.nan for i in range(len(['Centil_fBP_TOT_CORTMEAN_av45', 'Centil_fSUVR_TOT_CORTMEAN_av45', 
            'Centil_fBP_rsf_TOT_CORTMEAN_av45', 'Centil_fSUVR_rsf_TOT_CORTMEAN_av45', 'Centil_fBP_TOT_CORTMEAN_pib', 'Centil_fSUVR_TOT_CORTMEAN_pib', 
            'Centil_fBP_rsf_TOT_CORTMEAN_pib', 'Centil_fSUVR_rsf_TOT_CORTMEAN_pib']))]))
        pcd_oas_lmm.append(pcd_oas_lmm_one)
    j = j + 1

sub_oas_lmm = np.array(sub_oas_lmm)
sess_oas_lmm = np.array(sess_oas_lmm)
cluster_label_oas_lmm = np.array(cluster_label_oas_lmm)
dx_oas_lmm = np.array(dx_oas_lmm)
pcd_oas_lmm = np.array(pcd_oas_lmm).astype(float) 

sub_habs_lmm = []
sess_habs_lmm = []
pcd_habs_lmm = []
cluster_label_habs_lmm = []
age_habs_lmm = []
sex_habs_lmm = []
dx_habs_lmm = []
habs_sub = pd.Series(habs_sub_uni).str.split('_', expand = True).iloc[:,0]
habs_sess = pd.Series(habs_sub_uni).str.split('_', expand = True).iloc[:,-1]

habs_sub_all = pd.Series(pcd_all_habs['sub_sess']).str.split('_', expand = True).iloc[:,0]
habs_sess_all = pd.Series(pcd_all_habs['sub_sess']).str.split('_', expand = True).iloc[:,-1]
j = 0

habs_sub_uni_, uni_id = np.unique(habs_sub, return_index=True)

for sub in habs_sub_uni_:
    print(j)
    sess_used = np.unique(habs_sess)
    for sess in sess_used:
        pcd_habs_lmm_one = []
        sub_habs_lmm.append(str(sub))
        cluster_label_habs_lmm.append(other_habs_pred_label_avg[uni_id][j])
        if np.sum((habs_sub_all == sub)&(habs_sess_all == sess)) == 0:
            age = pcd_all_habs['NP_Age'][(habs_sub_all == sub)].values[0]
            age = age + float(sess) - float(habs_sess_all[habs_sub_all == sub].values[0])
            sex = pcd_all_habs['BiologicalSex'][(habs_sub_all == sub)].values[0]
            dx = pcd_all_habs['HABS_DX'][(habs_sub_all == sub)].values[0]
        else:
            age = pcd_all_habs['NP_Age'][(habs_sub_all == sub)&(habs_sess_all == sess)].values[0]
            sex = pcd_all_habs['BiologicalSex'][(habs_sub_all == sub)&(habs_sess_all == sess)].values[0]
            # dx = pcd_all_habs['HABS_DX'][(habs_sub_all == sub)&(habs_sess_all == sess)].values[0]
            dx = pcd_all_habs['HABS_DX'][(habs_sub_all == sub)].values[0] #ignore longitidinal dx change

        age_habs_lmm.append(age)
        if sex == 'F': 
            sex_habs_lmm.append(2)
        else:
            sex_habs_lmm.append(1)

        if '1.0' == sess:
            sess_habs_lmm.append(0)
        elif sess == '3.0':
            sess_habs_lmm.append(24)
        elif sess == '4.0':
            sess_habs_lmm.append(36)
        elif sess == '6.0':
            sess_habs_lmm.append(60)
        if sum((habs_sub_all == sub) & (habs_sess_all == sess))== 1:
            pcd_habs_lmm_one.extend(list(pcd_all_habs[['YrsOfEd', 'APOE_haplotype', 'BNT_30', 'CAT_Animal_Total', 'CAT_Vegetable_Total', 'Digits_Forward', 
                                                  'Digits_Backwards', 'FAS_Total', 'FCsrt_FNC', 'FCsrt_Free', 'LetterNum_Total', 'LogicMem_IL', 'LogicMem_DR',
                                                  'SRT_dr', 'SRT_cltr', 'SRT_cr', 'SRT_ltr','SRT_lts', 'SRT_mc', 'SRT_str', 'SRT_tr', 'TMT_A', 'TMT_B', 
                                                  'VFDT', 'CDR_Global', 'CDR_SB', 'CDR_Memory', 'MMSE_Total', 'MMSE_Orientation', 'MMSE_ImmRecall',
                                                  'MMSE_AttnCalc', 'MMSE_DelRecall', 'MMSE_Language', 'MMSE_Pentagons', 'GDS_Total', 'Hachinski', 'Race', 
                                                  'fdg_mean', 'ftp_mean', 'PIB_FS_DVR_FLR']][
                (habs_sub_all == sub) & (habs_sess_all == sess)].squeeze()))
        else:
            pcd_habs_lmm_one.extend([np.nan for i in range(len(['YrsOfEd', 'APOE_haplotype', 'BNT_30', 'CAT_Animal_Total', 'CAT_Vegetable_Total', 'Digits_Forward', 
                                                  'Digits_Backwards', 'FAS_Total', 'FCsrt_FNC', 'FCsrt_Free', 'LetterNum_Total', 'LogicMem_IL', 'LogicMem_DR',
                                                  'SRT_dr', 'SRT_cltr', 'SRT_cr', 'SRT_ltr','SRT_lts', 'SRT_mc', 'SRT_str', 'SRT_tr', 'TMT_A', 'TMT_B', 
                                                  'VFDT', 'CDR_Global', 'CDR_SB', 'CDR_Memory', 'MMSE_Total', 'MMSE_Orientation', 'MMSE_ImmRecall',
                                                  'MMSE_AttnCalc', 'MMSE_DelRecall', 'MMSE_Language', 'MMSE_Pentagons', 'GDS_Total', 'Hachinski', 'Race',
                                                  'fdg_mean', 'ftp_mean', 'PIB_FS_DVR_FLR']))])
        pcd_habs_lmm.append(pcd_habs_lmm_one)
        dx_habs_lmm.append(dx)
    j = j + 1

sub_habs_lmm = np.array(sub_habs_lmm)
sess_habs_lmm = np.array(sess_habs_lmm)
cluster_label_habs_lmm = np.array(cluster_label_habs_lmm)
dx_habs_lmm = np.array(dx_habs_lmm)
pcd_habs_lmm = np.array(pcd_habs_lmm).astype(float) 

sio.savemat(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\cluster\longi_symptoms\cluster_result_unique_lmm_corrected.mat', {'sub_oas': sub_oas_lmm, 'sess_oas': sess_oas_lmm,
                                                                                                'cluster_label_oas': cluster_label_oas_lmm, 'dx_oas': dx_oas_lmm, 'pcd_oas': pcd_oas_lmm.astype(float), 
                                                                                                'keys_oas': ['apoe', 'NPIQINF', 'DELSEV', 'HALLSEV', 'AGITSEV', 'DEPDSEV', 
                                                                                                              'ANXSEV', 'ELATSEV', 'APASEV', 'DISNSEV', 'IRRSEV', 'MOTSEV', 'NITESEV',
                                                                                                              'APPSEV', 'mmse', 'cdr', 'commun','homehobb', 'judgment', 'memory', 'orient', 'perscare', 'sumbox', 
                                                                                                              'DIGIF', 'DIGIB', 'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB',
                                                                                                              'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON',
                                                                                                              'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS',
                                                                                                              'PAYATTN', 'REMDATES', 'TRAVEL', 'Centil_fBP_TOT_CORTMEAN_av45', 'Centil_fSUVR_TOT_CORTMEAN_av45', 
                                                                                                              'Centil_fBP_rsf_TOT_CORTMEAN_av45', 'Centil_fSUVR_rsf_TOT_CORTMEAN_av45', 'Centil_fBP_TOT_CORTMEAN_pib', 'Centil_fSUVR_TOT_CORTMEAN_pib', 
                                                                                                              'Centil_fBP_rsf_TOT_CORTMEAN_pib', 'Centil_fSUVR_rsf_TOT_CORTMEAN_pib'],
                                                                                                'sub_adni': sub_adni_lmm, 'sess_adni': sess_adni_lmm, 'cluster_label_adni': cluster_label_adni_lmm, 
                                                                                                'dx_adni': dx_adni_lmm, 'pcd_adni': pcd_adni_lmm.astype(float), 'keys_adni': ['PTEDUCAT',
                                                                                                            'PTRACCAT', 'APOE4', 'FDG', 'PIB', 'AV45', 'FBB', 'CDRSB', 'RAVLT_immediate', 
                                                                                                            'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 
                                                                                                            'TRABSCOR', 'FAQ', 'MOCA', 'CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 
                                                                                                            'CDCARE', 'CDGLOBAL', 'FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG', 'FAQMEAL',
                                                                                                            'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL', 'ABETA42', 'TAU', 'PTAU', 'visuospatial_executive', 
                                                                                                            'naming', 'abstraction', 'delayed_recall', 'orientation', 'attention', 'language', 'NPIASEV', 'NPIBSEV', 
                                                                                                            'NPICSEV', 'NPIDSEV', 'NPIESEV', 'NPIFSEV', 'NPIGSEV', 'NPIHSEV', 'NPIISEV', 'NPIJSEV', 
                                                                                                            'NPIKSEV', 'NPILSEV','NPISCORE', 'LIMMTOTAL', 'DSPANFOR', 'DSPANFLTH', 'DSPANBAC', 
                                                                                                            'DSPANBLTH', 'CATANIMSC', 'CATANPERS', 'CATVEGESC', 'CATVGPERS', 'TRAASCOR', 
                                                                                                            'TRABSCOR', 'DIGITSCOR','LDELTOTAL', 'BNTTOTAL'], 
                                                                                                'sub_habs': sub_habs_lmm, 'sess_habs': sess_habs_lmm, 'cluster_label_habs': cluster_label_habs_lmm, 
                                                                                                'dx_habs': dx_habs_lmm, 'pcd_habs': pcd_habs_lmm.astype(float), 'keys_habs': ['YrsOfEd', 'APOE_haplotype', 
                                                                                                            'BNT_30', 'CAT_Animal_Total', 'CAT_Vegetable_Total', 'Digits_Forward', 'Digits_Backwards', 'FAS_Total',
                                                                                                            'FCsrt_FNC', 'FCsrt_Free', 'LetterNum_Total', 'LogicMem_IL', 'LogicMem_DR', 'SRT_dr', 'SRT_cltr', 
                                                                                                            'SRT_cr', 'SRT_ltr','SRT_lts', 'SRT_mc', 'SRT_str', 'SRT_tr', 'TMT_A', 'TMT_B', 'VFDT', 'CDR_Global',
                                                                                                            'CDR_SB', 'CDR_Memory', 'MMSE_Total', 'MMSE_Orientation', 'MMSE_ImmRecall', 'MMSE_AttnCalc', 
                                                                                                            'MMSE_DelRecall', 'MMSE_Language', 'MMSE_Pentagons', 'GDS_Total', 'Hachinski', 'Race',
                                                                                                            'fdg_mean', 'ftp_mean', 'PIB_FS_DVR_FLR']})

########################combat some tp with limited samples and match with another dataset
new_tp_one = [0, 12, 24, 36, 48, 60, 72, 96, 144, 192]
sub_adni_lmm_new = []
tp_adni_lmm_new = []
pcd_adni_lmm_new = []
label_adni_lmm_new = []
dx_adni_lmm_new = []

for i in range(len(other_sub_adni_unique)):
    print(i)
    for j in range(len(new_tp_one)):
        label_adni_lmm_new.append(other_adni_pred_label_avg[i])
        dx_adni_lmm_new.append(other_dx_adni[other_idx_adni_unique][i])
        tp_adni_lmm_new.append(new_tp_one[j])
        sub_adni_lmm_new.append(str(other_sub_adni_unique[i]))
        pcd = pcd_adni_lmm[(sub_adni_lmm == str(other_sub_adni_unique[i]))]
        sess_used = sess_adni_lmm[(sub_adni_lmm == str(other_sub_adni_unique[i]))]
        if new_tp_one[j] == 0:
            pcd_adni_lmm_new.append(pcd[sess_used==0].squeeze())
        elif new_tp_one[j] == 12:
            pcd_used = pcd[(sess_used==3)|(sess_used==6)|(sess_used==12)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_adni_lmm_new.append(pcd_used_raw)
        elif new_tp_one[j] == 24:
            pcd_used = pcd[(sess_used==18)|(sess_used==24)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_adni_lmm_new.append(pcd_used_raw)
        elif new_tp_one[j] == 36:
            pcd_used = pcd[(sess_used==30)|(sess_used==36)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_adni_lmm_new.append(pcd_used_raw)     
        elif new_tp_one[j] == 48:
            pcd_used = pcd[(sess_used==42)|(sess_used==48)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_adni_lmm_new.append(pcd_used_raw)    
        elif new_tp_one[j] == 60:
            pcd_used = pcd[(sess_used==54)|(sess_used==60)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_adni_lmm_new.append(pcd_used_raw)        
        elif new_tp_one[j] == 72:
            pcd_used = pcd[(sess_used==66)|(sess_used==72)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_adni_lmm_new.append(pcd_used_raw)     
        elif new_tp_one[j] == 96:
            pcd_used = pcd[(sess_used==78)|(sess_used==84)|(sess_used==90)|(sess_used==96)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_adni_lmm_new.append(pcd_used_raw)     
        elif new_tp_one[j] == 144:
            pcd_used = pcd[(sess_used==102)|(sess_used==108)|(sess_used==114)|(sess_used==120)|(sess_used==126)|(sess_used==132)|(sess_used==138)|(sess_used==144)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_adni_lmm_new.append(pcd_used_raw)     
        elif new_tp_one[j] == 192:
            pcd_used = pcd[(sess_used>144)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_adni_lmm_new.append(pcd_used_raw)            
            
sub_adni_lmm_new = np.array(sub_adni_lmm_new)    
tp_adni_lmm_new = np.array(tp_adni_lmm_new)    
pcd_adni_lmm_new = np.array(pcd_adni_lmm_new)    
label_adni_lmm_new = np.array(label_adni_lmm_new)    
dx_adni_lmm_new = np.array(dx_adni_lmm_new)    
 
sub_oas_lmm_new = []
tp_oas_lmm_new = []
pcd_oas_lmm_new = []
label_oas_lmm_new = []
dx_oas_lmm_new = []

for i in range(len(oasis_sub)):
    print(i)
    for j in range(len(new_tp_one)):
        label_oas_lmm_new.append(other_oas_pred_label_avg[i])
        dx_oas_lmm_new.append(other_dx_oas_encode[unique_id][i])
        tp_oas_lmm_new.append(new_tp_one[j])
        sub_oas_lmm_new.append(str(oasis_sub[i]))
        pcd = pcd_oas_lmm[(sub_oas_lmm == str(oasis_sub[i]))]
        sess_used = sess_oas_lmm[(sub_oas_lmm == str(oasis_sub[i]))]
        if new_tp_one[j] == 0:
            pcd_oas_lmm_new.append(pcd[sess_used==0].squeeze())
        elif new_tp_one[j] == 12:
            pcd_oas_lmm_new.append(pcd[sess_used==12].squeeze())
        elif new_tp_one[j] == 24:
            pcd_used = pcd[(sess_used==18)|(sess_used==24)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_oas_lmm_new.append(pcd_used_raw)
        elif new_tp_one[j] == 36:
            pcd_used = pcd[(sess_used==30)|(sess_used==36)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_oas_lmm_new.append(pcd_used_raw)     
        elif new_tp_one[j] == 48:
            pcd_used = pcd[(sess_used==42)|(sess_used==48)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_oas_lmm_new.append(pcd_used_raw)     
        elif new_tp_one[j] == 60:
            pcd_used = pcd[(sess_used==54)|(sess_used==60)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_oas_lmm_new.append(pcd_used_raw)     
        elif new_tp_one[j] == 72:
            pcd_used = pcd[(sess_used==66)|(sess_used==72)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_oas_lmm_new.append(pcd_used_raw)               
        elif new_tp_one[j] == 96:
            pcd_used = pcd[(sess_used==78)|(sess_used==84)|(sess_used==90)|(sess_used==96)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_oas_lmm_new.append(pcd_used_raw)     
        elif new_tp_one[j] == 144:
            pcd_used = pcd[(sess_used==102)|(sess_used==108)|(sess_used==114)|(sess_used==120)|(sess_used==126)|(sess_used==132)|(sess_used==138)|(sess_used==144)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_oas_lmm_new.append(pcd_used_raw)     
        elif new_tp_one[j] == 192:
            pcd_used = pcd[(sess_used>144)]
            pcd_used_raw = pcd_used.mean(0)
            for k in range(len(pcd_used_raw)):
                pcd_used_raw[k] = pcd_used[~np.isnan(pcd_used[:,k]),k].mean()
            pcd_oas_lmm_new.append(pcd_used_raw)            
            
sub_oas_lmm_new = np.array(sub_oas_lmm_new)    
tp_oas_lmm_new = np.array(tp_oas_lmm_new)    
pcd_oas_lmm_new = np.array(pcd_oas_lmm_new)    
label_oas_lmm_new = np.array(label_oas_lmm_new)    
dx_oas_lmm_new = np.array(dx_oas_lmm_new)    

sio.savemat(r'F:\PHD\learning\project\super_age\more_sub\SA_defined_ricado\cluster\longi_symptoms\cluster_result_unique_lmm_tpReasigned.mat', {'sub_oas': sub_oas_lmm_new, 'sess_oas': tp_oas_lmm_new,
                                                                                                'cluster_label_oas': label_oas_lmm_new, 'dx_oas': dx_oas_lmm_new, 'pcd_oas': pcd_oas_lmm_new.astype(float), 
                                                                                                'keys_oas': ['apoe', 'NPIQINF', 'DELSEV', 'HALLSEV', 'AGITSEV', 'DEPDSEV', 
                                                                                                              'ANXSEV', 'ELATSEV', 'APASEV', 'DISNSEV', 'IRRSEV', 'MOTSEV', 'NITESEV',
                                                                                                              'APPSEV', 'mmse', 'cdr', 'commun','homehobb', 'judgment', 'memory', 'orient', 'perscare', 'sumbox', 
                                                                                                              'DIGIF', 'DIGIB', 'ANIMALS', 'VEG', 'TRAILA', 'TRAILALI', 'TRAILB',
                                                                                                              'TRAILBLI', 'WAIS', 'LOGIMEM', 'MEMUNITS', 'MEMTIME', 'BOSTON',
                                                                                                              'BILLS', 'TAXES', 'SHOPPING', 'GAMES', 'STOVE', 'MEALPREP', 'EVENTS',
                                                                                                              'PAYATTN', 'REMDATES', 'TRAVEL', 'Centil_fBP_TOT_CORTMEAN_av45', 'Centil_fSUVR_TOT_CORTMEAN_av45', 
                                                                                                              'Centil_fBP_rsf_TOT_CORTMEAN_av45', 'Centil_fSUVR_rsf_TOT_CORTMEAN_av45', 'Centil_fBP_TOT_CORTMEAN_pib', 'Centil_fSUVR_TOT_CORTMEAN_pib', 
                                                                                                              'Centil_fBP_rsf_TOT_CORTMEAN_pib', 'Centil_fSUVR_rsf_TOT_CORTMEAN_pib'],
                                                                                                'sub_adni': sub_adni_lmm_new, 'sess_adni': tp_adni_lmm_new, 'cluster_label_adni': label_adni_lmm_new, 
                                                                                                'dx_adni': dx_adni_lmm_new, 'pcd_adni': pcd_adni_lmm_new.astype(float), 'keys_adni': ['PTEDUCAT',
                                                                                                            'PTRACCAT', 'APOE4', 'FDG', 'PIB', 'AV45', 'FBB', 'CDRSB', 'RAVLT_immediate', 
                                                                                                            'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL', 'DIGITSCOR', 
                                                                                                            'TRABSCOR', 'FAQ', 'MOCA', 'CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 
                                                                                                            'CDCARE', 'CDGLOBAL', 'FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG', 'FAQMEAL',
                                                                                                            'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL', 'ABETA42', 'TAU', 'PTAU', 'visuospatial_executive', 
                                                                                                            'naming', 'abstraction', 'delayed_recall', 'orientation', 'attention', 'language', 'NPIASEV', 'NPIBSEV', 
                                                                                                            'NPICSEV', 'NPIDSEV', 'NPIESEV', 'NPIFSEV', 'NPIGSEV', 'NPIHSEV', 'NPIISEV', 'NPIJSEV', 
                                                                                                            'NPIKSEV', 'NPILSEV','NPISCORE', 'LIMMTOTAL', 'DSPANFOR', 'DSPANFLTH', 'DSPANBAC', 
                                                                                                            'DSPANBLTH', 'CATANIMSC', 'CATANPERS', 'CATVEGESC', 'CATVGPERS', 'TRAASCOR', 
                                                                                                            'TRABSCOR', 'DIGITSCOR','LDELTOTAL', 'BNTTOTAL'],
                                                                                                'sub_habs': sub_habs_lmm, 'sess_habs': sess_habs_lmm, 'cluster_label_habs': cluster_label_habs_lmm, 
                                                                                                'dx_habs': dx_habs_lmm, 'pcd_habs': pcd_habs_lmm.astype(float), 'keys_habs': ['YrsOfEd', 'APOE_haplotype', 
                                                                                                            'BNT_30', 'CAT_Animal_Total', 'CAT_Vegetable_Total', 'Digits_Forward', 'Digits_Backwards', 'FAS_Total',
                                                                                                            'FCsrt_FNC', 'FCsrt_Free', 'LetterNum_Total', 'LogicMem_IL', 'LogicMem_DR', 'SRT_dr', 'SRT_cltr', 
                                                                                                            'SRT_cr', 'SRT_ltr','SRT_lts', 'SRT_mc', 'SRT_str', 'SRT_tr', 'TMT_A', 'TMT_B', 'VFDT', 'CDR_Global',
                                                                                                            'CDR_SB', 'CDR_Memory', 'MMSE_Total', 'MMSE_Orientation', 'MMSE_ImmRecall', 'MMSE_AttnCalc', 
                                                                                                            'MMSE_DelRecall', 'MMSE_Language', 'MMSE_Pentagons', 'GDS_Total', 'Hachinski', 'Race',
                                                                                                            'fdg_mean', 'ftp_mean', 'PIB_FS_DVR_FLR']})

