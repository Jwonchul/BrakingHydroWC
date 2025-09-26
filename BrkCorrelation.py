
from Func import *
from Func_stats import *
from Func_stats_wc import *
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import pickle
import numpy as np
import seaborn as sns
import scipy.stats as stats
import itertools
import math

# path 지정
path = r'D:\VehicleTest\Data\2025\0. 기반기술\1. Compd 온도별 평가\2차'
# path = r'C:\Users\HANTA\Desktop\작업\1. Compd 온도별 평가\WetBraking'

# pickle(Performance 데이터), txt(Weather&Road 데이터) 모두 선택
dfraw, fName = ReadData(path)

file_name = os.path.basename(fName[0])
folder_name = os.path.splitext(file_name)[0]
folder_path = os.path.join(os.path.dirname(fName[0]), folder_name)
# os.makedirs(folder_path, exist_ok=True)

# 모든 데이터프레임의 공통 컬럼 찾기
common_cols = set(dfraw[0].columns)
for df in dfraw[1:]:
    common_cols &= set(df.columns)

# 공통 컬럼만 선택해서 concat
df_final_total = pd.concat([df[list(common_cols)] for df in dfraw], ignore_index=True)
df_final_total['day-time'] = df_final_total['day'].astype(str) + '-' + df_final_total['AMPM'].astype(str)

# Compd mapping
mapping = {'A': 'P46','B': 'P54','C': 'P61','D': 'P84',
           'E': 'P37','F': 'P35','G': 'P25','H': 'P33','SRTT':'SRTT'}
df_final_total['Compd'] = df_final_total['GroupSpec'].map(mapping)

cond_golfA = df_final_total['Compd'].isin(['P46', 'P54', 'P37', 'P35'])  # Golf A
cond_golfB = df_final_total['Compd'].isin(['P61', 'P84', 'P25', 'P33'])  # Golf B
cond_SrttA = (df_final_total['Compd']=='SRTT') & (df_final_total['TestSet']=='T23')  # SRTT + T23 → Golf A
cond_SrttB = (df_final_total['Compd']=='SRTT') & (df_final_total['TestSet']=='C11')  # SRTT + C11 → Golf B
choices = ['Golf A', 'Golf B', 'Golf A', 'Golf B']
df_final_total['Vehicle'] = np.select([cond_golfA, cond_golfB, cond_SrttA, cond_SrttB], choices, default=np.nan)

# Braking Dist 정리
df_final = df_final_total.drop_duplicates(subset='Dist_mean', keep='first')
df_final["Month"] = df_final["Datetime"].dt.month

# 온도 1차 추세선
color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#393b79',
              '#637939', '#8c6d31', '#843c39', '#7b4173', '#1b9e77',
              '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#a6761d']

df_list_WHC = []
df_list_DHC = []
grouped = df_final.groupby(['RoadInfo', 'Compd'])
for (road, compd), group_df in grouped:
    str =f"Processing group: RoadInfo={road}, Compd={compd}"
    # 플롯 또는 결과 계산 함수 실행
    x_fit = np.linspace(0, 60, num=61)
    df_tmp_WHC = fit_linear_models(group_df,x_fit,'WHC','Dist_mean',
                                      ['RoadInfo', 'Compd'],plot=False)
    df_list_WHC.append(df_tmp_WHC)

    df_tmp_DHC = fit_linear_models(group_df,x_fit,'DHC','Dist_mean',
                                      ['RoadInfo', 'Compd'],plot=False)
    df_list_DHC.append(df_tmp_DHC)

df_WHC = pd.concat(df_list_WHC, ignore_index=True)
df_DHC = pd.concat(df_list_DHC, ignore_index=True)
df_combined = pd.concat([df_WHC, df_DHC], ignore_index=True)
df_effect = pd.merge(df_WHC, df_DHC, on=['RoadInfo','Compd'], how='outer')
df_effect['TestMethod'] = 'BRK'

df = pd.merge(df_final, df_effect, on=['RoadInfo','Compd'], how='outer')

df_effect_srtt = df_effect[df_effect['Compd']!='SRTT']

# SRTT 경향성
df_srtt = df_final[df_final['GroupSpec']=='SRTT']

# Partial Dist
df_spec = df_final[df_final['GroupSpec']!='SRTT']

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
##
df_spec = df_spec.sort_values('Datetime')
df_srtt = df_srtt.sort_values('Datetime')
df_srtt_rename = df_srtt.rename(columns={'meanAx': 'meanAx_srtt'})
# df_merged = pd.merge_asof(df_spec,df_srtt_rename[['Dist_mean', 'meanAx_srtt']],
#                           on='Dist_mean',direction='nearest')
df_spec['meanAx_srtt'] = df_spec['Dist_srtt'].map(df_srtt_rename.set_index('Dist_mean')['meanAx_srtt'])

df_merged_asp = df_spec[df_spec['RoadInfo']!='B1C220']

df_list_mu = []
grouped = df_merged_asp.groupby(['Compd'])
for compd, group_df in grouped:
    # 플롯 또는 결과 계산 함수 실행
    x_fit = np.linspace(0.6, 0.9, num=60)
    df_tmp_mu = fit_linear_models_srtt(group_df,x_fit,'meanAx_srtt','Dist_mean',
                                      ['Compd'],plot=False)
    df_list_mu.append(df_tmp_mu)

df_mu = pd.concat(df_list_mu, ignore_index=True)
df_effect_mu = pd.merge(df_merged_asp, df_mu, on=['Compd'], how='outer')
df_effect = df_effect_mu.drop_duplicates(subset='meanAx_srtt_0.6d', keep='first')
df_effect['meanAx_srtt_Delta']=df_effect['meanAx_srtt_Delta'].abs()
group_bar(df_effect,['meanAx_srtt_0.6d','meanAx_srtt_0.75d'],'Compd',
          'RoadInfo','meanAx_srtt_Delta','m',color=color_list)
group_bar(df_effect,['meanAx_srtt_0.75d','meanAx_srtt_0.65d'],'Compd',
          'RoadInfo','meanAx_srtt_Delta','m',color=color_list)
plt_group_multi_linear(df_merged_asp,['WaterDepth','Compd'],'meanAx_srtt','Dist_mean')

df_DHC = pd.concat(df_list_DHC, ignore_index=True)
df_combined = pd.concat([df_WHC, df_DHC], ignore_index=True)
df_effect = pd.merge(df_WHC, df_DHC, on=['RoadInfo','Compd'], how='outer')


df_merge_gp = df_merged_asp.groupby(['Compd'])

df_srtt[['Dist_srtt','Dist_mean']]

plt_group_multi_linear(df_merged_asp,['WaterDepth','Compd'],'meanAx_srtt','Dist_mean')
plt.figure()
for compd, grp_data in df_merge_gp:
    plt.figure()
    plt.plot(grp_data['meanAx_srtt'],grp_data['Dist_mean'],'.')
    plt.title(compd)

grouped = df_spec.groupby(['Compd', 'RoadInfo'])['Dist_mean'].sum().reset_index()
grouped['Percentage'] = grouped.groupby('Compd')['Dist_mean'].transform(lambda x: x / x.sum() * 100)


df_cut = df[df['GroupSpec']!='SRTT']

import statsmodels.api as sm
from statsmodels.formula.api import ols

# # model = ols("Dist ~ C(Road) + Temp + C(Vehicle)", data=df_cut).fit()
# model = ols("Dist_mean ~ C(RoadInfo) + WHC + C(Compd)", data=df_cut).fit()
# # model = ols("Dist_mean ~ C(RoadInfo) + DHC + C(Compd)", data=df_cut_tmp).fit()
# # ANOVA 테이블
# anova_table = sm.stats.anova_lm(model, typ=2)
# anova_table["contribution"] = anova_table["sum_sq"] / anova_table["sum_sq"].sum() * 100

# df_cut['WHC'].min()
# df_cut['WHC'].max()
temp_ranges = [(0,20),(20,40),(0,40)]
results = {}

for low, high in temp_ranges:
    subset = df_cut[(df_cut['WHC'] >= low) & (df_cut['WHC'] <= high)]
    # model = ols("Dist_mean ~ C(RoadInfo) + WHC + C(Compd)", data=subset).fit()
    model = ols("Dist_mean ~ C(RoadInfo) + WHC + C(Compd)+ C(Vehicle)", data=subset).fit()
    # model = ols("Dist ~ C(Road) + C(Vehicle) + Temp", data=subset).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_table["contribution"] = anova_table["sum_sq"] / anova_table["sum_sq"].sum() * 100
    results[f"{low}-{high}"] = anova_table


for vehicle, group in df_cut.groupby("Compd"):
    model = ols('Dist_mean ~ C(RoadInfo)', data=group).fit()
    print(model.summary())

model = ols("Dist_mean ~ C(Compd)*C(RoadInfo)*WHC", data=df_cut).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

#
# ## 제동거리 분포 계산
# df_spec = df_final_total[df_final_total['GroupSpec']!='SRTT']
# group_cols = ['Compd','RoadInfo','Dist_mean']
# colname = 'Dist_detail'
# stats_t = StatsAnalyzer.compute_stats(df_spec, group_cols, colname, method='tdist', confidence=95)
# df_t = df_spec.merge(stats_t, on=group_cols, how='left')
#
# ## Spec 경향성
# plt_stats_bar_v2(df_t,'Compd','RoadInfo','Dist_mean',num='WHC',numunit='°C')
# plt_stats_bar_v2(df_t,'GroupSpec','RoadInfo','Dist_mean',st='AMPM',num='WHC',numunit='°C')
# plt_stats_bar_v2(df_t,'GroupSpec','RoadInfo','Dist_mean',st='GroupSpec',num='Per_Dist_mean',numunit='%')
# plt_group_multi(df_spec,['RoadInfo', 'GroupSpec', 'day'],'WHC','Dist_mean',text='AMPM')
