from unoUtil.readDataFile import *
from unoUtil.unoPostProcessing import *
from Func import *
from Func_stats import *
from Func_stats_wc import *
from unoUtil.vehicleTest import makeStandardDF
from unoUtil.gpsProcessing import GPStoDistance2, changeVBOXCoord
import os
from unoUtil.unoPostProcessing import SP_MovingAverage
from unoUtil.vehicleTest._HndlTest_ import SP_CurveFit_MF
from unoUtil.unoAI.fitMinimizeOptimization import MinimizeEstimator
from unoUtil.unoAI._DynamicModel_ import TireMagicFormula
from unoUtil.readDataFile.ReadDataFile import ReadDataFile
import itertools
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
path = r'G:\hankooktire\3. 계측평가\2025\6. PG Compare\Braking\Total'
# path = r'C:\Users\HANTA\Desktop\작업\1. Compd 온도별 평가\WetBraking'

# pickle(Performance 데이터), txt(Weather&Road 데이터) 모두 선택
dfraw, fName = ReadData(path)

file_name = os.path.basename(fName[0])
folder_name = os.path.splitext(file_name)[0]
folder_path = os.path.join(os.path.dirname(fName[0]), folder_name)
# os.makedirs(folder_path, exist_ok=True)

time_data = dfraw[0]['df_time_sum'].iloc[0]
time_data22 = dfraw[0]['Front_df_AN'].iloc[0]
time_data11 = dfraw[0]['Front_Raw_Slip'].iloc[0]
time_data33 = dfraw[0]['dfbrk'].iloc[0]
plt.figure()
plt.plot(time_data33['AbsLapT'],time_data33["VelHorizontal"],'.')
plt.plot(time_data33['AbsLapT'],time_data33["wsFL"],'.r')
plt.plot(time_data33['AbsLapT'],time_data33["wsFR"],'.b')
plt.plot(time_data33['AbsLapT'],time_data33["wsRL"],'.y')
plt.plot(time_data33['AbsLapT'],time_data33["wsRR"],'.g')

plt.figure()
plt.plot(time_data['AbsLapT'],time_data["VelHorizontal"],'.')
plt.figure()
plt.plot(time_data['AbsLapT'],time_data['filterAcc'],'.')

plt.figure()
plt.plot(time_data22['Front_AN_slip'],time_data22['Front_AN_pdf'],'.')

df = dfraw[0]
testset_list = df['TestSet'].unique()

for test_value in testset_list:
    subset = df[df['TestSet'] == test_value]

    plt.figure(figsize=(10,6))

    for idx, row in subset.iterrows():
        df_inner = row['df_time_sum']  # 내부 DataFrame (AbsLapT, filterAcc 포함)

        x = df_inner['AbsLapT']
        y = df_inner['filterAcc']

        # Label = PG + RoadInfo
        label = f"{row['PG']}_{row['RoadInfo']}"

        plt.plot(x, y, label=label)

    plt.xlabel("AbsLapT")
    plt.ylabel("filterAcc")
    plt.title(f"TestSet = {test_value}")
    plt.legend()
    plt.grid()

df['PG-Road'] = df['PG'] + '_'+df['RoadInfo']
plt_group_multi(df, ['PG','TestItem'], 'Dist_mean', 'perDist', text='TestSet')

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#393b79',
              '#637939', '#8c6d31', '#843c39', '#7b4173', '#1b9e77',
              '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#a6761d']

df_tmp = df[~df['TestSet'].isin(['C1', 'C2']) & ~df['RoadInfo'].isin(['B1CM', 'B1C220'])]
df_tmp['PG-Road'] = df_tmp['PG'] + '_'+df_tmp['RoadInfo']
plt_group(df_tmp, ['PG'], 'TestItem', 'Dist_mean', 'perDist')
plt_group_multi(df_tmp, ['PG','TestItem'], 'Dist_mean', 'perDist', text='TestSet')
plt_group_multi(df_tmp, ['PG-Road','TestItem'], 'Dist_mean', 'perDist', text='TestSet')
df_ID = df_tmp[df_tmp['PG']=='ID']
plt_group_multi(df_ID, ['TestSet','RoadInfo'], 'Dist_mean', 'perDist', text='TestSet')
df_TR = df_tmp[df_tmp['PG']=='TR']
plt_group_multi(df_TR, ['TestSet','RoadInfo'], 'Dist_mean', 'perDist', text='TestSet')
df_PF = df_tmp[df_tmp['PG']=='PF']
plt_group_multi(df_PF, ['TestSet','RoadInfo'], 'Dist_mean', 'perDist', text='TestSet')


testset_list = df_tmp['PG'].unique()
plt.figure(figsize=(10,6))
for i, test_value in enumerate(testset_list):
    subset = df_tmp[df_tmp['PG'] == test_value]
    plt.plot(subset['perDist'],'.',color=color_list[i],label=test_value)
    plt.legend()

testset_list = df['TestSet'].unique()

for test_value in testset_list:
    subset = df[df['TestSet'] == test_value]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # --- 1번 subplot: Front ---
    ax = axes[0]

    for idx, row in subset.iterrows():
        df_inner = row['Front_df_AN']

        x = df_inner['Front_AN_slip']
        y = df_inner['Front_AN_pdf']

        label = f"{row['PG']}_{row['RoadInfo']}"
        ax.plot(x, y, label=label)

    ax.set_xlabel('Front_AN_slip')
    ax.set_ylabel('Front_AN_pdf')
    ax.set_title(f"[Front] TestSet = {test_value}")
    ax.grid(True)
    ax.legend()

    # --- 2번 subplot: Rear ---
    ax = axes[1]

    for idx, row in subset.iterrows():
        df_inner = row['Rear_df_AN']

        x = df_inner['Rear_AN_slip']
        y = df_inner['Rear_AN_pdf']

        label = f"{row['PG']}_{row['RoadInfo']}"
        ax.plot(x, y, label=label)

    ax.set_xlabel('Rear_AN_slip')
    ax.set_ylabel('Rear_AN_pdf')
    ax.set_title(f"[Rear] TestSet = {test_value}")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

testset_list = df['TestSet'].unique()

for test_value in testset_list:
    subset = df[df['TestSet'] == test_value]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    ax = axes[0]
    for idx, row in subset.iterrows():
        df_inner = row['Front_Raw_Slip']

        label = f"{row['PG']}_{row['RoadInfo']}"
        ax.plot(df_inner, '.',label=label)

    ax.set_title(f"[Front] TestSet = {test_value}")
    ax.grid(True)
    ax.legend()

    ax = axes[1]

    for idx, row in subset.iterrows():
        df_inner = row['Rear_Raw_Slip']

        label = f"{row['PG']}_{row['RoadInfo']}"
        ax.plot(df_inner, '.',label=label)
    ax.set_title(f"[Rear] TestSet = {test_value}")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

df_select = df[df['RoadInfo'].isin(['A1CM','A2CM','A1C60'])]
testset_list = df_select['TestSet'].unique()

for test_value in testset_list:
    subset = df_select[df_select['TestSet'] == test_value]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    ax = axes[0]

    for idx, row in subset.iterrows():
        df_inner = row['dfbrk']

        x = df_inner['AbsLapT']
        y = df_inner['wsFL']

        label = f"{row['PG']}_{row['RoadInfo']}"
        ax.plot(x, y, '.',label=label)

    ax.set_xlabel('Front_AN_slip')
    ax.set_ylabel('Front_AN_pdf')
    ax.set_title(f"[Front] TestSet = {test_value}")
    ax.grid(True)
    ax.legend()

    # --- 2번 subplot: Rear ---
    ax = axes[1]

    for idx, row in subset.iterrows():
        df_inner = row['dfbrk']

        x = df_inner['AbsLapT']
        y = df_inner['wsRL']

        label = f"{row['PG']}_{row['RoadInfo']}"
        ax.plot(x, y, '.',label=label)

    ax.set_xlabel('Rear_AN_slip')
    ax.set_ylabel('Rear_AN_pdf')
    ax.set_title(f"[Rear] TestSet = {test_value}")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()

## 그래프 추가
columns = ['wsFL', 'wsFR', 'wsRL', 'wsRR']       # y 컬럼
x_column = 'AbsLapT'                             # x 컬럼
titles = ['Front Left', 'Front Right', 'Rear Left', 'Rear Right']


def plot_characteristic_curve(df_select, groupname, comparename, **kwargs):
    plot = kwargs.get('plot', False)

    for test_value in testset_list:
        subset = df_select[df_select[groupname] == test_value]
        # subset = df_select[df_select['TestSet'] == test_value]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()  # 2x2 배열을 1차원으로 변환

        for ax, y_col, title in zip(axes, columns, titles):
            for idx, row in subset.iterrows():
                # df_inner = row['dfbrk']

                x = df_inner[x_column]
                y = df_inner[y_col]

                label = f"{row[comparename]}"
                ax.plot(x, y, '.', label=label)

            ax.set_ylabel(f"{y_col}")
            ax.set_title(f"[{title}] {groupname}={test_value}")
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel(f"{x_column}")  # 마지막 subplot에 x축 라벨
        plt.tight_layout()




testset_list = df['TestSet'].unique()

for test_value in testset_list:
    subset = df[df['TestSet'] == test_value]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # --- 1번 subplot: Front ---
    ax = axes[0]

    for idx, row in subset.iterrows():
        df_inner = row['dfbrk']

        x = df_inner['AbsLapT']
        y = df_inner['filterAcc']

        label = f"{row['PG']}_{row['RoadInfo']}"
        ax.plot(x, y, label=label)

    ax.set_xlabel('Front_AN_slip')
    ax.set_ylabel('Front_AN_pdf')
    ax.set_title(f"[Front] TestSet = {test_value}")
    ax.grid(True)
    ax.legend()

    # --- 2번 subplot: Rear ---
    ax = axes[1]

    for idx, row in subset.iterrows():
        df_inner = row['dfbrk']

        x = df_inner['Rear_AN_slip']
        y = df_inner['Rear_AN_pdf']

        label = f"{row['PG']}_{row['RoadInfo']}"
        ax.plot(x, y, label=label)

    ax.set_xlabel('Rear_AN_slip')
    ax.set_ylabel('Rear_AN_pdf')
    ax.set_title(f"[Rear] TestSet = {test_value}")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()


# 모든 데이터프레임의 공통 컬럼 찾기
common_cols = set(dfraw[0].columns)
for df in dfraw[1:]:
    common_cols &= set(df.columns)

# 공통 컬럼만 선택해서 concat
df_final_total = pd.concat([df[list(common_cols)] for df in dfraw], ignore_index=True)
if "day" not in df_final_total.columns:
    df_final_total["day"] = df_final_total["Datetime"].dt.date
df_final_total['day-time'] = df_final_total['day'].astype(str) + '-' + df_final_total['AMPM'].astype(str)

# 데이터 제거
# df_final_total = df_final_total[~((df_final_total['Compd']=='P33') & (df_final_total['AMPM']=='AM') & (df_final_total['RoadInfo']=='A1C60'))]

# Braking Dist 정리
df_final = df_final_total.drop_duplicates(subset='Dist_mean', keep='first')
df_final["Month"] = df_final["Datetime"].dt.month

# ## 제어 구간 확인
# Fslip_chname = ['Front_MaxSlip','Front_Kurtosis','Front_Skew','Dist_mean']
# tmpdata = df_final[Fslip_chname]
# selectdata = tmpdata.iloc[[0]]
# x, y = slip_stats_distribution(selectdata)
# plt.figure()
# plt.plot(x, y, linestyle='-')

# offset Braking
# df_offset_WHC = data_correction(df_final,['RoadInfo', 'Compd'],'WHC','Dist_mean',[10,20])
# plt_group_multi_linear(df_offset_WHC,['RoadInfo', 'Compd'],'WHC','Dist_mean_offset')
# plt_group_multi_linear(df_offset_WHC,['RoadInfo'],'WHC','Dist_mean_offset')
df_final.rename(columns={'AMPM': 'Compd'}, inplace=True)

# 온도 1차 추세선
color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#393b79',
              '#637939', '#8c6d31', '#843c39', '#7b4173', '#1b9e77',
              '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#a6761d']

group_bar(df_final,['Dist_min','Dist_max'],'Compd','RoadInfo','perDist','m',color=color_list)
group_bar(df_final,['Dist_min','Dist_max'],'RoadInfo','Compd','perDist','m',color=color_list)
group_bar(df_final,['Front_MaxSlip','Rear_MaxSlip'],'RoadInfo','Compd','perDist','/',color=color_list)
plt_group_multi(df_final,['WaterDepth','Compd'],'Dist_mean','Front_MaxSlip')

df_PIndex_WHC = plt_group_normal(df_final, ['RoadInfo', 'Compd'],
                                                'Compd', 'WHC', [10,20], 'Dist_mean')

df_WHC_brkdist = select_group_data(df_final, ['RoadInfo', 'Compd'],'Compd', 'WHC', [10,20])

plt_group_multi_linear(df_PIndex_WHC,['RoadInfo', 'Compd'],'WHC','Dist_mean_norm')
plt_group_multi_linear(df_PIndex_WHC,['RoadInfo'],'WHC','Dist_mean_norm')

# ## 제어 vs 제동
# plt_group_multi_linear(df_PIndex_WHC,['RoadInfo', 'Compd'],'Front_MaxSlip','Dist_mean_norm')
# plt_group_multi_linear(df_final,['RoadInfo', 'Compd'],'Front_MaxSlip','WHC')
# plt_group_multi_linear(df_final,['RoadInfo', 'Compd'],'Front_MaxSlip','Dist_mean')
# plt_group_multi_linear(df_final,['RoadInfo', 'Compd'],'Rear_MaxSlip','Dist_mean')
# df_final["Sum_MaxSlip"] = df_final["Front_MaxSlip"]+df_final["Rear_MaxSlip"]
# plt_3D_Colorbar(df_final,'Front_MaxSlip','Dist_mean','WHC','MaxSlip. vs.Braking')
# plt_3D_Colorbar(df_final,'Rear_MaxSlip','Dist_mean','WHC','MaxSlip. vs.Braking')
# plt_3D_Colorbar(df_final,'Sum_MaxSlip','Dist_mean','WHC','MaxSlip. vs.Braking')
# plt_3D_Colorbar(df_final,'Front_MaxSlip','Rear_MaxSlip','Dist_mean','MaxSlip. vs.Braking')
# plt_3D_Colorbar(df_PIndex_WHC,'Front_MaxSlip','Dist_mean_norm','WHC','MaxSlip. vs.Braking')

# df_offset_DHC = data_correction(df_final,['RoadInfo', 'Compd'],'DHC','Dist_mean',[20,30])
# plt_group_multi_linear(df_offset_DHC,['RoadInfo', 'Compd'],'DHC','Dist_mean_offset')
# plt_group_multi_linear(df_offset_DHC,['RoadInfo'],'DHC','Dist_mean_offset')

df_PIndex_DHC = plt_group_normal(df_final, ['RoadInfo', 'Compd'], 'Compd', 'DHC', [20,30], 'Dist_mean')
plt_group_multi_linear(df_PIndex_DHC,['RoadInfo', 'Compd'],'DHC','Dist_mean_norm')
plt_group_multi_linear(df_PIndex_DHC,['RoadInfo'],'DHC','Dist_mean_norm')

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

df = pd.merge(df_final, df_effect, on=['RoadInfo','Compd'], how='outer')

df_WHC_brkdist = select_group_data(df, ['RoadInfo', 'Compd'],'Compd', 'WHC', [10,20])
df_WHC_brkdist_high = select_group_data(df, ['RoadInfo', 'Compd'],'Compd', 'WHC', [30,40])

plt_3D_Colorbar(df_effect,'WHC_20d','WHC_Delta','WHC_R2','Temp.vs.Braking')

df_effect['TestMethod'] = 'BRK'
group_bar(df_effect,['WHC_10d','WHC_40d'],'Compd','RoadInfo','WHC_Delta','m/10°C',color=color_list)
group_bar(df_effect,['WHC_10d','WHC_40d'],'RoadInfo','Compd','WHC_Delta','m/10°C',color=color_list)
group_bar(df_effect,['DHC_10d','DHC_50d'],'Compd','RoadInfo','DHC_Delta','m/10°C',color=color_list)
group_bar(df_effect,['DHC_10d','DHC_50d'],'RoadInfo','Compd','DHC_Delta','m/10°C',color=color_list)

df_effect_srtt = df_effect[df_effect['Compd']!='SRTT']
group_bar(df_effect_srtt,['WHC_10d','WHC_40d'],'Compd','RoadInfo','WHC_Delta','m',color=color_list)
group_bar(df_effect_srtt,['WHC_10d','WHC_30d'],'Compd','RoadInfo','WHC_Delta','m',color=color_list,grpplot=True)
# group_bar(df_effect_srtt,['WHC_10d','WHC_40d'],'Compd','RoadInfo','WHC_Delta','m/10°C',color=color_list)
# group_bar(df_effect_srtt,['WHC_10d','WHC_40d'],'Compd','RoadInfo',
#           'WHC_Delta','m/10°C',color=color_list,grpplot=True)
# group_bar(df_effect_srtt,['WHC_10d','DHC_50d'],'Compd','RoadInfo','DHC_Delta','m/10°C',color=color_list)
group_bar(df_effect_srtt,['DHC_20d','DHC_50d'],'Compd','RoadInfo','DHC_Delta','m',color=color_list)
# group_bar(df_effect_srtt,['WHC_10d','DHC_50d'],'Compd','RoadInfo','DHC_Delta',color=color_list,grpplot=True)

plt_group(df_effect, ['RoadInfo','Compd'], 'RoadInfo', 'DHC_Delta', 'WHC_Delta')
plt_group(df_effect, ['RoadInfo'], 'TestMethod', 'DHC_Delta', 'WHC_Delta')
plt_group_multi(df_effect,['RoadInfo'],'DHC_Delta','WHC_Delta')
single_bar(df_effect,'Compd','DHC_Delta','RoadInfo',plot=True,color=color_list,pltname='DryTemperature')
single_bar(df_effect,'Compd','WHC_Delta','RoadInfo',plot=True,color=color_list,pltname='WetTemperature')


plt_group(df_effect_srtt, ['RoadInfo','Compd'], 'RoadInfo', 'DHC_Delta', 'Dist_mean')

# # Road 구분
# spec_list = sorted(df_final['RoadInfo'].unique())
# n = len(spec_list)
# for idx, spec_val in enumerate(spec_list):
#     df_idx = df_final[df_final['RoadInfo'] == spec_val]
#     plt_group(df_idx, ['Compd','day','day-time'], 'day', 'WHC', 'Per_Dist_mean')
#     plt.suptitle(f'{spec_val}', fontsize=16)
#     plt.tight_layout()
#     plt_group(df_idx, ['Compd','day','day-time'], 'day', 'WHC', 'Dist_mean',z='Dist_srtt')
#     plt.suptitle(f'{spec_val}', fontsize=16)
#     plt.tight_layout()
#     plt_group(df_idx, ['Compd','day','day-time'], 'GroupSpec', 'WHC', 'Dist_mean',z='Dist_srtt')
#     plt.suptitle(f'{spec_val}', fontsize=16)
#     plt.tight_layout()

# 경향성 분석
# plt_group(df_final, ['RoadInfo','Compd'], 'RoadInfo', 'WHC', 'Dist_mean')
# plt_group(df_final, ['RoadInfo','GroupSpec'], 'RoadInfo', 'WHC', 'Dist_mean')
# plt_group_multi(df_final,['RoadInfo', 'Compd'],'WHC','Dist_mean')
plt_group_multi_linear(df_final,['RoadInfo', 'Compd'],'WHC','Dist_mean')
plt_group_multi_linear(df_final,['RoadInfo', 'Compd'],'Temperature','Dist_mean')

# plt_group(df_final, ['RoadInfo','Compd'], 'RoadInfo', 'DHC', 'Dist_mean')
# plt_group(df_final, ['RoadInfo','GroupSpec'], 'RoadInfo', 'DHC', 'Dist_mean')
# plt_group_multi(df_final,['RoadInfo', 'Compd'],'DHC','Dist_mean')
plt_group_multi_linear(df_final,['RoadInfo', 'Compd'],'DHC','Dist_mean')



plt_group_multi(df_final,['RoadInfo'],'perDist','Dist_mean')
plt_group_multi(df_final,['RoadInfo', 'Compd'],'perDist','Dist_mean')
plt_group_multi(df_final,['Month','RoadInfo','Compd'],'perDist','Dist_mean')

# plt_group_multi(df_final,['RoadInfo', 'Compd'],'WHC','Dist_mean',text='day-time')
# plt_group_multi(df_final,['RoadInfo', 'Compd'],'DHC','Dist_mean',text='day-time')
# plt_group_multi(df_final, ['RoadInfo','day','AMPM'],'WHC','Dist_mean',text='day')
# plt_group_multi(df_final, ['RoadInfo','Compd'],'WHC','Dist_mean',text='AMPM')
# plt_group_multi(df_final,['RoadInfo', 'Compd', 'day'],'WHC','Dist_mean',text='AMPM')
# plt_group_multi(df_final,['RoadInfo', 'Compd', 'day'],'WHC','Per_Dist_mean',text='AMPM')
# plt_group_multi(df_final,['RoadInfo', 'Compd'],'Front_Kurtosis','Dist_mean',text='AMPM')

## 기상 통계값
wt_cols = ['WHC', 'DHC', 'Temperature', 'WS']
# groupby 및 통계값 계산 (예: 평균)
grouped_stats = df_final.groupby('day-time')[wt_cols].agg(['mean', 'std', 'min', 'max'])
folder_path = os.path.dirname(fName[0])
save_weather = os.path.join(folder_path, "weahterdf.xlsx")
grouped_stats.to_excel(save_weather, index=True)
# tmptmp = df_final[df_final['day']!='2025-06-26']

# SRTT 경향성
df_srtt = df_final[df_final['GroupSpec']=='SRTT']
# plt_group_multi(df_srtt, ['RoadInfo','Vehicle'],'WHC','Dist_mean',text='day-time')
# plt_group_multi(df_srtt, ['RoadInfo','Vehicle'],'DHC','Dist_mean',text='day-time')
plt_group_multi(df_srtt, ['RoadInfo','Vehicle'],'WHC','Dist_mean')
plt_group_multi(df_srtt, ['RoadInfo','Vehicle'],'DHC','Dist_mean')
plt_group_multi(df_srtt,['WaterDepth','Month'],'perDist','Dist_mean')
plt_group_multi(df_srtt,['WaterDepth','RoadInfo'],'perDist','Dist_mean')

Fslip_chname = ['Front_MaxSlip','Front_Kurtosis','Front_Skew']
# plt_group_slipstats_single(df_srtt,['RoadInfo', 'Vehicle'],Fslip_chname) # 각각 그래프 그리기
plt_group_slipstats(df_srtt,['RoadInfo', 'Vehicle'],Fslip_chname)

Rslip_chname = ['Rear_MaxSlip','Rear_Kurtosis','Rear_Skew']
plt_group_slipstats(df_srtt,['RoadInfo',  'Vehicle'],Rslip_chname)

## SRTT 제동거리 평균
grouped_stats = (df_srtt.groupby(['RoadInfo', 'Vehicle'])[['Dist_mean']].agg(['mean', 'max', 'min']))
grouped_stats.columns = ['Dist_mean_' + stat for stat in ['mean', 'max', 'min']]
grouped_stats = grouped_stats.reset_index()
df[df['Vehicle']=='Golf A']['Compd'].unique()
df[df['Vehicle']=='Golf B']['Compd'].unique()

## Partial Dist
df_spec = df_final[df_final['GroupSpec']!='SRTT']
plt_group_multi(df_spec,['WaterDepth','RoadInfo'],'perDist','Dist_mean',text='Compd')
plt_group_multi(df_spec,['WaterDepth','RoadInfo'],'perDist','Dist_mean')
plt_group_multi(df_spec,['RoadInfo','Compd'],'perDist','Dist_mean')
plt_3D_Colorbar(df_spec,'perDist','Dist_mean','WHC','Temp.vs.Braking')

df_asp1 = df_spec[df_spec['RoadInfo']=='A1C60']
plt_group_multi(df_asp1,['WaterDepth','Month'],'perDist','Dist_mean',text='Compd')
plt_group_multi(df_asp1,['WaterDepth','Compd'],'perDist','Dist_mean',text='Compd')

df_asp4 = df_spec[df_spec['RoadInfo']=='A4C50']
plt_group_multi(df_asp4,['WaterDepth','Month'],'perDist','Dist_mean',text='Compd')
plt_group_multi(df_asp4,['WaterDepth','Compd'],'perDist','Dist_mean',text='Compd')

df_con1 = df_spec[df_spec['RoadInfo']=='B1C220']
plt_group_multi(df_con1,['WaterDepth','Month'],'perDist','Dist_mean',text='Compd')
plt_group_multi(df_con1,['WaterDepth','Compd'],'perDist','Dist_mean',text='Compd')


df_DHC_20 = df_spec[(df_spec['DHC'] >= 20) & (df_spec['DHC'] <= 30)]
plt_group_multi(df_DHC_20,['WaterDepth','RoadInfo'],'perDist','Dist_mean')
plt_group_multi(df_DHC_20,['RoadInfo','Compd'],'perDist','Dist_mean')


folder_path = os.path.dirname(fName[0])
for fig_num in plt.get_fignums():  # 현재 생성된 모든 figure 번호 가져오기
    fig = plt.figure(fig_num)  # 해당 figure 불러오기
    axes = fig.get_axes()

    fig_title = f"figure_{fig_num}"  # 아무 제목도 없으면 figure 번호 사용

    # 파일 이름에 사용할 수 없는 문자 제거 (예: 파일명 오류 방지)
    fig_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in fig_title)

    # 저장 경로 설정
    save_path = os.path.join(folder_path, f"{fig_title}.png")
    fig.savefig(save_path)  # 그래프 저장
    print(f"Saved: {save_path}")

plt.close('all')

save_pickle = os.path.join(os.path.dirname(fName[0]), "Brk_data_total.pkl")
with open(save_pickle, "wb") as f:
    pickle.dump(df, f)

save_excel = os.path.join(folder_path, "df_final.xlsx")
df_final.to_excel(save_excel, index=False)

save_excel = os.path.join(folder_path, "df.xlsx")
df.to_excel(save_excel, index=False)

save_excel = os.path.join(folder_path, "1020brkdist.xlsx")
df_WHC_brkdist.to_excel(save_excel, index=False)

save_excel = os.path.join(folder_path, "3040brkdist.xlsx")
df_WHC_brkdist_high.to_excel(save_excel, index=False)

print(f"Analysis End: {file_name}")

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
