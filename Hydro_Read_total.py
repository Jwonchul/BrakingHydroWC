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

# path 지정
path = r'D:\VehicleTest\Data\2025\0. 기반기술\1. Compd 온도별 평가\2차'
# path = r'C:\Users\HANTA\Desktop\작업\1. Compd 온도별 평가\WetBraking'

# pickle(Performance 데이터), txt(Weather&Road 데이터) 모두 선택
dfraw, fName = ReadData(path)

file_name = os.path.basename(fName[0])
folder_name = os.path.splitext(file_name)[0]
folder_path = os.path.join(os.path.dirname(fName[0]), folder_name)
os.makedirs(folder_path, exist_ok=True)

# 모든 데이터프레임의 공통 컬럼 찾기
common_cols = set(dfraw[0].columns)
for df in dfraw[1:]:
    common_cols &= set(df.columns)

# 공통 컬럼만 선택해서 concat
df_final_total = pd.concat([df[list(common_cols)] for df in dfraw], ignore_index=True)
# df_final_total['day-time'] = df_final_total['day'].astype(str) + '-' + df_final_total['AMPM'].astype(str)

# # Compd mapping
# mapping = {'A': 'P46','B': 'P54','C': 'P61','D': 'P84',
#            'E': 'P37','F': 'P35','G': 'P25','H': 'P33','SRTT':'SRTT'}
# df_final_total['Compd'] = df_final_total['GroupSpec'].map(mapping)
#
# cond_golfA = df_final_total['Compd'].isin(['P46', 'P54', 'P37', 'P35'])  # Golf A
# cond_golfB = df_final_total['Compd'].isin(['P61', 'P84', 'P25', 'P33'])  # Golf B
# cond_SrttA = (df_final_total['Compd']=='SRTT') & (df_final_total['TestSet']=='T23')  # SRTT + T23 → Golf A
# cond_SrttB = (df_final_total['Compd']=='SRTT') & (df_final_total['TestSet']=='C11')  # SRTT + C11 → Golf B
# choices = ['Golf A', 'Golf B', 'Golf A', 'Golf B']
# df_final_total['Vehicle'] = np.select([cond_golfA, cond_golfB, cond_SrttA, cond_SrttB], choices, default=np.nan)
# # df_final_total['Vehicle'] = np.select([cond_golfA, cond_golfB, cond_SrttA, cond_SrttB], choices, default=np.nan)
#
# # 데이터 제거
# # df_final_total = df_final_total[~((df_final_total['Compd']=='P33') & (df_final_total['AMPM']=='AM') & (df_final_total['RoadInfo']=='A1C60'))]
#
# # Braking Dist 정리
# df_final = df_final_total.drop_duplicates(subset='Dist_mean', keep='first')

df_final = df_final_total
import datetime
df_final = df_final[(df_final['day'] != datetime.date(2025, 6, 18))]

# 온도 1차 추세선
color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#393b79',
              '#637939', '#8c6d31', '#843c39', '#7b4173', '#1b9e77',
              '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#a6761d']

df_list_WHC = []
df_list_DHC = []
# grouped = df_final.groupby(['RoadInfo', 'Compd'])
grouped = df_final.groupby(['TestItem', 'Compd'])
for (road, compd), group_df in grouped:
    # str =f"Processing group: RoadInfo={road}, Compd={compd}"
    str = f"Processing group: Test={road}, Compd={compd}"
    x_fit = np.linspace(0, 60, num=61)
    if road == 'hyc':
    # 플롯 또는 결과 계산 함수 실행
        df_tmp_WHC = fit_linear_models(group_df,x_fit,'WHC','Area',
                                          ['TestItem', 'Compd'],plot=False)
        df_list_WHC.append(df_tmp_WHC)

        df_tmp_DHC = fit_linear_models(group_df,x_fit,'DHC','Area',
                                          ['TestItem', 'Compd'],plot=False)
        df_list_DHC.append(df_tmp_DHC)
    elif road == 'hys':
        df_tmp_WHC = fit_linear_models(group_df, x_fit, 'WHC', 'SlipRatio 15%',
                                       ['TestItem', 'Compd'], plot=False)
        df_list_WHC.append(df_tmp_WHC)

        df_tmp_DHC = fit_linear_models(group_df, x_fit, 'DHC', 'SlipRatio 15%',
                                       ['TestItem', 'Compd'], plot=False)
        df_list_DHC.append(df_tmp_DHC)

df_WHC = pd.concat(df_list_WHC, ignore_index=True)
df_DHC = pd.concat(df_list_DHC, ignore_index=True)
df_combined = pd.concat([df_WHC, df_DHC], ignore_index=True)
df_effect = pd.merge(df_WHC, df_DHC, on=['TestItem','Compd'], how='outer')
df = pd.merge(df_final, df_effect,on=['TestItem','Compd'], how='outer')

df_hyc = df[df['TestItem']=='hyc']
# df_hys = df[df['TestItem']=='hys']
import datetime
df_hys = df[(df['TestItem'] == 'hys') &(df['day'] != datetime.date(2025, 6, 18))]
# plt_3D_Colorbar(df_hyc,'WHC_20d','WHC_Delta','WHC_R2','Temp.vs.Braking')

# df_effect['TestMethod'] = 'BRK'
group_bar(df_hyc,['WHC_10d','WHC_50d'],'Compd','TestItem','WHC_Delta','/10°C',color=color_list)
group_bar(df_hyc,['DHC_10d','DHC_50d'],'Compd','TestItem','DHC_Delta','/10°C',color=color_list)
plt_group_multi_linear(df_hyc,['TestItem', 'Compd'],'WHC','Area')
plt_group_multi_linear(df_hyc,['TestItem', 'Compd'],'DHC','Area')
plt_group_multi_linear(df_hyc,['TestItem', 'Compd'],'WHC','MaxLatAcc')
plt_group_multi_linear(df_hyc,['TestItem', 'Compd'],'DHC','MaxLatAcc')

group_bar(df_hys,['WHC_10d','WHC_50d'],'Compd','TestItem','WHC_Delta','/10°C',color=color_list)
group_bar(df_hys,['DHC_10d','DHC_50d'],'Compd','TestItem','DHC_Delta','/10°C',color=color_list)
plt_group_multi_linear(df_hys,['TestItem', 'Compd'],'WHC','SlipRatio 15%')
plt_group_multi_linear(df_hys,['TestItem', 'Compd'],'DHC','SlipRatio 15%')

## MaxG
df_Max_WHC = []
df_Max_DHC = []
# grouped = df_final.groupby(['RoadInfo', 'Compd'])
grouped = df_final.groupby(['TestItem', 'Compd'])
for (road, compd), group_df in grouped:
    # str =f"Processing group: RoadInfo={road}, Compd={compd}"
    str = f"Processing group: Test={road}, Compd={compd}"
    x_fit = np.linspace(0, 60, num=61)
    if road == 'hyc':
    # 플롯 또는 결과 계산 함수 실행
        df_tmp_Max_WHC = fit_linear_models(group_df,x_fit,'WHC','MaxLatAcc',
                                          ['TestItem', 'Compd'],plot=False)
        df_Max_WHC.append(df_tmp_Max_WHC)

        df_tmp_Max_DHC = fit_linear_models(group_df,x_fit,'DHC','MaxLatAcc',
                                          ['TestItem', 'Compd'],plot=False)
        df_Max_DHC.append(df_tmp_Max_DHC)
    elif road == 'hys':
        df_tmp_Max_WHC = fit_linear_models(group_df, x_fit, 'WHC', 'SlipRatio 20%',
                                       ['TestItem', 'Compd'], plot=False)
        df_Max_WHC.append(df_tmp_Max_WHC)

        df_tmp_Max_DHC = fit_linear_models(group_df, x_fit, 'DHC', 'SlipRatio 20%',
                                       ['TestItem', 'Compd'], plot=False)
        df_Max_DHC.append(df_tmp_Max_DHC)

df_Max_WHC = pd.concat(df_Max_WHC, ignore_index=True)
df_Max_DHC = pd.concat(df_Max_DHC, ignore_index=True)
df_Max_combined = pd.concat([df_Max_WHC, df_Max_DHC], ignore_index=True)
df_Max_effect = pd.merge(df_Max_WHC, df_Max_DHC, on=['TestItem','Compd'], how='outer')
df_Max = pd.merge(df_final, df_Max_effect,on=['TestItem','Compd'], how='outer')

df_Max_hyc = df_Max[df_Max['TestItem']=='hyc']
df_Max_hys = df_Max[df_Max['TestItem']=='hys']

# plt_3D_Colorbar(df_hyc,'WHC_20d','WHC_Delta','WHC_R2','Temp.vs.Braking')

# df_effect['TestMethod'] = 'BRK'
group_bar(df_Max_hyc,['WHC_10d','WHC_50d'],'Compd','TestItem','WHC_Delta','/10°C',color=color_list)
group_bar(df_Max_hyc,['DHC_10d','DHC_50d'],'Compd','TestItem','DHC_Delta','/10°C',color=color_list)
plt_group_multi_linear(df_hyc,['TestItem', 'Compd'],'WHC','Area')
plt_group_multi_linear(df_hyc,['TestItem', 'Compd'],'DHC','Area')
plt_group_multi_linear(df_hyc,['TestItem', 'Compd'],'WHC','MaxLatAcc')
plt_group_multi_linear(df_hyc,['TestItem', 'Compd'],'DHC','MaxLatAcc')

group_bar(df_hys,['WHC_10d','WHC_50d'],'Compd','TestItem','WHC_Delta','/10°C',color=color_list)
group_bar(df_hys,['DHC_10d','DHC_50d'],'Compd','TestItem','DHC_Delta','/10°C',color=color_list)
plt_group_multi_linear(df_hys,['TestItem', 'Compd'],'WHC','SlipRatio 15%')
plt_group_multi_linear(df_hys,['TestItem', 'Compd'],'DHC','SlipRatio 15%')


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

save_excel = os.path.join(folder_path, "df_final.xlsx")
df_final.to_excel(save_excel, index=False)

save_excel = os.path.join(folder_path, "df.xlsx")
df.to_excel(save_excel, index=False)


## HydroBRK
dfhy, fNamehy = ReadData(path)
df = dfhy[0]
plt_group_multi(df,['RoadInfo','Compd'],'Area','Dist_mean',text='perDist')
plt_group_multi(df,['RoadInfo','Compd'],'MaxLatAcc','Dist_mean',text='perDist')
plt_group_multi(df,['RoadInfo','Compd'],'Area','perDist',text='Dist_mean')
plt_group_multi(df,['RoadInfo','Compd'],'SlipRatio 15%_1st','Area',text='Dist_mean')
plt_group_multi(df,['RoadInfo','Compd'],'SlipRatio 15%_2st','Area',text='Dist_mean')
plt_group_multi(df,['RoadInfo','Compd'],'Dist_mean','perDist',text='Compd')

df_A1 = df[df['RoadInfo']=='A1C60']
df_A4 = df[df['RoadInfo']=='A4C50']
df_B1 = df[df['RoadInfo']=='B1C220']
plt_3D_Colorbar(df_A1,'Dist_mean','Area','perDist','Brk1.vs.Hydro')

plt_3D_Colorbar(df_A4,'Dist_mean','Area','perDist','Brk4(Area).vs.Hydro')
plt_3D_Colorbar(df_A4,'Dist_mean','MaxLatAcc','perDist','Brk4(MaxG).vs.Hydro')
# plt_3D_Colorbar(df_A4,'Area','perDist','Dist_mean','Brk4.vs.Hydro')
plt_3D_Colorbar(df_B1,'Dist_mean','Area','perDist','Brk(Con,Area).vs.Hydro')
plt_3D_Colorbar(df_B1,'Dist_mean','MaxLatAcc','perDist','Brk(Con,MaxG).vs.Hydro')
plt_3D_Colorbar(df_B1,'Area','perDist','Dist_mean','Brk(Con).vs.Hydro')
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
