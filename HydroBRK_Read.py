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

## HydroBRK
dfhy, fNamehy = ReadData(path)
df = dfhy[0]
df['Mode'] = 1
plt_group_multi(df,['RoadInfo','Compd'],'Area','Dist_mean',text='Compd')
plt_group_multi(df,['RoadInfo','Compd'],'SlipRatio 15%_1st','Dist_mean',text='Compd')
plt_group_multi(df,['RoadInfo','Compd'],'SlipRatio 15%_2st','Dist_mean',text='Compd')

plt_group_multi(df,['RoadInfo','Compd'],'Area','Dist_mean',text='perDist')
plt_group_multi(df,['RoadInfo','Compd'],'MaxLatAcc','Dist_mean',text='perDist')
plt_group_multi(df,['RoadInfo','Compd'],'Area','perDist',text='Dist_mean')

plt_group_multi(df,['RoadInfo','Compd'],'SlipRatio 15%_1st','Area',text='Dist_mean')
plt_group_multi(df,['RoadInfo','Compd'],'SlipRatio 15%_1st','perDist',text='Dist_mean')
plt_group_multi_linear(df,['RoadInfo'],'SlipRatio 15%_1st','perDist',text='Dist_mean')

plt_group_multi(df,['RoadInfo','Compd'],'SlipRatio 15%_2st','Area',text='Dist_mean')
plt_group_multi(df,['RoadInfo','Compd'],'SlipRatio 15%_2st','perDist',text='Dist_mean')
plt_group_multi(df,['RoadInfo','Compd'],'Dist_mean','perDist',text='Compd')


# plt_group_multi(df,['Mode','RoadInfo'],'Dist_mean','perDist',text='Compd')
plt_group_multi(df,['Mode','RoadInfo'],'perDist','Dist_mean',text='Compd')
plt_group_multi(df,['Mode','RoadInfo'],'perDist','Dist_mean',text='Compd')
plt_group_multi(df,['Season','RoadInfo'],'perDist','Dist_mean',text='Compd')


plt_group_multi(df,['RoadInfo','Compd'],'Dist_mean','BRK_Delta',text='Compd')

folder_path = os.path.dirname(fNamehy[0])
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

plt.figure()
#
# df_A1 = df[df['RoadInfo']=='A1C60']
# df_A4 = df[df['RoadInfo']=='A4C50']
# df_B1 = df[df['RoadInfo']=='B1C220']
# plt_3D_Colorbar(df_A1,'Dist_mean','Area','perDist','Brk1.vs.Hydro')
#
# plt_3D_Colorbar(df_A4,'Dist_mean','Area','perDist','Brk4(Area).vs.Hydro')
# plt_3D_Colorbar(df_A4,'Dist_mean','MaxLatAcc','perDist','Brk4(MaxG).vs.Hydro')
# # plt_3D_Colorbar(df_A4,'Area','perDist','Dist_mean','Brk4.vs.Hydro')
# plt_3D_Colorbar(df_B1,'Dist_mean','Area','perDist','Brk(Con,Area).vs.Hydro')
# plt_3D_Colorbar(df_B1,'Dist_mean','MaxLatAcc','perDist','Brk(Con,MaxG).vs.Hydro')
# plt_3D_Colorbar(df_B1,'Area','perDist','Dist_mean','Brk(Con).vs.Hydro')
# #
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
