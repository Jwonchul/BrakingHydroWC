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
if "day" not in df_final_total.columns:
    df_final_total["day"] = df_final_total["Datetime"].dt.date
df_final_total['day-time'] = df_final_total['day'].astype(str) + '-' + df_final_total['AMPM'].astype(str)

df_sorted = df_final_total[sorted(df_final_total.columns, key=lambda x: str(x))]

# df_result = df_sorted.drop_duplicates(subset='Dist_mean', keep='first')
df_result = df_sorted.drop_duplicates(subset='DHC_Delta', keep='first')
plt_group_multi(df_result ,['TestItem'],'DHC_Delta','WHC_Delta',text='Compd')

