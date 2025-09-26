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
from datetime import datetime
import pickle
import numpy as np
import seaborn as sns
import scipy.stats as stats
import itertools

# path 지정
path = r'D:\VehicleTest\Data\2025\0. 기반기술\1. Compd 온도별 평가\Hydroplaning'
# path = r'C:\Users\HANTA\Desktop\작업\1. Compd 온도별 평가\WetBraking'

list_hyc = []
list_hys = []
dfraw, fName = ReadData(path)
for k in range(len(dfraw)):
    # 파일명에 따라 폴더 만들고 저장
    file_name = os.path.basename(fName[k])
    folder_name = os.path.splitext(file_name)[0]
    folder_path = os.path.join(os.path.dirname(fName[k]), folder_name)
    # os.makedirs(folder_path, exist_ok=True)

    # 파일명에서 데이터 끌어오기 (파일명에 따라서 변경 가능)
    parts = [part for part in folder_name.replace("-", "_").split("_")]
    # parts.append(parts[4].split('T')[1])
    # parts.append(re.search(r'W(\d+)T(\d+)', parts[4]).group(1))
    # parts.append(re.search(r'W(\d+)T(\d+)', parts[4]).group(2))

    dt = datetime.strptime(parts[-1], '%Y%m%d%H%M')
    parts[-1] = dt.strftime('%Y-%m-%d %H:%M:%S')
    columns = ['TestItem', 'ReqNo', 'TestSet', 'RoadInfo', 'CondInfo', 'AMPM', 'GroupSpec','Datetime']
    # columns = ['TestItem', 'ReqNo', 'TestSet', 'RoadInfo', 'CondInfo', 'WK', 'GroupSpec','Datetime']
    df_spec = pd.DataFrame([parts], columns=columns).reset_index(drop=True)

    testname = re.split(r"[-_]", file_name)
    if testname[0] =='hyc':
        value_hyc = pd.DataFrame({"MaxLatAcc": [dfraw[k]['MaxG']],
                                 "V1": [dfraw[k]['V1']],
                                 "V2": [dfraw[k]['V2']],
                                 "Area": [dfraw[k]['Area']]})
        df_hyc_single = pd.concat([df_spec, value_hyc], axis=1)
        list_hyc.append(df_hyc_single)
    elif testname[0] == 'hys':

        hys_tmp = dfraw[k]['Vel_Slip_df']
        targets = [5, 10, 15, 20]
        value_hys_tmp = {}
        for target in targets:
            closest_idx = (hys_tmp['SlipRatio'] - target).abs().idxmin()
            # closest_value = hys_tmp.loc[closest_idx, 'SlipRatio']
            closest_value = hys_tmp.loc[closest_idx, 'VelHorizontal']
            col_name = f'SlipRatio {target}%'
            value_hys_tmp[col_name] = closest_value

        # 결과를 하나의 행으로 된 DataFrame으로 변환
        value_hys = pd.DataFrame([value_hys_tmp])
        df_hys_single = pd.concat([df_spec, value_hys], axis=1)
        list_hys.append(df_hys_single)

# df_hyc = pd.concat(list_hyc, ignore_index=True)
# df_hys = pd.concat(list_hys, ignore_index=True)
# df_all = pd.concat([df_hyc, df_hys], axis=0, ignore_index=True)
# df_all['Datetime'] = pd.to_datetime(df_all['Datetime'])

df_list = []

# df_hyc 생성 (list_hyc가 비어있지 않은 경우에만)
if list_hyc:
    df_hyc = pd.concat(list_hyc, ignore_index=True)
    df_list.append(df_hyc)

# df_hys 생성 (list_hys가 비어있지 않은 경우에만)
if list_hys:
    df_hys = pd.concat(list_hys, ignore_index=True)
    df_list.append(df_hys)

# df_all 생성 (데이터가 있는 경우에만)
if df_list:
    df_all = pd.concat(df_list, ignore_index=True)
    df_all['Datetime'] = pd.to_datetime(df_all['Datetime'])
else:
    df_all = pd.DataFrame(columns=['Datetime'])

save_pickle = os.path.join(os.path.dirname(fName[0]), "hy_data.pkl")
with open(save_pickle, "wb") as f:
    pickle.dump(df_all, f)

plt.figure()