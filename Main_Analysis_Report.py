import matplotlib.pyplot as plt
from unoUtil.readDataFile.ReadDataFile import ReadDataFile
from unoUtil.vehicleTest._BRKTest_ import *
from unoUtil.unoPostProcessing import SP_MovingAverage
import os
from scipy.stats import skew, kurtosis
import re
import pickle
from Func_stats import *
from scipy import stats
import io

path = r'D:\VehicleTest\Data\2025\0. 기반기술\1. Compd 온도별 평가'
rawdf, fName = ReadDataFile(path, filter='*.vbo', reftime={'Time': 9}, date=True, headendline=300)

## 기본 Setting
select_columns = ['TestItem', 'ReqNo', 'TestSet', 'RoadInfo', 'CondInfo', 'AMPM', 'GroupSpec']
# select_columns = ['TestItem', 'ReqNo', 'TestSet', 'RoadInfo', 'CondInfo', 'AMPM', 'GroupSpec', 'WaterDepth', 'TempSEP']
select_plt_col = ['VelHorizontal', 'LongitudinalAccel', 'distance']
# select_plt_col = ['VelHorizontal', 'AccForward', 'LongitudinalAccel', 'distance']

start_vel = 80
end_vel = 5
brk_select_num = 4
acc_merge_size = 30

custom_ranges = [(20, 80, "80_20"),
                 (5, 80, "80_5"),
                 (50, 80, "80_50"),
                 (20, 50, "50_20"),
                 (70, 80, "80_70"),
                 (60, 70, "70_60"),
                 (50, 60, "60_50"),
                 (40, 50, "50_40"),
                 (30, 40, "40_30"),
                 (20, 30, "30_20"),
                 (10, 20, "20_10"),
                 (5, 10, "10_5")]

color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#393b79',
              '#637939', '#8c6d31', '#843c39', '#7b4173', '#1b9e77',
              '#d95f02', '#7570b3', '#e7298a', '#66a61e']

plt_label_name = ['TestSet','AMPM']

## 분석 Start
time_results = []
detail_results = []

for k in range(len(rawdf)):
    # Report 폴더 만들고 저장
    file_name = os.path.basename(fName[k])
    folder_name = os.path.splitext(file_name)[0]
    # folder_path = os.path.join(os.path.dirname(fName[k]), folder_name)
    folder_path = os.path.join(os.path.dirname(fName[k]), 'Report')
    os.makedirs(folder_path, exist_ok=True)

    # 파일명에서 데이터 끌어오기 (파일명에 따라서 변경 가능)
    parts = [part for part in folder_name.replace("-", "_").split("_")]
    # parts.append(parts[4].split('T')[1])
    # parts.append(re.search(r'W(\d+)T(\d+)', parts[4]).group(1))
    # parts.append(re.search(r'W(\d+)T(\d+)', parts[4]).group(2))

    # 파일명에 따라서 변경
    df_spec = pd.DataFrame([parts], columns=select_columns).reset_index(drop=True)

    # 제동 분석
    df = AN_ACCBRK(rawdf[k][1], rawdf[k][3], slip=any('WheelSpeed' in col for col in rawdf[k][1].columns),
                   target= [start_vel, end_vel], valid=brk_select_num, plot=False)
    # df = AN_ACCBRK(rawdf[k][1], rawdf[k][3], slip=any('WheelSpeed' in col for col in rawdf[k][1].columns),
    #                target=[80, 5], valid=4, plot=True)
    # df_sub = AN_ACCBRK(rawdf[k][1], rawdf[k][3], slip=any('WheelSpeed' in col for col in rawdf[k][1].columns),
    #                 target=[80, 20], valid=4, plot=True)

    df['LongitudinalAccel'] = SP_MovingAverage(df[["AccForward"]], ws=acc_merge_size, plot=True).values
    df["distance"] = df["VelHorizontal"] / 3.6 / rawdf[0][3]

    # BrakingDist
    # dfbrk = df.loc[~df["testNum"].isna()] # 전체 제동거리 선택
    dfbrk = df.loc[~df["effNum"].isna()]

    # Time interval
    time_diffs = dfbrk['Time'].diff().dropna().round(3)
    timeinter = time_diffs.mode().iloc[0] # 최빈값 구하기

    # 절대 시간 만들기 (Lap)
    dfbrk = dfbrk.copy()
    dfbrk['AbsLapT'] = dfbrk.groupby('testNum').cumcount() * timeinter

    # plt.figure()
    # plt.plot(dfbrk['AbsLapT'],dfbrk['VelHorizontal'],'.k')
    # plt.plot(dfbrk['AbsLapT'], dfbrk['wsFL'], '.r')
    # plt.plot(dfbrk['AbsLapT'], dfbrk['wsFR'], '.b')
    # plt.plot(dfbrk['AbsLapT'], dfbrk['wsRL'], '.y')
    # plt.plot(dfbrk['AbsLapT'], dfbrk['wsRR'], '.g')


    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2행 2열 subplot
    axes = axes.flatten()  # 2차원 배열 → 1차원 리스트로 변환

    # 각 subplot에 다른 데이터 그리기
    axes[0].plot(dfbrk['AbsLapT'], dfbrk['VelHorizontal'], '.k')
    axes[0].plot(dfbrk['AbsLapT'], dfbrk['wsFL'], '.r')
    axes[0].set_title('wsFL')

    axes[1].plot(dfbrk['AbsLapT'], dfbrk['VelHorizontal'], '.k')
    axes[1].plot(dfbrk['AbsLapT'], dfbrk['wsFR'], '.b')
    axes[1].set_title('wsFR')

    axes[2].plot(dfbrk['AbsLapT'], dfbrk['VelHorizontal'], '.k')
    axes[2].plot(dfbrk['AbsLapT'], dfbrk['wsRL'], '.y')
    axes[2].set_title('wsRL')

    axes[3].plot(dfbrk['AbsLapT'], dfbrk['VelHorizontal'], '.k')
    axes[3].plot(dfbrk['AbsLapT'], dfbrk['wsRR'], '.g')
    axes[3].set_title('wsRR')


    # 공통 축 라벨 및 여백 조정
    for ax in axes:
        ax.set_xlabel('AbsLapT')
        ax.set_ylabel('Value')
        ax.grid(True)

    fig.suptitle(folder_name+'WS', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Time Sum
    df_time_sum = dfbrk.groupby(['AbsLapT'], as_index=False)[select_plt_col].mean()
    df_time_sum['distance'] = df_time_sum['distance'].cumsum()
    time_results.append(df_time_sum)

    # Check plot
    n = len(select_plt_col)
    fig, axes = plt.subplots(n, 1, figsize=(8, 4*n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, select_plt_col):
        ax.plot(dfbrk['AbsLapT'], dfbrk[col], '.b', label=f'Raw')
        ax.plot(df_time_sum['AbsLapT'], df_time_sum[col], '.r', label=f'Merge')
        ax.set_ylabel(col)
        ax.legend()
    axes[-1].set_xlabel('Time')
    plt.suptitle(folder_name, fontsize=16)
    plt.tight_layout()

    # Longitudinal Acc
    df_acc = pd.DataFrame()
    df_dist_detail = pd.DataFrame({"testNum": dfbrk["testNum"].unique()})

    acc_values = []

    for low, high, label in custom_ranges:
        mean_acc = abs(dfbrk.loc[(dfbrk['VelHorizontal'] >= low) & (dfbrk['VelHorizontal'] < high),'LongitudinalAccel'].mean())
        acc_values.append(mean_acc)
        df_dist_temp = (dfbrk.loc[(dfbrk["VelHorizontal"] >= low) & (dfbrk["VelHorizontal"] <= high)]
                        .groupby("testNum")["distance"].sum().reset_index(name=label))
        df_dist_detail = df_dist_detail.merge(df_dist_temp, on="testNum", how="left")

    df_dist = pd.DataFrame([df_dist_detail.mean().round(3)])
    # df_dist = pd.concat([df_dist, df_spec[columns].reset_index(drop=True)], axis=1)
    df_dist = pd.concat([df_spec[select_columns].reset_index(drop=True),df_dist], axis=1)
    df_dist.insert(0, 'Output', 'Distance')

    df_acc = pd.DataFrame([acc_values], columns=[label for _, _, label in custom_ranges]).round(3)
    # df_acc = pd.concat([df_acc, df_spec[columns].reset_index(drop=True)], axis=1)
    df_acc = pd.concat([df_spec[select_columns].reset_index(drop=True),df_acc], axis=1)
    df_acc.insert(0, 'Output', 'AccX')

    common_cols = df_dist.columns.intersection(df_acc.columns)
    df_combined = pd.concat([df_dist[common_cols], df_acc[common_cols]], ignore_index=True)
    df_combined['Time_plot'] = [df_time_sum] * len(df_combined)

    detail_results.append(df_combined)

    for fig_num in plt.get_fignums():  # 현재 생성된 모든 figure 번호 가져오기
        fig = plt.figure(fig_num)  # 해당 figure 불러오기
        axes = fig.get_axes()

        # 전체 Figure 제목 가져오기
        if fig._suptitle and fig._suptitle.get_text():  # suptitle이 존재하면 사용
            fig_title = fig._suptitle.get_text()
        elif axes and axes[0].get_title():  # suptitle이 없으면 첫 번째 subplot 제목 사용
            fig_title = axes[0].get_title()
        else:
            fig_title = f"figure_{fig_num}"  # 아무 제목도 없으면 figure 번호 사용

        # 파일 이름에 사용할 수 없는 문자 제거 (예: 파일명 오류 방지)
        fig_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in fig_title)

        # 저장 경로 설정
        save_path = os.path.join(folder_path, f"{fig_title}.png")
        fig.savefig(save_path)  # 그래프 저장
        print(f"Saved: {save_path}")
        plt.close()

    print(f"Analysis End: {file_name}")

df_results = pd.concat(detail_results, ignore_index=True)

# 저장
df_data = df_results.copy()  # 전체 DataFrame
excel_path = os.path.join(folder_path, 'result.xlsx')

with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    workbook = writer.book

    grouped = df_data.groupby(['ReqNo', 'RoadInfo'])
    for (req_no, road_info), group_df in grouped:
        # 시트 이름
        sheet_name = f"{req_no}_{road_info}"[:31]

        # Time_plot 제외한 데이터 저장
        group_df.drop(columns=['Time_plot']).T.to_excel(writer, sheet_name=sheet_name, index=False)
        # group_df.drop(columns=['Time_plot']).to_excel(writer, sheet_name=sheet_name, index=False)

        # 그래프 삽입 준비
        worksheet = writer.sheets[sheet_name]
        cut_df = group_df[group_df['Output']=='Distance']

        time_results = cut_df['Time_plot'].tolist()  # 리스트 형태

        if len(time_results) == 0:
            continue

        # x, y 컬럼 설정
        x_col = time_results[0].columns[0]
        y_cols = time_results[0].columns[1:]

        for y_idx, y_col in enumerate(y_cols):
            plt.figure(figsize=(6,4))
            for df_idx, df in enumerate(time_results):
                # fName 리스트가 있다면 레이블 설정
                # label_name = f"DF{df_idx+1}"
                # label_name = cut_df['TestSet'].iloc[df_idx]+ '-' + cut_df['AMPM'].iloc[df_idx]
                label_name = '-'.join([str(cut_df[col].iloc[df_idx]) for col in plt_label_name])
                plt.plot(df[x_col], df[y_col], linestyle='-', linewidth=2,
                         color=color_list[df_idx % len(color_list)],
                         label=label_name)

            plt.xlabel('Time')
            plt.ylabel(y_col)
            plt.title(f"{y_col} Comparison")
            plt.legend()
            plt.tight_layout()

            # # 그래프를 이미지로 메모리에 저장
            # img_path = f"{sheet_name}_{y_col}.png"
            # plt.savefig(img_path)
            # plt.close()
            #
            # # 엑셀에 이미지 삽입
            # worksheet.insert_image(f'K{y_idx*20 + 2}', img_path)
            # os.remove(img_path)

            img_data = io.BytesIO()
            plt.savefig(img_data, format='png')
            plt.close()
            img_data.seek(0)

            worksheet.insert_image(f'K{y_idx * 20 + 2}', 'plot.png', {'image_data': img_data})


# # Compare 그래프
# color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#               '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#393b79',
#               '#637939', '#8c6d31', '#843c39', '#7b4173', '#1b9e77',
#               '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#a6761d']
#
# x_col = time_results[0].columns[0]
# y_cols = time_results[0].columns[1:]  # 첫 번째 컬럼 제외
#
# for y_idx, y_col in enumerate(y_cols):
#     plt.figure(figsize=(8, 5))
#
#     # time_results 안의 각 DataFrame 비교
#     for df_idx, df in enumerate(time_results):
#         label_name_tmp = os.path.basename(fName[df_idx])
#         label_name = os.path.splitext(label_name_tmp)[0]
#
#         plt.plot(df[x_col], df[y_col], linestyle='-',linewidth=2,
#                  color=color_list[df_idx % len(color_list)],
#                  label=label_name)
#
#     plt.xlabel('Time')
#     plt.ylabel(y_col)
#     plt.title(f"Comparison of {y_col}")
#     plt.legend()
#     plt.tight_layout()
#
# for fig_num in plt.get_fignums():  # 현재 생성된 모든 figure 번호 가져오기
#     fig = plt.figure(fig_num)  # 해당 figure 불러오기
#     axes = fig.get_axes()
#
#     # 전체 Figure 제목 가져오기
#     if fig._suptitle and fig._suptitle.get_text():  # suptitle이 존재하면 사용
#         fig_title = fig._suptitle.get_text()
#     elif axes and axes[0].get_title():  # suptitle이 없으면 첫 번째 subplot 제목 사용
#         fig_title = axes[0].get_title()
#     else:
#         fig_title = f"figure_{fig_num}"  # 아무 제목도 없으면 figure 번호 사용
#
#     # 파일 이름에 사용할 수 없는 문자 제거 (예: 파일명 오류 방지)
#     fig_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in fig_title)
#
#     # 저장 경로 설정
#     save_path = os.path.join(folder_path, f"{fig_title}.png")
#     fig.savefig(save_path)  # 그래프 저장
#     print(f"Saved: {save_path}")
#
# save_excel = os.path.join(folder_path, "df_result.xlsx")
# df_transposed.to_excel(save_excel, index=False)

plt.figure()