from unoUtil.readDataFile import *
from unoUtil.unoPostProcessing import *
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
from scipy import stats
import seaborn as sns
from Func_stats import *
from scipy.stats import mannwhitneyu
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.metrics import r2_score

def ReadDataPickle(FileName=None, **kwargs) :
    """
    :param FileName:
    :param kwargs:
    :return:
    """

    from PyQt5 import QtWidgets
    import pickle
    Filter = kwargs.get('filter', 'Datafile(*.pkl);;Datafile(*.*)')
    dfList = []

    # QApplication이 없으면 새로 생성
    if QtWidgets.QApplication.instance() is None :
        app = QtWidgets.QApplication([])

    if not FileName :
        fName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select File', "", Filter)
        for file_path in fName:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                dfList.append(data)
        FileName = fName if fName else None
    else :
        if isinstance(FileName, str) :
            FileName = [FileName]
        FileName = [name.replace("\\", "/") for name in FileName]

        if not FileName[0].split(r'/\\')[-1].endswith('.') :
            fName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select File', FileName[0], Filter)
            for file_path in fName:
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    dfList.append(data)
            FileName = fName if fName else None
        else :
            for file_path in fName:
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    dfList.append(data)

    return dfList, FileName


def ReadDataWeather(FileName=None, **kwargs) :
    """
    :param FileName:
    :param kwargs:
    :return:
    """

    from PyQt5 import QtWidgets
    import pickle
    import pandas as pd
    Filter = kwargs.get('filter', 'Datafile(*.txt);;Datafile(*.*)')
    dfList = []

    # QApplication이 없으면 새로 생성
    if QtWidgets.QApplication.instance() is None :
        app = QtWidgets.QApplication([])

    if not FileName :
        fName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select File', "", Filter)
        for file_path in fName:
            with open(file_path, 'rb') as file:
                # data = pickle.load(file)
                data = pd.read_csv(file,sep='\t')
                dfList.append(data)
        FileName = fName if fName else None
    else :
        if isinstance(FileName, str) :
            FileName = [FileName]
        FileName = [name.replace("\\", "/") for name in FileName]

        if not FileName[0].split(r'/\\')[-1].endswith('.') :
            fName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select File', FileName[0], Filter)
            for file_path in fName:
                with open(file_path, 'rb') as file:
                    # data = pickle.load(file)
                    data = pd.read_csv(file, sep='\t')
                    dfList.append(data)
            FileName = fName if fName else None
        else :
            for file_path in fName:
                with open(file_path, 'rb') as file:
                    # data = pickle.load(file)
                    data = pd.read_csv(file, sep='\t')
                    dfList.append(data)

    return dfList, FileName

def ReadDataPickleWeahter(FileName=None, **kwargs) :
    """
    :param FileName:
    :param kwargs:
    :return:
    """

    from PyQt5 import QtWidgets
    import pickle
    import pandas as pd
    import os

    Filter = kwargs.get('filter', 'Datafile(*.txt);;Datafile(*.pkl);;Datafile(*.*)')
    dfList = []

    def read_file_by_extension(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, 'rb') as file:
            if ext in ['.pkl', '.pickle']:
                data = pickle.load(file)
            elif ext in ['.txt', '.tsv']:
                data = pd.read_csv(file, sep='\t')
            elif ext == '.csv':
                data = pd.read_csv(file)
            else:
                print(f"지원되지 않는 파일 형식: {ext}")
                data = None
        return data

    # QApplication이 없으면 새로 생성
    if QtWidgets.QApplication.instance() is None :
        app = QtWidgets.QApplication([])

    if not FileName :
        fName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select File', "", Filter)
        for file_path in fName:
            data = read_file_by_extension(file_path)
            if data is not None:
                dfList.append(data)
        FileName = fName if fName else None
    else :
        if isinstance(FileName, str) :
            FileName = [FileName]
        FileName = [name.replace("\\", "/") for name in FileName]

        if not FileName[0].split(r'/\\')[-1].endswith('.') :
            fName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select File', FileName[0], Filter)
            for file_path in fName:
                data = read_file_by_extension(file_path)
                if data is not None:
                    dfList.append(data)
            FileName = fName if fName else None
        else :
            for file_path in fName:
                data = read_file_by_extension(file_path)
                if data is not None:
                    dfList.append(data)
    return dfList, FileName

def split_time_columns(df):
    # time_split = df['Time'].str.split(':', expand=True)
    time_split = df['Time'].astype(str).str.split(r'[:.]', expand=True)
    df['hour'] = time_split[0].astype(int)
    df['min'] = time_split[1].astype(int)
    df['sec'] = time_split[2].astype(int)
    return df

def ReadData(FileName=None, **kwargs) :
    """
    :param FileName:
    :param kwargs:
    :return:
    """

    from PyQt5 import QtWidgets
    import pickle
    import pandas as pd
    import os

    Filter = kwargs.get('filter', 'Datafile(*.*);;Datafile(*.txt);;Datafile(*.pkl)')
    dfList = []

    def read_file_by_extension(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, 'rb') as file:
            if ext in ['.pkl', '.pickle']:
                data = pickle.load(file)
            elif ext in ['.txt', '.tsv']:
                data = pd.read_csv(file, sep='\t')

                if len(data.columns) < 5:
                    data = pd.read_csv(file_path, sep='|')
                elif data.columns[0].startswith('Unnamed'):
                    data = data.drop(index=0).reset_index(drop=True)
                    data.columns = data.iloc[0]
                    data = data.drop(index=0).reset_index(drop=True)

            elif ext == '.csv':
                data = pd.read_csv(file)
            elif ext == '.mat':
                file_select_name = os.path.basename(file.name)
                parts = re.split(r"[-_]", file_select_name)
                # parts = file_select_name.split("-")
                if parts[0]=='hyc':
                    data = read_hylat_mat(file)
                elif parts[0]=='hys':
                    data = read_hylong_mat(file)
                    read_hylong_mat
                else:
                    print(f"Test Mode 추가 필요: {ext}")
            else:
                print(f"지원되지 않는 파일 형식: {ext}")
                data = None
        return data

    # QApplication이 없으면 새로 생성
    if QtWidgets.QApplication.instance() is None :
        app = QtWidgets.QApplication([])

    if not FileName :
        fName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select File', "", Filter)
        for file_path in fName:
            data = read_file_by_extension(file_path)
            if data is not None:
                dfList.append(data)
        FileName = fName if fName else None
    else :
        if isinstance(FileName, str) :
            FileName = [FileName]
        FileName = [name.replace("\\", "/") for name in FileName]

        if not FileName[0].split(r'/\\')[-1].endswith('.') :
            fName, _ = QtWidgets.QFileDialog.getOpenFileNames(None, 'Select File', FileName[0], Filter)
            for file_path in fName:
                data = read_file_by_extension(file_path)
                if data is not None:
                    dfList.append(data)
            FileName = fName if fName else None
        else :
            for file_path in fName:
                data = read_file_by_extension(file_path)
                if data is not None:
                    dfList.append(data)
    return dfList, FileName

# def plt_group(df,groupby_cols,group,xcol,ycol):
#     spec_list = sorted(df[group].unique())
#     n = len(spec_list)
#
#     ncols = 3
#     nrows = (n + ncols - 1) // ncols
#
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
#     axes = axes.flatten()
#
#     for idx, spec_val in enumerate(spec_list):
#         ax = axes[idx]
#
#         df_spec = df[df[group] == spec_val]
#
#         for group_keys, grp in df_spec.groupby(groupby_cols):
#             if not isinstance(group_keys, tuple):
#                 group_keys = (group_keys,)
#             label = " - ".join(str(g) for g in group_keys)
#
#             ax.plot(grp[xcol], grp[ycol], marker='o', linestyle='', label=label)
#
#         ax.set_title(f"{spec_val}")
#         ax.set_xlabel(xcol)
#         ax.set_ylabel(ycol)
#         ax.grid(True, linestyle="--", alpha=0.4)
#         ax.legend(loc="best", fontsize=7)
#
#     for j in range(len(spec_list), len(axes)):
#         fig.delaxes(axes[j])
#     plt.tight_layout()

def plt_change(df,div_group,group,**kwargs):

    xcol = kwargs.get('x', np.nan)
    ycol = kwargs.get('y', np.nan)
    plot = kwargs.get('plot', False)

    spec_list = sorted(df[div_group].unique())
    groups = dict(tuple(df.groupby(group)))  # {road: df_part}

    palette = plt.cm.get_cmap("tab20")  # 최대 20색
    color_cycle = itertools.cycle(palette.colors)
    marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", "*"])
    color_map = {spec: next(color_cycle) for spec in spec_list}
    marker_map = {spec: next(marker_cycle) for spec in spec_list}

    df_total = []
    for grp, df_grp in groups.items():
        valid_spec = df_grp.groupby(div_group).filter(lambda g: len(g) == 2)
        if valid_spec.empty:
            continue

        diff_rows = []
        for spec_val, df_spec in valid_spec.groupby(div_group):
            df_spec = df_spec.sort_values(by='Datetime')
            df_numeric = df_spec.select_dtypes(include='number')
            diff = (df_numeric.iloc[1] - df_numeric.iloc[0]).to_frame().T
            diff[div_group] = spec_val
            diff[group] = grp
            diff_rows.append(diff)
        df_diff = pd.concat(diff_rows, ignore_index=True)
        df_total.append(df_diff)

        if plot:
            fig, ax = plt.subplots(figsize=(7, 5))
            # 실제 산점도
            for spec_val, select_grp in df_diff.groupby([div_group]):
                ax.plot(select_grp[xcol], select_grp[ycol],
                        marker=marker_map[spec_val[0]], linestyle="", markersize=8,
                        color=color_map[spec_val[0]], label=spec_val[0])

            ax.set_title(f"group = {grp}")
            ax.set_xlim(-10, 10)
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(title=div_group, fontsize=8, loc="best")
            plt.tight_layout()

    df_result = pd.concat(df_total, ignore_index=True)

    return df_result

def plt_group(df,groupby_cols,group,xcol,ycol,**kwargs):

    zcol = kwargs.get('z', None)

    spec_list = sorted(df[group].unique())
    n = len(spec_list)

    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    color_cycle = plt.cm.tab10.colors

    for idx, spec_val in enumerate(spec_list):
        ax = axes[idx]

        df_spec = df[df[group] == spec_val]

        for i, (group_keys, grp) in enumerate(df_spec.groupby(groupby_cols)):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            label = " - ".join(str(g) for g in group_keys)

            color = color_cycle[i % len(color_cycle)]

            ax.plot(grp[xcol], grp[ycol], marker='o', linestyle='', label=label, color=color)

            if zcol is not None and zcol in grp.columns:
                ax.plot(grp[xcol], grp[zcol], marker='*', linestyle='', label=label, color=color)

        ax.set_title(f"{spec_val}")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc='center left',bbox_to_anchor=(1.05, 0.5),fontsize=8)

    for j in range(len(spec_list), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()

def plt_group_multi(df,groupby_cols,xcol,ycol,**kwargs):

    zcol = kwargs.get('z', None)
    textcol = kwargs.get('text', None)

    for select_grp in groupby_cols:
        subplot_list = sorted(df[select_grp].unique())
        n = len(subplot_list)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows),constrained_layout=True)
        axes = axes.flatten()
        color_cycle = plt.cm.tab10.colors

        for idx, subplot_dt in enumerate(subplot_list):
            ax = axes[idx]
            df_spec = df[df[select_grp] == subplot_dt]

            for i, (group_keys, grp) in enumerate(df_spec.groupby(groupby_cols)):
                if not isinstance(group_keys, tuple):
                    group_keys = (group_keys,)
                label = " - ".join(str(g) for g in group_keys)
                color = color_cycle[i % len(color_cycle)]
                ax.plot(grp[xcol], grp[ycol], marker='o', linestyle='', label=label, color=color)

                if zcol is not None and zcol in grp.columns:
                    ax.plot(grp[xcol], grp[zcol], marker='*', linestyle='', label=label, color=color)
                elif textcol is not None and textcol in grp.columns:
                    for x, y, textname in zip(grp[xcol], grp[ycol], grp[textcol]):
                        if isinstance(textname, (int, float)):
                            text_display = f"{round(textname, 1)}"
                        else:
                            text_display = str(textname)
                        ax.text(x, y, text_display, va='bottom', ha='center', fontsize=8, clip_on=False)

            ax.set_title(f"{subplot_dt}")
            ax.set_xlabel(xcol)
            ax.set_xlim([df[xcol].min()-df[xcol].std(),df[xcol].max()+df[xcol].std()])
            ax.set_ylabel(ycol)
            ax.set_ylim([df[ycol].min()-df[ycol].std(), df[ycol].max()+df[ycol].std()])
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc='center left',bbox_to_anchor=(1.05, 0.5),fontsize=8)

        for j in range(len(subplot_list), len(axes)):
            fig.delaxes(axes[j])

def plt_change_v2(df,div_group,group,**kwargs):

    xcol = kwargs.get('x', np.nan)
    ycol = kwargs.get('y', np.nan)
    numdist = kwargs.get('num', 3)
    plot = kwargs.get('plot', False)

    spec_list = sorted(df[div_group].unique())
    groups = dict(tuple(df.groupby(group)))  # {road: df_part}

    palette = plt.cm.get_cmap("tab20")  # 최대 20색
    color_cycle = itertools.cycle(palette.colors)
    marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", "*"])
    color_map = {spec: next(color_cycle) for spec in spec_list}
    marker_map = {spec: next(marker_cycle) for spec in spec_list}

    df_total = []
    df_total_per = []
    for grp, df_grp in groups.items():
        valid_spec = df_grp.groupby(div_group).filter(lambda g: len(g) == 2)
        if valid_spec.empty:
            continue

        diff_rows = []
        diff_rows_per = []
        for spec_val, df_spec in valid_spec.groupby(div_group):

            df_spec = df_spec.sort_values(by='Datetime')
            df_numeric = df_spec.select_dtypes(include='number')

            cols = [f"Dist_{i}" for i in range(numdist)]
            df_stat = df_numeric[cols]
            # f_stat, p_value = stats.f_oneway(*df_stat.values)
            group1 = df_stat.iloc[:, 0].dropna()
            group2 = df_stat.iloc[:, 1].dropna()
            f_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

            n1 = n2 = 4
            mean1 = df_numeric.iloc[1]['Dist_mean']
            std1 = df_numeric.iloc[1]['Dist_std']
            mean2 = df_numeric.iloc[0]['Dist_mean']
            std2 = df_numeric.iloc[0]['Dist_std']
            co_result = cohens_d_compare(mean1, std1, n1, mean2, std2, n2)

            diff = (df_numeric.iloc[1] - df_numeric.iloc[0]).to_frame().T
            diff[div_group] = spec_val
            diff[group] = grp
            diff['cohens_d'] = co_result
            diff['fstat'] = f_stat
            diff['pvalue'] = p_value
            diff_rows.append(diff)

            diff_per = (((df_numeric.iloc[1]/df_numeric.iloc[0])-1)*100).to_frame().T
            diff_per[div_group] = spec_val
            diff_per[group] = grp
            diff_per['cohens_d'] = co_result
            diff_per['fstat'] = f_stat
            diff_per['pvalue'] = p_value

            diff_rows_per.append(diff_per)
        df_diff = pd.concat(diff_rows, ignore_index=True)
        df_diff_per = pd.concat(diff_rows_per, ignore_index=True)

        df_total.append(df_diff)
        df_total_per.append(df_diff_per)

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            # 실제 산점도
            plot_info = [(df_diff, axes[0], "[Value]"),(df_diff_per, axes[1], "[Percent]")]

            for df, ax, title_prefix in plot_info:
                for spec_val, select_grp in df.groupby([div_group]):
                    ax.plot(select_grp[xcol], select_grp[ycol],
                            marker=marker_map[spec_val[0]], linestyle="", markersize=8,
                            color=color_map[spec_val[0]], label=spec_val[0])

                ax.set_title(f"{title_prefix} group = {grp}")
                ax.set_xlabel(xcol)
                ax.set_ylabel(ycol)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend(title=div_group, fontsize=8, loc="best")

            plt.tight_layout()

    df_result = pd.concat(df_total, ignore_index=True)
    df_result_per = pd.concat(df_total_per, ignore_index=True)

    return df_result, df_result_per

# def plt_bar(df,div_group,group,**kwargs):
#
#     xcol = kwargs.get('x', np.nan)
#     name = kwargs.get('name', np.nan)
#
#     spec_list = sorted(df[div_group].unique())
#     groups = dict(tuple(df.groupby(group)))  # {road: df_part}
#
#     palette = plt.cm.get_cmap("Set1")  # 최대 20색
#     color_cycle = itertools.cycle(palette.colors)
#     marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", "*"])
#     color_map = {spec: next(color_cycle) for spec in spec_list}
#     marker_map = {spec: next(marker_cycle) for spec in spec_list}
#     padding = 0.02 # 막대 사이 여유 간격
#
#     df_total = []
#     for grp, df_grp in groups.items():
#
#         fig, ax = plt.subplots(figsize=(8, 4))
#         y_pos = np.arange(len(df_grp[div_group].unique()))
#         # diff_rows = []
#         for i, (spec_val, df_spec) in enumerate(df_grp.groupby(div_group)):
#             # df_tmp = df_grp[df_grp[div_group]=='SRTT']
#             df_spec = df_spec.sort_values(by='Datetime')
#             # df_numeric = df_spec.select_dtypes(include='number')
#             df_select = df_spec[xcol]
#             # tdist = stats_tdist(df_spec_data.loc[0], confidence=95, chname='tdist')
#             df_tdist = pd.concat([stats_tdist(row, confidence=95, chname='tdist').assign(index=idx).set_index('index')
#                                   for idx, row in df_select.iterrows()])
#             # df_tmp = pd.concat([df_select, df_tdist], axis=1)
#             # df_merged = pd.concat([df_spec, df_spec_data], axis=1)
#             df_plot = pd.concat([df_spec, df_tdist], axis=1)
#
#             n_bars = len(df_plot)
#             offset = (1.0 - padding * (n_bars - 1)) / n_bars # 전체를 1 안에 균등 분배 + padding
#
#             for j, (_,row) in enumerate(df_plot.iterrows()):
#                 y_centered = y_pos[i] - 0.5 + (offset + padding) * j + offset / 2
#
#                 color = palette(j % palette.N)
#                 ax.barh(y=y_centered,
#                         width=row['tdist_upper'] - row['tdist_lower'],
#                         left=row['tdist_lower'],
#                         height=offset,
#                         color=color,
#                         alpha=0.6)
#
#                 ax.plot(row['tdist_mean'], y_centered,
#                         marker=marker_map[spec_val[0:]],
#                         color=color_map[spec_val[0:]],
#                         markersize=4)
#
#                 ax.text(row['tdist_upper'] + 0.02, y_centered,
#                         f"{row[div_group]}/{row[name]}",
#                         va='center', fontsize=8)
#
#             ax.set_yticks(y_pos)
#             # ax.set_yticklabels(df_plot['GroupSpec'])
#             ax.set_xlabel("Braking Distance")
#             ax.set_title(grp)
#             ax.grid(True, linestyle="--", alpha=0.4)
#
#     return

def plt_stats_bar(df,div_group,group,**kwargs):

    xcol = kwargs.get('x', np.nan)
    st = kwargs.get('st', np.nan)
    num = kwargs.get('num', np.nan)
    numunit = kwargs.get('numunit','°C')

    spec_list = sorted(df[div_group].unique())
    grouped = df.groupby(group) # {road: df_part}

    palette = matplotlib.colormaps["Set1"]  # 최대 20색
    color_cycle = itertools.cycle(palette.colors)
    marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", "*"])

    color_map = {spec: next(color_cycle) for spec in spec_list}
    marker_map = {spec: next(marker_cycle) for spec in spec_list}
    padding = 0.02 # 막대 사이 여유 간격

    for grp_name, df_grp in grouped:
        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = np.arange(0, len(df_grp[div_group].unique())*2, 2)
        y_labels = [row[div_group] for _, row in df_grp.iterrows()]

        for i, (spec_val, df_spec) in enumerate(df_grp.groupby(div_group)):
            df_spec = df_spec.sort_values(by='Datetime')
            df_select = df_spec[xcol]

            df_tdist = pd.concat([stats_tdist(row, confidence=95, chname='tdist').assign(index=idx).
                                 set_index('index') for idx, row in df_select.iterrows()])

            df_plot = pd.concat([df_spec, df_tdist], axis=1)
            n_bars = len(df_plot)
            offset = (1.0 - padding * (n_bars - 1)) / n_bars

            for j, (_, row) in enumerate(df_plot.iterrows()):
                y_centered = y_pos[i] - 0.5 + (offset + padding) * j + offset / 2
                color = palette(j % palette.N)

                # ax.barh(y=y_centered,
                #         width=row['tdist_upper'] - row['tdist_lower'],
                #         left=row['tdist_lower'],height=offset,
                #         color=color,edgecolor=color_map[spec_val],linewidth=1.5,alpha=0.5)
                #
                # ax.plot(row['tdist_mean'],y_centered,
                #         marker=marker_map[spec_val],
                #         color=color_map[spec_val],
                #         markersize=4)
                #
                # ax.text(row['tdist_upper'] + 0.02,
                #         y_centered,
                #         f"{row[div_group]}/{row[name]}",va='center',fontsize=8)

                ax.barh(y=y_centered,
                        width=row['tdist_upper'] - row['tdist_lower'],
                        left=row['tdist_lower'],height=offset,
                        color=color_map[spec_val],alpha=0.5)

                ax.plot(row['tdist_mean'],y_centered,
                        marker=marker_map[spec_val],
                        color=color,
                        markersize=4)

                text_parts = [str(row[div_group])]
                if pd.isna(st) == False:
                    text_parts.append(str(row[st]))
                if pd.isna(num) == False:
                    text_parts.append(f"{row[num]:.1f} {numunit}")
                text_parts.append(f"{round(row['tdist_mean'], 1)} m")

                ax.text(row['tdist_upper'] + 0.02,y_centered,"-".join(text_parts),
                        va='center', fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_xlabel("Braking Distance")
        ax.set_title(grp_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return

def plt_group_slipstats(df,groupby_cols,xname,**kwargs):

    for select_grp in groupby_cols:
        subplot_list = sorted(df[select_grp].unique())
        n = len(subplot_list)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows),constrained_layout=True)
        axes = axes.flatten()
        color_cycle = plt.cm.tab10.colors

        for idx, subplot_dt in enumerate(subplot_list):
            ax = axes[idx]
            df_spec = df[df[select_grp] == subplot_dt]

            for i, (group_keys, grp) in enumerate(df_spec.groupby(groupby_cols)):
                if not isinstance(group_keys, tuple):
                    group_keys = (group_keys,)
                label = " - ".join(str(g) for g in group_keys)
                color = color_cycle[i % len(color_cycle)]
                selectdata = grp[xname]

                x, y = slip_stats_distribution(selectdata)
                # x, y = slip_stats_distribution(selectdata, plot =True)
                ax.plot(x, y, linestyle='-', label=label, color=color)
                # textname = '-'.join(str(grp[col].iloc[0]) for col in groupby_cols)

                text_display = f"{round(x[np.argmax(y)], 1)}"
                ax.text(x[np.argmax(y)], y.max(), text_display, va='bottom', ha='center', fontsize=12, clip_on=False)
                #
                # if textcol is not None and textcol in grp.columns:
                #     for x, y, textname in zip(grp[xcol], grp[ycol], grp[textcol]):
                #         if isinstance(textname, (int, float)):
                #             text_display = f"{round(textname, 1)}"
                #         else:
                #             text_display = str(textname)
                #         ax.text(x, y, text_display, va='bottom', ha='center', fontsize=8, clip_on=False)

            ax.set_title(f"{subplot_dt}")
            ax.set_xlabel(xname)
            ax.set_xlim(0, 10)
            # ax.set_xlim([df[xcol].min()-df[xcol].std(),df[xcol].max()+df[xcol].std()])
            # ax.set_ylabel(ycol)
            # ax.set_ylim([df[ycol].min()-df[xcol].std(), df[ycol].max()+df[xcol].std()])
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc='center left',bbox_to_anchor=(1.05, 0.5),fontsize=8)

        for j in range(len(subplot_list), len(axes)):
            fig.delaxes(axes[j])

def plt_group_slipstats_single(df,groupby_cols,xname,**kwargs):

    for select_grp in groupby_cols:
        subplot_list = sorted(df[select_grp].unique())
        for idx, subplot_dt in enumerate(subplot_list):
            df_spec = df[df[select_grp] == subplot_dt]

            for i, (group_keys, grp) in enumerate(df_spec.groupby(groupby_cols)):
                if not isinstance(group_keys, tuple):
                    group_keys = (group_keys,)
                # label = " - ".join(str(g) for g in group_keys)
                label = " - ".join(str(g) for g in group_keys) + f" - {subplot_dt}"
                selectdata = grp[xname]

                plt.figure()
                x, y = slip_stats_distribution(selectdata, plot =True, plotname = label)


def plt_stats_bar_v2(df,div,subtitle,bar,**kwargs):

    # xcol = kwargs.get('x', np.nan)
    st = kwargs.get('st', np.nan)
    num = kwargs.get('num', np.nan)
    numunit = kwargs.get('numunit','°C')
    """
    div_group = y축의 기준 축
    subtitle_group = subplot 기준 그래프
    bar_group = bar 그래프 비교 그래프
    """
    div_group = div
    subtitle_group = subtitle
    bar_group = bar

    spec_list = sorted(df[div_group].unique())
    grouped = df.groupby(subtitle_group) # {road: df_part}

    palette = matplotlib.colormaps["Set1"]  # 최대 20색
    color_cycle = itertools.cycle(palette.colors)
    marker_cycle = itertools.cycle(["o", "s", "^", "D", "v", "P", "X", "*"])

    color_map = {spec: next(color_cycle) for spec in spec_list}
    marker_map = {spec: next(marker_cycle) for spec in spec_list}
    padding = 0.02 # 막대 사이 여유 간격

    for grp_name, df_grp in grouped:
        fig, ax = plt.subplots(figsize=(8, 4))
        y_pos = np.arange(0, len(df_grp[div_group].unique())*2, 2)
        y_labels = [row[div_group] for _, row in df_grp.iterrows()]

        for i, (spec_val, df_spec) in enumerate(df_grp.groupby(div_group)):
            df_spec = df_spec.sort_values(by='Datetime')
            df_plot = df_spec.drop_duplicates(subset=bar_group, keep='first')

            n_bars = len(df_plot)
            offset = (1.0 - padding * (n_bars - 1)) / n_bars

            for j, (_, row) in enumerate(df_plot.iterrows()):
                y_centered = y_pos[i] - 0.5 + (offset + padding) * j + offset / 2
                color = palette(j % palette.N)

                upper_col = [col for col in df_plot.columns if 'supper' in str(col)][0]
                lower_col = [col for col in df_plot.columns if 'slower' in str(col)][0]
                mean_col = [col for col in df_plot.columns if 'smean' in str(col)][0]

                ax.barh(y=y_centered,
                        width=row[upper_col] - row[lower_col],
                        left=row[lower_col],height=offset,
                        color=color_map[spec_val],alpha=0.5)

                ax.plot(row[mean_col],y_centered,
                        marker=marker_map[spec_val],
                        color=color,
                        markersize=4)

                text_parts = [str(row[div_group])]
                if pd.isna(st) == False:
                    text_parts.append(str(row[st]))
                if pd.isna(num) == False:
                    text_parts.append(f"{row[num]:.1f} {numunit}")
                text_parts.append(f"{round(row[mean_col], 1)} m")

                ax.text(row[upper_col] + 0.02,y_centered,"-".join(text_parts),
                        va='center', fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_xlabel("Braking Distance")
        ax.set_title(grp_name)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    return
def read_hylat_mat(path):
    from scipy.io import loadmat
    from scipy.integrate import simpson
    def near_index(array, value):  # 목표 수치에 가장 가까운 값의 Index 반환
        abs_array = np.abs(array)
        diff = np.abs(abs_array - value)
        index = np.argmin(diff)
        return index

    mat = loadmat(path)
    header = mat['__header__']
    header_str = header.decode()
    date = header_str.split("Created on: ")[-1]
    data = mat['fitResult']

    MaxG = data[3][0][0][0][0][0][0]
    MaxSpeed = data[3][0][0][0][1][0][0]
    HalfSpeed = data[3][0][0][0][2][0][0]
    Area = data[3][0][0][0][3][0][0]
    Area_to_max = data[3][0][0][0][4][0][0]

    mat_dot = data[0]
    mat_ideal = data[1]
    mat_fit = data[2]

    dot_df = np.vstack(mat_dot).astype(np.float64)
    dot_df = pd.DataFrame(dot_df.T)
    dot_df.columns = ['Vel', 'Latacc']

    y_pred = mat_fit[1].astype(float)
    vel_created = mat_fit[0].astype(float)
    # Max lateral acceleration
    latacc_max_index = np.argmax(y_pred)
    latacc_max = np.max(y_pred)
    latacc_40 = latacc_max * 0.4
    latacc_40_idx = near_index(y_pred, latacc_40)
    # Max speed at max lateral acceleration
    vel_max = vel_created[latacc_max_index][0]
    # Speed at half of max lateral acceleration
    vel_half_index = np.argmin(np.abs(y_pred - latacc_max / 2))
    vel_half = vel_created[vel_half_index][0]
    # Area between 40% of MaxG
    area = simpson(y=y_pred[:latacc_40_idx].flatten(), x=vel_created[:latacc_40_idx].flatten())

    vel_latacc_df = pd.DataFrame({
        "Velocity": vel_created.flatten(),
        'LatAcc': y_pred.flatten()
    })

    result = {
        "MaxG": latacc_max,
        "V1": vel_max,
        "V2": vel_half,
        "Area": area,
        "Vel_Latacc_df": vel_latacc_df,
        "Dot_df": dot_df
    }

    return result

def read_hylong_mat(path):
    from scipy.io import loadmat
    def near_index(array, value):  # 목표 수치에 가장 가까운 값의 Index 반환
        abs_array = np.abs(array)
        diff = np.abs(abs_array - value)
        index = np.argmin(diff)
        return index

    mat = loadmat(path)
    header = mat['__header__']
    header_str = header.decode()
    date = header_str.split("Created on: ")[-1]
    data = mat['dataList']
    test_item = data[0][0][0][0][0]

    vel_15 = data[7][0][0][0]
    dist_15 = data[8][0][0][0]

    vel = data[9][0][:, 0]
    slip = data[9][0][:, 1]

    idx_5 = near_index(slip, 5)
    idx_10 = near_index(slip, 10)
    idx_15 = near_index(slip, 15)
    idx_20 = near_index(slip, 20)

    vel_slip_df = pd.DataFrame({
        'VelHorizontal': vel.flatten(),
        'distSum': dist_15,
        'SlipRatio': slip.flatten()
    })

    idx_4 = near_index(vel_slip_df['SlipRatio'], 3.5)

    vel_slip_df = (vel_slip_df.iloc[idx_4:])
    vel_slip_df.reset_index(drop=True, inplace=True)

    result = {
        'Velocity15%': vel_15,
        'Distance15%': dist_15,
        'Slip5%': vel[idx_5],
        'Slip10%': vel[idx_10],
        'Slip15%': vel[idx_15],
        'Slip20%': vel[idx_20],
        "Vel_Slip_df": vel_slip_df
    }

    return result

def fit_linear_models(df,x_range,x_col,y_col,group_name,**kwargs):

    plot = kwargs.get('plot', False)

    df_result = pd.DataFrame()
    df_result[f"{x_col}_fit"] = x_range

    dict_summary = {}
    for col in group_name:
        dict_summary[col] = df[col].iloc[0]

    group_str = '_'.join(str(df[col].iloc[0]) for col in group_name)

    raw_df = df[[x_col, y_col]].dropna()
    if raw_df.empty:
        return pd.DataFrame()

    X_train = raw_df[[x_col]]
    y_train = raw_df[y_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_pred = pd.DataFrame({x_col: x_range})
    y_pred = model.predict(X_pred)
    df_result[f"{y_col}_fit"] = y_pred

    slope = float(model.coef_[0])
    r2 = model.score(X_train, y_train)
    n = len(raw_df)
    # x_per_10_y = 10 / slope if slope != 0 else float('inf')
    y_per_10_x = slope * 10

    dict_summary[f"{x_col}_R2"] = r2
    dict_summary[f"{x_col}_Slope"] = round(slope, 2)
    dict_summary[f"{x_col}_Delta"] = round(y_per_10_x, 2)

    for target_x in [10, 20, 30, 40, 50]:
        if target_x in df_result[f"{x_col}_fit"].values:
            target_y = df_result.loc[df_result[f"{x_col}_fit"] == target_x, f"{y_col}_fit"].values[0]
            dict_summary[f"{x_col}_{target_x}d"] = round(target_y, 2)
        else:
            dict_summary[f"{x_col}_{target_x}d"] = None

    df_summary = pd.DataFrame([dict_summary])

    # 시각화
    if plot:
        color = '#1f77b4'
        plt.figure(figsize=(8, 6))
        plt.plot(raw_df[x_col], raw_df[y_col], 'o', color=color, label=y_col)
        label_name = f"{round(y_per_10_x, 2)}/10°C"
        plt.plot(df_result[f"{x_col}_fit"], df_result[f"{y_col}_fit"], '-', color=color, label=label_name)

        plt.title(group_str)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xlim([0, 60])
        plt.grid()
        plt.legend()
        y_min = round(raw_df[y_col].min() - 2, 0)
        y_max = round(raw_df[y_col].max() + 2, 0)
        plt.ylim([y_min, y_max])

    return df_summary

def fit_linear_models_srtt(df,x_range,x_col,y_col,group_name,**kwargs):

    plot = kwargs.get('plot', False)

    df_result = pd.DataFrame()
    df_result[f"{x_col}_fit"] = x_range

    dict_summary = {}
    for col in group_name:
        dict_summary[col] = df[col].iloc[0]

    group_str = '_'.join(str(df[col].iloc[0]) for col in group_name)

    raw_df = df[[x_col, y_col]].dropna()
    if raw_df.empty:
        return pd.DataFrame()

    X_train = raw_df[[x_col]]
    y_train = raw_df[y_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_pred = pd.DataFrame({x_col: x_range})
    y_pred = model.predict(X_pred)
    df_result[f"{y_col}_fit"] = y_pred

    slope = float(model.coef_[0])
    r2 = model.score(X_train, y_train)
    n = len(raw_df)
    # x_per_10_y = 10 / slope if slope != 0 else float('inf')
    y_per_10_x = slope * 0.1

    dict_summary[f"{x_col}_R2"] = r2
    dict_summary[f"{x_col}_Slope"] = round(slope, 2)
    dict_summary[f"{x_col}_Delta"] = round(y_per_10_x, 2)

    for target_x in [0.6, 0.65, 0.75, 0.80, 0.85]:
        # x_fit 컬럼
        x_values = df_result[f"{x_col}_fit"].values
        # 가장 가까운 값의 index 찾기
        idx_nearest = (abs(x_values - target_x)).argmin()

        # 해당 y 값 가져오기
        target_y = df_result.iloc[idx_nearest][f"{y_col}_fit"]
        dict_summary[f"{x_col}_{target_x}d"] = round(target_y, 2)

    df_summary = pd.DataFrame([dict_summary])

    # 시각화
    if plot:
        color = '#1f77b4'
        plt.figure(figsize=(8, 6))
        plt.plot(raw_df[x_col], raw_df[y_col], 'o', color=color, label=y_col)
        label_name = f"{round(y_per_10_x, 2)}/10°C"
        plt.plot(df_result[f"{x_col}_fit"], df_result[f"{y_col}_fit"], '-', color=color, label=label_name)

        plt.title(group_str)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.xlim([0.5, 0.9])
        plt.grid()
        plt.legend()
        y_min = round(raw_df[y_col].min() - 2, 0)
        y_max = round(raw_df[y_col].max() + 2, 0)
        plt.ylim([y_min, y_max])

    return df_summary

def plt_group_multi_linear(df,groupby_cols,xcol,ycol,**kwargs):

    zcol = kwargs.get('z', None)
    textcol = kwargs.get('text', None)

    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#393b79',
                  '#637939', '#8c6d31', '#843c39', '#7b4173', '#1b9e77',
                  '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#a6761d']

    for select_grp in groupby_cols:
        subplot_list = sorted(df[select_grp].unique())
        n = len(subplot_list)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows),constrained_layout=True)
        axes = axes.flatten()
        # color_cycle = plt.cm.tab10.colors
        color_cycle = color_list

        for idx, subplot_dt in enumerate(subplot_list):
            ax = axes[idx]
            df_spec = df[df[select_grp] == subplot_dt]

            for i, (group_keys, grp) in enumerate(df_spec.groupby(groupby_cols)):
                if not isinstance(group_keys, tuple):
                    group_keys = (group_keys,)
                label = " - ".join(str(g) for g in group_keys)
                color = color_cycle[i % len(color_cycle)]
                ax.plot(grp[xcol], grp[ycol], marker='o', linestyle='', label=label, color=color)

                ## 추세선
                x_train = grp[[xcol]]
                y_train = grp[[ycol]]

                model = LinearRegression()
                model.fit(x_train, y_train)
                slope = round(float(model.coef_[0]), 3)

                # x_pred = np.linspace(df[xcol].min(), df[xcol].max(), 100).reshape(-1, 1)
                x_pred = np.linspace(df[xcol].min(), df[xcol].max(), 100)
                x_pred = pd.DataFrame({xcol: x_pred})
                y_pred = model.predict(x_pred)

                ax.plot(x_pred, y_pred, linestyle='-', label=slope, color=color)

                if zcol is not None and zcol in grp.columns:
                    ax.plot(grp[xcol], grp[zcol], marker='*', linestyle='', label=label, color=color)
                elif textcol is not None and textcol in grp.columns:
                    for x, y, textname in zip(grp[xcol], grp[ycol], grp[textcol]):
                        if isinstance(textname, (int, float)):
                            text_display = f"{round(textname, 2)}"
                        else:
                            text_display = str(textname)
                        ax.text(x, y, text_display, va='bottom', ha='center', fontsize=8, clip_on=False)

            ax.set_title(f"{subplot_dt}")
            ax.set_xlabel(xcol)
            ax.set_xlim([df[xcol].min()-df[xcol].std(),df[xcol].max()+df[xcol].std()])
            ax.set_ylabel(ycol)
            ax.set_ylim([df[ycol].min()-df[ycol].std(), df[ycol].max()+df[ycol].std()])
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc='center left',bbox_to_anchor=(1.05, 0.5),fontsize=8)

        for j in range(len(subplot_list), len(axes)):
            fig.delaxes(axes[j])

def single_bar(df,x_col,y_col,gr_col,**kwargs):

    plot = kwargs.get('plot', False)
    color_list = kwargs.get('color', False)
    pltname = kwargs.get('pltname', 'plot')

    # unique_categories = df[x_col].unique()
    grouped = df.groupby(gr_col)

    n_groups = len(grouped)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups,4))

    if n_groups == 1:
        axes = [axes]

    if plot:
        y_min = df[y_col].min()
        y_max = df[y_col].max()

        for ax, (group_name, group_df) in zip(axes, grouped):
            sns.barplot(x=x_col,y=y_col,hue=x_col,data=group_df,ax=ax,palette=color_list,legend=False)
            ax.set_title(f'Group: {group_name}')
            ax.set_ylabel('Value')
            ax.set_ylim(y_min*0.9, y_max * 1.1)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.suptitle(pltname)
        plt.tight_layout()

def group_bar(df,x_cols,y_col,gr_col,str_col,str_unit,**kwargs):

    plot = kwargs.get('plot', True)
    color_list = kwargs.get('color', False)
    grpplot = kwargs.get('grpplot', False)

    # sr = 0.2
    sr = 0.1
    lr = 0.5
    compd_index = list(df[y_col].unique())
    compd_to_color = {compd: color_list[i % len(color_list)] for i, compd in enumerate(compd_index)}
    grouped = df.groupby(gr_col)

    if plot:
        n = len(grouped)
        fig, axes = plt.subplots(n, 1, figsize=(10, 4 * n), sharex=True)

        if n == 1:
            axes = [axes]

        for ax, (roadinfo, group) in zip(axes, grouped):
            for idx, row in group.iterrows():
                compd = row[y_col]
                min_val = row[x_cols].min()
                max_val = row[x_cols].max()
                mid_val = (max_val+min_val)/2
                perindex = round(row[str_col]/mid_val*100,1)
                width = max_val - min_val
                ax.barh(row[y_col], width, left=min_val, color=compd_to_color[compd], height=0.6)
                ax.text(min_val - (width*sr), row[y_col], f"{min_val:.1f}",
                        va='center', ha='right', fontsize=8, color='black')
                ax.text(max_val + (width*sr), row[y_col], f"{max_val:.1f}",
                        va='center', ha='left', fontsize=8, color='black')
                # ax.text(mid_val, row[y_col], f"{row[str_col]:.1f}{str_unit}({perindex}%)",
                #         va='center', ha='center', fontsize=9, color='black')
                ax.text(mid_val, row[y_col], f"{row[str_col]:.1f}{str_unit}",
                        va='center', ha='center', fontsize=9, color='black')
            ax.set_title(f"{gr_col}: {roadinfo}/{group[str_col].mean():.1f}")
            # ax.set_xlabel('Value')
            ax.set_ylabel(y_col)
            ax.grid(axis='x', linestyle='--', alpha=0.5)
        # 전체 범위 지정 (원한다면)
        df_min = df.apply(lambda df: df[x_cols].min(), axis=1)
        df_max = df.apply(lambda df: df[x_cols].max(), axis=1)
        plt.xlim(df_min.min()-(width*lr), df_max.max()+(width*lr))
        fig.suptitle(str_col, fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

    if grpplot:
        for i, (roadinfo, group) in enumerate(grouped):
            fig, ax = plt.subplots(figsize=(10, 4))

            for idx, row in group.iterrows():
                compd = row[y_col]
                min_val = row[x_cols].min()
                max_val = row[x_cols].max()
                mid_val = (max_val + min_val) / 2
                perindex = round(row[str_col] / mid_val * 100, 1)
                width = max_val - min_val
                ax.barh(row[y_col], width, left=min_val, color=compd_to_color[compd], height=0.6)
                ax.text(min_val - (width*sr), row[y_col], f"{min_val:.1f}",
                        va='center', ha='right', fontsize=8, color='black')
                ax.text(max_val + (width*sr), row[y_col], f"{max_val:.1f}",
                        va='center', ha='left', fontsize=8, color='black')
                # ax.text(mid_val, row[y_col],f"{row[str_col]:.2f}({perindex}%){str_unit}",
                #         va='center', ha='center', fontsize=10, color='white')
                ax.text(mid_val, row[y_col], f"{row[str_col]:.1f}{str_unit}",
                        va='center', ha='center', fontsize=9, color='black')
            ax.set_title(f"{gr_col}: {roadinfo}/{group[str_col].mean():.1f}")
            ax.set_ylabel(y_col)
            ax.grid(axis='x', linestyle='--', alpha=0.5)
            ax.set_xlim(df[x_cols[0]].min()-(width*lr), df[x_cols[1]].max() + (width*lr))
            fig.suptitle(str_col, fontsize=10)
            plt.tight_layout(rect=[0, 0, 1, 0.96])


def plt_3D_Colorbar(df,chx,chy,chz,pltname):
    # 절대값 데이터 준비
    x = abs(df[chx].values)
    y = abs(df[chy].values)
    z = abs(df[chz].values)

    # R² 계산을 위한 선형 회귀
    model = LinearRegression()
    x_reshape = x.reshape(-1, 1)  # sklearn은 2D 입력 필요
    model.fit(x_reshape, y)
    y_pred = model.predict(x_reshape)
    r2 = r2_score(y, y_pred)

    # 산점도 + 컬러바
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(x, y, c=z, cmap='coolwarm', s=100, edgecolors='none')
    plt.colorbar(sc, label=chz)

    # 회귀선 추가
    x_line = np.linspace(min(x), max(x), 100)
    y_line = model.predict(x_line.reshape(-1, 1))
    plt.plot(x_line, y_line, 'k--', label=f'Linear Fit (R² = {r2:.3f})')

    # Plot 설정
    plt.title(pltname)
    plt.xlabel(f'{chx} (abs)')
    plt.ylabel(f'{chy} (abs)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def data_correction(df,group_cols,xcol,ycol,std,**kwargs):

    plot = kwargs.get('plot', False)

    mask_all = (df[xcol] >= std[0]) & (df[xcol] <= std[1])
    aver = df.loc[mask_all,ycol].mean()

    df_group = df.groupby(group_cols)

    def correction_offset(group,gx,gy,gstd,gaver):
        mask = (group[gx] >= gstd[0]) & (group[gx] <= gstd[1])
        value = group.loc[mask,gy].mean()
        offset = value - gaver
        group[gy+'_offset'] = group[gy] - offset
        # plt.figure()
        # plt.plot(group['WHC'],group['Dist_mean'],'.r')
        # plt.plot(group['WHC'], group['y_col_offset'], '.b')
        return group

    df_final = df_group.apply(correction_offset,xcol,ycol,std,aver).reset_index(drop=True)

    if plot:
        plt.figure()
        plt.plot(df_final[xcol], df_final[ycol], '.r',label='Raw')
        plt.plot(df_final[xcol], df_final[ycol+'_offset'], '.b',label='Offset')
        plt.grid(True)
        plt.legend()
        plt.xlabel(xcol)
        plt.ylabel(ycol)

    return df_final

def plt_group_normal(df,groupby_cols,group,xcol,xrange,ycol,**kwargs):

    zcol = kwargs.get('z', None)

    spec_list = sorted(df[group].unique())
    n = len(spec_list)

    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    color_cycle = plt.cm.tab10.colors

    total_group_list = []
    # perf_index_list = []

    for idx, spec_val in enumerate(spec_list):
        ax = axes[idx]

        df_spec = df[df[group] == spec_val]
        df_group_list = []

        for i, (group_keys, grp) in enumerate(df_spec.groupby(groupby_cols)):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            label = " - ".join(str(g) for g in group_keys)

            color = color_cycle[i % len(color_cycle)]

            mask = (grp[xcol] >= xrange[0]) & (grp[xcol] <= xrange[1])
            subset = grp.loc[mask]

            if not subset.empty:
                perf_index = subset[ycol].mean()
            else:
                target = (xrange[0] + xrange[1]) / 2
                closest_idx = (grp[xcol] - target).abs().idxmin()
                perf_index = grp.loc[closest_idx, ycol]

            grp = grp.copy()
            grp[ycol + "_norm"] = grp[ycol] / perf_index * 100

            # ax.plot(grp[xcol], grp[ycol], marker='o', linestyle='', label=label, color=color)
            ax.plot(grp[xcol], grp[ycol + "_norm"], marker='o', linestyle='',label=label, color=color)

            if zcol is not None and zcol in grp.columns:
                ax.plot(grp[xcol], grp[ycol + "_norm"], marker='*', linestyle='', label=label, color=color)

            df_group_list.append(grp)

            # perf_index_list.append({
            #     "spec_val": spec_val,
            #     "label": label,
            #     "perf_index": perf_index})

        ax.set_title(f"{spec_val}")
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc='center left',bbox_to_anchor=(1.05, 0.5),fontsize=8)

        df_group_data = pd.concat(df_group_list, ignore_index=True)
        total_group_list.append(df_group_data)

    df_result = pd.concat(total_group_list, ignore_index=True)
    # df_performance = pd.DataFrame(perf_index_list)

    for j in range(len(spec_list), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()

    return df_result

def select_group_data(df,groupby_cols,group,xcol,xrange):

    spec_list = sorted(df[group].unique())

    # total_group_list = []
    perf_index_list = []

    for idx, spec_val in enumerate(spec_list):

        df_spec = df[df[group] == spec_val]
        df_group_list = []

        for i, (group_keys, grp) in enumerate(df_spec.groupby(groupby_cols)):
            if not isinstance(group_keys, tuple):
                group_keys = (group_keys,)
            label = " - ".join(str(g) for g in group_keys)

            mask = (grp[xcol] >= xrange[0]) & (grp[xcol] <= xrange[1])
            num_data = grp.select_dtypes(include=["number"])
            subset = num_data.loc[mask]

            if not subset.empty:
                perf_index = subset.mean()
            else:
                target = (xrange[0] + xrange[1]) / 2
                closest_idx = (num_data[xcol] - target).abs().idxmin()
                perf_index = grp.loc[closest_idx]

            # grp = grp.copy()
            # grp[ycol + "_norm"] = grp[ycol] / perf_index * 100
            #
            # df_group_list.append(grp)

            perf_index_dict = perf_index.to_dict()
            perf_index_dict.update({
                "spec_val": spec_val,
                "label": label
            })
            perf_index_list.append(perf_index_dict)
        # df_group_data = pd.concat(df_group_list, ignore_index=True)
        # total_group_list.append(df_group_data)

    # df_result = pd.concat(total_group_list, ignore_index=True)
    df_performance = pd.DataFrame(perf_index_list)
    df_performance.columns = df_performance.columns.map(str)
    df_performance = df_performance.sort_index(axis=1)

    return df_performance
