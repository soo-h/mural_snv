import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations

def process_and_plot(dfs, start, end, titles=['prob1', 'prob2', 'prob3'], title=None, out_name=None):
    # 创建3个子图（1行3列）
    fig, axes = plt.subplots(3, 1, figsize=(18, 6))

    # 遍历列表中的DataFrame，每个df对应一个子图
    for idx, df in enumerate(dfs):
        # 截取数据范围
        subset = df[start:end].copy()

        # 计算相关性
        corr = subset['avg_obs'].corr(subset['avg_pred'])
        print(f'Correlation for {titles[idx]}: {corr:.4f}')
        
        # 在对应的子图上绘图
        ax = axes[idx]
        ax.fill_between(subset['window_end'] / 1_000_000, subset['avg_obs'], alpha=0.3, color='grey', label="avg_obs")
        ax.plot(subset['window_end'] / 1_000_000, subset['avg_pred'], label="avg_pred", linewidth=1.5)

        #obs_mut = subset['obs_mut'].values
        #site_count = subset['site_count'].values

        #site_count = (site_count - np.mean(site_count)) / np.std(site_count)
        #obs_mut = (obs_mut - np.mean(obs_mut)) / np.std(obs_mut)
        #ax.plot(subset['window_end'] / 1_000_000, obs_mut , label="mut_number", linewidth=1.5)
        #ax.plot(subset['window_end'] / 1_000_000, site_count , label="site_count", linewidth=1.5)
        
        # 设置子图标题
        ax.set_title(f"{titles[idx]}(corr: {corr})", fontsize=14)
        ax.legend(fontsize=9)

    axes[2].set_xlabel("Chr6 (Mb)", fontsize=10)
    axes[1].set_ylabel("Average Mutation Rate (Z-score)", fontsize=10)
    # 设置整个图的标题
    plt.suptitle(title, fontsize=22)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整图形布局，防止标题重叠
    if out_name:
        plt.savefig(f'{out_name}.png')
    plt.show()

def plot_obs_mutation(df, start, end, title=None):
    # 创建3个子图（1行3列）
    fig, axes = plt.subplots(3, 1, figsize=(18, 6))

    # 遍历列表中的DataFrame，每个df对应一个子图

    subset = df[start:end].copy()
    subset = subset[subset['used_or_deprecated'] == 'used']

    # Z-score 标准化
    subset['avg_obs'] = (subset['avg_obs'] - subset['avg_obs'].mean()) / subset['avg_obs'].std()

    # 在对应的子图上绘图
    axes[0].plot(subset['window_end'] / 1_000_000, subset['avg_obs'], label="avg_obs", linewidth=1.5)

    obs_mut = subset['obs_mut'].values
    site_count = subset['site_count'].values

    #site_count = (site_count - np.mean(site_count)) / np.std(site_count)
    #obs_mut = (obs_mut - np.mean(obs_mut)) / np.std(obs_mut)
    axes[1].plot(subset['window_end'] / 1_000_000, obs_mut , label="mut_number", linewidth=1.5)
    axes[2].plot(subset['window_end'] / 1_000_000, site_count , label="site_count", linewidth=1.5)
        
        # 设置子图标题
    for i in axes:
        i.legend(fontsize=9)

    axes[2].set_xlabel("Chr6 (Mb)", fontsize=10)
    axes[1].set_ylabel("Average Mutation Rate (Z-score)", fontsize=10)
    # 设置整个图的标题
    plt.suptitle(title, fontsize=22)

    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整图形布局，防止标题重叠
    plt.show()




def normalize(df, columns):
    """标准化指定列的函数"""
    for col in columns:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

def calc_prob_correlation(dfs, start, end):
    # 切片并标准化所有数据
    dfs = [df[start:end].copy() for df in dfs]
    
    # 过滤掉 'used_or_deprecated' 列不为 'used' 的数据
    dfs = [df[df['used_or_deprecated'] == 'used'] for df in dfs]

    # 对每个数据框进行标准化
    columns_to_normalize = ['avg_obs', 'avg_pred']
    dfs = [normalize(df, columns_to_normalize) for df in dfs]

    # 计算每对 DataFrame 之间的相关性
    result = {}
    for idx1, idx2 in combinations(range(len(dfs)), 2):
        subset1, subset2 = dfs[idx1], dfs[idx2]

        
        # 计算相关性
        corr_obs = subset1['avg_obs'].corr(subset2['avg_obs'])
        corr_pred = subset1['avg_pred'].corr(subset2['avg_pred'])
        result[(idx1, idx2)] = {'obs': corr_obs, 'pred': corr_pred}
        
        print(f'Correlation for prob{idx1+1} and prob{idx2+1} obs: {corr_obs:.4f}')
        print(f'Correlation for prob{idx1+1} and prob{idx2+1} pred: {corr_pred:.4f}')
    return result

def calculate_correlation_by_region(dfs, region_size=10_000_000, end=None):
    """
    计算多个数据集按区域分段的相关性。

    参数：
    - dfs: 包含多个数据集的列表，每个数据集为一个 Pandas DataFrame。
    - region_size: 每个区域的大小（默认为10,000,000）。
    - end: 数据的最后窗口结束点，如果为None，则自动取最后一个数据集的最后一个窗口结束点。

    返回：
    - results: 包含每个区域相关性和变异数量的结果列表。
    """
    # 确保所有输入数据的最后窗口结束点一致
    if end is None:
        end = dfs[0]['window_end'].values[-1]
    
    # 存储所有结果
    results = []
    
    # 遍历多个数据集
    for idx, df in enumerate(dfs):
        print(f"Processing prob{idx}...")
        
        # 筛选 'used' 数据
        df_used = df[df['used_or_deprecated'] == 'used']
        
        # 检查数据是否为空
        if df_used.empty:
            print(f"prob{idx} has no 'used' data. Skipping...")
            continue
        
        # 分区域计算相关性
        for start in range(1, end - region_size + 1, region_size):
            region_end = start + region_size
            
            # 筛选当前区域的数据
            region_data = df_used[(df_used['window_end'] >= start) & (df_used['window_end'] <= region_end)]
            
            # 检查区域数据是否为空
            if region_data.empty:
                print(f"prob{idx}, region ({start}, {region_end}) has no data.")
                continue
            
            # 计算 avg_obs 和 avg_pred 的相关性
            corr = region_data['avg_obs'].corr(region_data['avg_pred'])
            results.append({
                "prob": f"prob{idx}",
                "region_start": start,
                "region_end": region_end,
                "correlation": corr,
                'mutation_number': region_data['obs_mut'].sum(),
            })

            # 检查是否出现 NaN
            if np.isnan(corr):
                print(f"NaN correlation detected in prob{idx}, region ({start}, {region_end}). Data:")
                print(region_data)
                break
    
    return results

def max_min_normalize(data):
    return (data - data.min()) / (data.max() - data.min())

def plot_correlation_by_region(results_df, region_col='region_start', correlation_col='correlation', mutation_col='mutation_number', binsize=1_000_000, title='', out_name=None):
    """
    绘制不同数据集的相关性与变异数量随区域变化的折线图，支持多子图。

    参数：
    - results_df: 结果数据集，包含计算的相关性和变异数量。
    - region_col: 区域起始列名称（默认为 'region_start'）。
    - correlation_col: 相关性列名称（默认为 'correlation'）。
    - mutation_col: 变异数量列名称（默认为 'mutation_number'）。
    - binsize: 每个区域的大小（默认为 1_000_000，即每个区域 1Mb）。
    - title: 图形的标题。
    """
    # 设置绘图风格
    plt.style.use('ggplot')

    # 创建多个子图
    n_probs = len(results_df['prob'].unique())
    fig, axes = plt.subplots(1, n_probs, figsize=(6 * n_probs, 6))

    # 如果只有一个子图，则 axes 不是列表，需要转换为列表
    if n_probs == 1:
        axes = [axes]

    # 绘制不同数据集的相关性随区域变化的折线图
    for idx, prob in enumerate(results_df['prob'].unique()):
        prob_data = results_df[results_df['prob'] == prob]
        
        # 最大最小归一化
        mutation_number = max_min_normalize(prob_data[mutation_col])
        
        # 绘制到对应的子图
        axes[idx].plot(prob_data[region_col] / binsize, prob_data[correlation_col], label=f"Correlation (prob{idx+1})", marker='o')
        axes[idx].plot(prob_data[region_col] / binsize, mutation_number, label="Mutation Number (Max-Min Norm)", marker='x')

        # 设置每个子图的标题和标签
        axes[idx].set_xlabel('Region Start (Mb)')
        axes[idx].set_ylabel('Correlation')
        axes[idx].set_title(f'prob{idx+1}')
        axes[idx].legend()

    # 调整布局
    plt.suptitle(title, fontsize=22)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # 留出上方空间给总标题
    
    if out_name:
        plt.savefig(f'{out_name}.png')

    # 显示图形
    plt.show()