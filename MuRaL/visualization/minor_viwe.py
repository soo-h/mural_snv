import os
import matplotlib.pyplot as plt
import numpy as np
import re
import sys

from minor_read import LogParser

def plot_minor(log_path):
    log_saver = LogParser(log_path)
    generate_loss_plot(log_saver)
    generate_kmer_plot(log_saver)
    generate_regional_plot(log_saver)
    generate_time_cost_plot(log_saver)
    generate_best_model_kmer_plot(log_saver)
    generate_best_model_regional_plot(log_saver)
    generate_contributions_plot(log_saver)
    
##########
# Loss view
#######
def generate_loss_plot(log_saver, save_path=None):
    """
    生成并保存训练和验证loss随epoch变化的图像。
    
    参数:
    - log_path: 日志文件路径
    """
    log_path = log_saver.file_path
    train_loss = log_saver.metrics.loss_metrics['train_loss']
    valid_loss = log_saver.metrics.loss_metrics['valid_loss']
    valid_loss_fdiri_cal = log_saver.metrics.loss_metrics['valid_loss_fdiri_cal']

    name = extract_name_from_path(log_path)
    title = f"Training and Validation Loss Over Epochs ({name})"
    if not save_path:
        opt_name = f"loss_over_epochs_{name}.png"
        save_path = get_save_path(log_path, opt_name)
    plot_loss_over_epochs(train_loss, valid_loss, valid_loss_fdiri_cal, title, save_path=save_path)

def get_save_path(log_path, opt_name):
    opt_path = '/'.join(log_path.split('/')[:-1])
    save_path = os.path.join(opt_path, opt_name)
    return save_path

def extract_name_from_path(log_path):
    """
    从文件路径中提取文件名（不包含扩展名）。
    
    参数:
    - log_path: 文件路径
    
    返回:
    - 文件名（不包含扩展名）
    """
    name = log_path.split('/')[-1]
    try:
        name = name.split('.')[-2]
    except:
        return name
    return name

def plot_loss_over_epochs(train_loss, valid_loss, valid_loss_fdiri_cal, title='Training and Validation Loss Over Epochs', save_path=None):
    """
    绘制训练和验证loss随epoch变化的图。
    
    参数:
    - train_loss: 训练loss列表
    - valid_loss: 验证loss列表
    - valid_loss_fdiri_cal: 经过FDIRI校正的验证loss列表
    - title: 图表标题,默认为'Training and Validation Loss Over Epochs'
    - save_path: 图像保存路径,若为None,则不保存图像
    """
    epochs = range(len(train_loss))
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(epochs, train_loss, label='Train Loss', color='blue', alpha=0.7, edgecolor='k')
    ax.scatter(epochs, valid_loss, label='Validation Loss', color='green', alpha=0.7, edgecolor='k')
    ax.scatter(epochs, valid_loss_fdiri_cal, label='Validation Loss (FDIRI Cal)', color='red', alpha=0.7, edgecolor='k')

    ax.plot(epochs, train_loss, color='blue', alpha=0.7)
    ax.plot(epochs, valid_loss, color='green', alpha=0.7)
    ax.plot(epochs, valid_loss_fdiri_cal, color='red', alpha=0.7)

    ax.legend()
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.yaxis.label.set_color('grey')
    ax.xaxis.label.set_color('grey')
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

#######
# kmer view
########
def generate_kmer_plot(log_saver, save_path=None):
    """
    生成并保存训练和验证loss随epoch变化的图像。
    
    参数:
    - log_path: 日志文件路径
    """
    log_path = log_saver.file_path
    kmer_dict = log_saver.metrics.kmer_metrics

    name = extract_name_from_path(log_path)
    title = f"Kmer Metrics Over Epochs ({name})"
    if not save_path:
        opt_name = f"kmer_over_epochs_{name}.png"
        save_path = get_save_path(log_path, opt_name)

    plot_kmer_metrics(kmer_dict, title, save_path=save_path)

    return 0

def check_and_sort_kmer(data_dict):
    """
    检查数据是否按首个数字从小到大排列，如果不是，则进行排序。

    参数:
    - data_dict: 包含数据的字典，键为数据名称，值为相应的数据列表或其他类型。

    返回:
    - 排序后的数据字典。
    """
    # 提取首个数字并转换为整数，用于排序
    def extract_first_digit(key):
        return int(key.split('mer_')[0])

    # 检查数据是否按首个数字从小到大排列
    keys = list(data_dict.keys())
    keys1, keys2 = [], []
    for key in keys:
        if 'fdiri_cal' in key:
            keys2.append(key)
        else:
            keys1.append(key)
    
    sorted_keys1 = sorted(keys1, key=extract_first_digit)
    sorted_keys2 = sorted(keys2, key=extract_first_digit)
    sorted_keys = sorted_keys1 + sorted_keys2
    if keys != sorted_keys:
        # 需要进行排序
        sorted_data = {key: data_dict[key] for key in sorted_keys}
        return sorted_data
    else:
        # 数据已经按首个数字从小到大排列
        return data_dict
    
def plot_kmer_metrics(kmer_dict, title='Kmer Metrics Over Epochs', save_path=None, n_class=4):
    """
    绘制某指标随epoch变化的图像。

    参数:
    - kmer_dict: 包含kmer指标数据的字典, 键为指标名称, 值为相应的数据列表或数组。
    - n_class: 类别数, 默认为4。

    返回:
    - None
    """
    col_num = len(kmer_dict) // 2  # 每行列数
    kmer_dict = check_and_sort_kmer(kmer_dict)

    fig, ax = plt.subplots(2, col_num, figsize=(12, 8))

    for idx, key in enumerate(kmer_dict):
        i, j = idx // col_num, idx % col_num
        kmer_data = np.asarray(kmer_dict[key])
        epochs = range(len(kmer_data))

        for clss in range(n_class):
            kmer_prob = kmer_data[:, clss]

            ax[i, j].scatter(epochs, kmer_prob, label=f'Prob {clss}', alpha=0.7)
            ax[i, j].plot(epochs, kmer_prob, alpha=0.7)

        ax[i, j].set_ylim(0, 1)
        ax[i, j].set_title(f'{key}')
        ax[i, j].legend()
        ax[i, j].grid(True)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局以确保标题不重叠
    
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

###########
# regional view
##########
def generate_regional_plot(log_saver, save_path=None):
    """
    生成并保存训练和验证loss随epoch变化的图像。
    
    参数:
    - log_path: 日志文件路径
    """
    log_path = log_saver.file_path
    regional_dict = log_saver.metrics.regional_metrics

    name = extract_name_from_path(log_path)
    title = f"Regional Metrics Over Epochs ({name})"
    if not save_path:
        opt_name = f"Regional_over_epochs_{name}.png"
        save_path = get_save_path(log_path, opt_name)
    plot_regional_metrics(regional_dict, title, save_path=save_path)

def check_and_sort_regional(data_dict):
    """
    检查数据是否按首个数字从小到大排列，如果不是，则进行排序。

    参数:
    - data_dict: 包含数据的字典，键为数据名称，值为相应的数据列表或其他类型。

    返回:
    - 排序后的数据字典。
    """
    # 提取首个数字并转换为整数，用于排序
    def extract_first_digit(key):
        return int(key.split('bp_')[0])

    # 检查数据是否按首个数字从小到大排列
    keys = list(data_dict.keys())
    keys1, keys2 = [], []
    for key in keys:
        if 'fdiri_cal' in key:
            keys2.append(key)
        else:
            keys1.append(key)
    
    sorted_keys1 = sorted(keys1, key=extract_first_digit)
    sorted_keys2 = sorted(keys2, key=extract_first_digit)
    sorted_keys = sorted_keys1 + sorted_keys2
    if keys != sorted_keys:
        # 需要进行排序
        sorted_data = {key: data_dict[key] for key in sorted_keys}
        return sorted_data
    else:
        # 数据已经按首个数字从小到大排列
        return data_dict
    
def plot_regional_metrics(regional_dict, title='Kmer Metrics Over Epochs', save_path=None, n_class=4):
    """
    绘制某指标随epoch变化的图像。

    参数:
    - regional_dict: 包含regional指标数据的字典, 键为指标名称, 值为相应的数据列表或数组。
    - n_class: 类别数, 默认为4。

    返回:
    - None
    """
    col_num = len(regional_dict) // 2  # 每行列数
    regional_dict = check_and_sort_regional(regional_dict)

    fig, ax = plt.subplots(2, col_num, figsize=(12, 8))

    for idx, key in enumerate(regional_dict):
        i, j = idx // col_num, idx % col_num
        kmer_data = np.asarray(regional_dict[key])
        epochs = range(len(kmer_data))

        for clss in range(n_class):
            kmer_prob = kmer_data[:, clss]

            ax[i, j].scatter(epochs, kmer_prob, label=f'Prob {clss}', alpha=0.7)
            ax[i, j].plot(epochs, kmer_prob, alpha=0.7)

        ax[i, j].set_ylim(0, 1)
        ax[i, j].set_title(f'{key}')
        ax[i, j].legend()
        ax[i, j].grid(True)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局以确保标题不重叠
    
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    return 0

########
# time view
#########

def generate_time_cost_plot(log_saver, save_path=None):
    """
    save plot that time cost distribution. (second/epoch) 
    
    参数:
    - log_path: 日志文件路径
    """
    # time cost infor read
    log_path = log_saver.file_path
    time_used = get_time_used_mean_epoch(log_saver)
    # opt config set
    name = extract_name_from_path(log_path)
    title = f"Time Distribution for Different Processes ({name})"
    if not save_path:
        opt_name = f"time_cost_mean_epoch_{name}.png"
        save_path = get_save_path(log_path, opt_name)
    plot_cost_time_bar(time_used, title, save_path=save_path)

def get_time_used_mean_epoch(log_saver):
    
    time_used_mean_epoch = {}
    epoch_num = len(log_saver.metrics.loss_metrics['train_loss'])

    time_used_mean_epoch['preprocess_time_train'] = log_saver.time_cost.preprocess_times['train']
    time_used_mean_epoch['preprocess_time_valid'] = log_saver.time_cost.preprocess_times['valid']
    time_used_mean_epoch['get_data'] = log_saver.time_cost.get_batch_time / epoch_num
    time_used_mean_epoch['train_data'] = log_saver.time_cost.train_batch_time / epoch_num
    time_used_mean_epoch['epoch_time'] = sum(log_saver.time_cost.epoch_times) / len(log_saver.time_cost.epoch_times)
    return time_used_mean_epoch

def plot_cost_time_bar(data, title='Time Distribution for Different Processes', 
                          ylabel='Time (seconds/epoch)', save_path=None):
    """
    绘制按值排序的堆叠直方图。

    参数:
    - data: 包含数据的字典，键为类别名称，值为相应的值。
    - title: 图表标题。
    - ylabel: Y轴标签。
    - colors: 颜色列表，长度应与数据项数量匹配。
    """
    # 数据排序（从大到小）
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

    # 准备数据
    categories = list(sorted_data.keys())
    values = list(sorted_data.values())

    colors = ['#4c72b0', '#55a868', '#c44e52', '#8172b3']
    

    # 绘制堆叠直方图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, values, color=colors)

    # 设置标签和标题
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=12)

    # 设置格式
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.yaxis.label.set_color('grey')
    ax.xaxis.label.set_color('grey')
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

    # 调整布局
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    # 显示图像
    plt.show()


#############
# best model view
#########
def generate_best_model_kmer_plot(log_saver, save_path=None):
    # time cost infor read
    log_path = log_saver.file_path
    kmer_corr_best = get_best_model_kmer(log_saver)
    name = extract_name_from_path(log_path)
    title = f"Kmer Metrics best checkpoint {name}.png"
    if not save_path:
        opt_name = f"kmer_plot_best_checkpoint_{name}.png"
        save_path = get_save_path(log_path, opt_name)
    plot_best_model_in_validation_kmer(kmer_corr_best, title=title, save_path=save_path)

def get_best_model_kmer(log_saver):
    best_idx = np.argmin(log_saver.metrics.loss_metrics['valid_loss'])
    kmer_corr = log_saver.metrics.kmer_metrics
    kmer_corr_best = {key: kmer_corr[key][best_idx] for key in kmer_corr}
    return kmer_corr_best

def plot_best_model_in_validation_kmer(kmer_dict, title='Kmer Metrics best model', save_path=None, n_class=4):
    """
    Plot Kmer metrics over epochs for the best model in validation.

    Parameters:
    - kmer_dict (dict): Dictionary containing Kmer metrics data.
    - title (str): Title of the plot.
    - save_path (str): Path to save the plot.
    - n_class (int): Number of classes.

    Returns:
    None
    """
    kmer_dict = check_and_sort_kmer(kmer_dict)
    minor_num = int(len(kmer_dict) / 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(n_class)
    delta =  0.8 / minor_num
    
    for cof, key in enumerate(kmer_dict):
        kmer_data = np.asarray(kmer_dict[key])
        ax_index = 0 if 'fdiri_cal' not in key else 1
        if cof >= minor_num:
            cof -= minor_num
        
        x_ = x + cof * delta

        axes[ax_index].scatter(x_, kmer_data, label=f'{key}', alpha=0.7)

    for i, ax in enumerate(axes):
        ax.set_ylim(0, 1)
        ax.set_xticks(x+delta * int(minor_num / 2))
        ax.set_xticklabels([f'prob{i}' for i in range(n_class)])
        ax.legend(title='Kmer Types')
        ax.grid(True)
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Corr', fontsize=12)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to ensure the title does not overlap
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

############
# best model view -- regional
############
def generate_best_model_regional_plot(log_saver, save_path=None):
    # time cost infor read
    log_path = log_saver.file_path
    regional_corr_best = get_best_model_regional(log_saver)
    name = extract_name_from_path(log_path)
    title = f"Regional Metrics best model'({name})"
    if not save_path:
        opt_name = f"regional_plot_best_checkpoint_{name}.png"
        save_path = get_save_path(log_path, opt_name)
    plot_best_model_in_validation_regional(regional_corr_best, title=title, save_path=save_path)

def get_best_model_regional(log_saver):
    best_idx = np.argmin(log_saver.metrics.loss_metrics['valid_loss'])
    regional_corr = log_saver.metrics.regional_metrics
    regional_corr_best = {key: regional_corr[key][best_idx] for key in regional_corr}
    return regional_corr_best

def plot_best_model_in_validation_regional(regional_dict, title='Regional Metrics best model', save_path=None, n_class=4):
    """
    Plot Kmer metrics over epochs for the best model in validation.

    Parameters:
    - regional_dict (dict): Dictionary containing Kmer metrics data.
    - title (str): Title of the plot.
    - save_path (str): Path to save the plot.
    - n_class (int): Number of classes.

    Returns:
    None
    """
    regional_dict = check_and_sort_regional(regional_dict)
    minor_num = int(len(regional_dict) / 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(n_class)
    delta =  0.8 / minor_num
    
    for cof, key in enumerate(regional_dict):
        kmer_data = np.asarray(regional_dict[key])
        ax_index = 0 if 'fdiri_cal' not in key else 1
        if cof >= minor_num:
            cof -= minor_num
        
        x_ = x + cof * delta

        axes[ax_index].scatter(x_, kmer_data, label=f'{key}', alpha=0.7)

    for i, ax in enumerate(axes):
        ax.set_ylim(0, 1)
        ax.set_xticks(x+delta * int(minor_num / 2))
        ax.set_xticklabels([f'prob{i}' for i in range(n_class)])
        ax.legend(title='Kmer Types')
        ax.grid(True)
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Corr', fontsize=12)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to ensure the title does not overlap
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

############
# best model view -- contribution(mean) 
############
class ContributionFactory:
    @staticmethod
    def clear_contribution_dict(contri_dict):
        rescale_contri_dict = {}
        if len(contri_dict) == 2:
            for key in contri_dict:
                rescale_contri_dict[key] = contri_dict[key]
        
        elif len(contri_dict) == 6:
            for key in contri_dict:
                if '0.5' in key or '0.25' in key:
                    rescale_contri_dict[key] = contri_dict[key]
        
        else:
            sys.exit(f"Error: unconsider situation, check {contri_dict}")
        
        values = np.asarray(list(rescale_contri_dict.values()))[:, 1:]
        values = values / np.sum(values, axis=0)
        for i, key in enumerate(rescale_contri_dict.keys()):
            rescale_contri_dict[key] = values[i] 
        return rescale_contri_dict
    
class ContributionPlotter:
    
    def __init__(self, saver_list, labels):

        self.colors = plt.get_cmap('Set1').colors  # Use a consistent color palette
        self.rescale_contri_dict = ContributionFactory.clear_contribution_dict
        self.best_model_contri_list = [self._get_best_model_contri(log_saver) for log_saver in saver_list]
        self.labels = labels

    def _get_best_model_contri(self, log_saver):
        best_idx = np.argmin(log_saver.metrics.loss_metrics['valid_loss'])
        contri_dict = log_saver.contri_dict
        contri_dict_best = {key: contri_dict[key][best_idx] for key in contri_dict}

        rescale_contri_dict = self.rescale_contri_dict(contri_dict_best)
        return rescale_contri_dict

    def _extract_contribution_data(self, contribution_dict):
        parsed_data = []
        for key, values in contribution_dict.items():
            type_match = re.match(r'([a-zA-Z]+\d*)([\d.]+)', key)
            if type_match:
                contribution_type = type_match.group(1)
                if 'local' in contribution_type:
                    contribution_type = 'Local'
                elif 'distal0' in contribution_type or 'distal2' in contribution_type:
                    contribution_type = 'Distal_Large'
                elif 'distal1' in contribution_type:
                    contribution_type = 'Distal_Middle'
                weight = float(type_match.group(2))
                parsed_data.append((contribution_type, weight, values))
        return parsed_data
    
    def plot_contributions_stacked(self, title='', save_path=None):
        contributions_list = self.best_model_contri_list
        n_positions = len(next(iter(contributions_list[0].values())))  # Number of data points per dataset (e.g., 4 positions)
        bar_width = 0.15
        index = np.arange(n_positions)
        clss = ['No_mut', 'A>C', 'A>G', 'A>T']

        fig, axs = plt.subplots(n_positions, 1, figsize=(5, 8), sharey=True)
        contribution_type_plotted = set()

        for position_idx in range(n_positions):
            ax = axs[position_idx]
            for i, contributions in enumerate(contributions_list):
                parsed_data = self._extract_contribution_data(contributions)
                
                cumulative = np.zeros(n_positions)
                for j, (contribution_type, weight, values) in enumerate(parsed_data):
                    label = contribution_type if contribution_type not in contribution_type_plotted else None
                    contribution_type_plotted.add(contribution_type)
                    
                    color = self._get_color_for_type(contribution_type)

                    ax.bar(i, values[position_idx], bar_width, bottom=cumulative[position_idx],
                           label=label, color=color, alpha=0.8)

                    cumulative[position_idx] += values[position_idx]
            
            ax.set_title(f'{clss[position_idx]}', fontsize=18)
            ax.set_xticks(np.arange(len(contributions_list)))
        #ax.set_xticklabels(self.labels, rotation=15, fontsize=18)
        ax.yaxis.set_label_coords(-1, 0.5)  # Adjust y-label position for centering
            
        axs[2].set_ylabel('Contribution Value', fontsize=20)
        fig.suptitle(title, fontsize=16, fontweight='bold')

        # Legend customization for better appearance
        if not title:
            title = 'Contributions comparison'
        fig.legend(loc='upper center', fontsize=12, ncol=3)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the legend

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def _get_color_for_type(self, contribution_type):
        """Helper function to assign colors based on contribution type."""
        if contribution_type == 'Local':
            return self.colors[0]
        elif contribution_type == 'Distal_Large':
            return self.colors[1]
        elif contribution_type == 'Distal_Middle':
            return self.colors[2]

def generate_contributions_plot(log_saver, save_path=None):
    # time cost infor read
    log_path = log_saver.file_path
    name = extract_name_from_path(log_path)

    contributions_list = [log_saver]
    labels = [name]
    #title = f"Contribution comparison(mean) "
    title = " "
    plotter = ContributionPlotter(contributions_list, labels)
    if not save_path:
        opt_name = f"Contribution comparison(mean)_{name}.png"
        save_path = get_save_path(log_path, opt_name)
    plotter.plot_contributions_stacked(title=title, save_path=save_path)