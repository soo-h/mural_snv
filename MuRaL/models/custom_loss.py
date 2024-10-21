import sys
import numpy as np
from typing import List

import torch
import torch.nn as nn

def cal_mse_loss(y_pred, y_true, idx=0):
    mse_loss = nn.MSELoss()
    y_true_tensor = torch.tensor(y_true == idx, dtype=torch.float32, device=y_pred.device)
    loss_mse = mse_loss(y_pred[:,0], y_true_tensor)
    return loss_mse

#def mse_loss(y_pred, y_true):
 #   return torch.pow(torch.sum(y_true == 0) - torch.sum(y_pred, axis=1)[0], 2)

def combined_mse_loss(criterion, y_pred, y_true, alpha_mse=0.5):
    loss1 = criterion(y_pred, y_true)
    loss_mse = alpha_mse * cal_mse_loss(y_pred, y_true)
    return loss_mse + (1 - alpha_mse) * loss1, loss_mse

def combined_mse_loss_10mult(criterion, y_pred, y_true, alpha_mse=10):
    loss1 = criterion(y_pred, y_true)
    loss_mse = alpha_mse * cal_mse_loss(y_pred, y_true)
    return alpha_mse * loss_mse + loss1, loss_mse

def combined_mse_loss_100mult(criterion, y_pred, y_true, alpha_mse=100):
    loss1 = criterion(y_pred, y_true)
    loss_mse = alpha_mse * cal_mse_loss(y_pred, y_true)
    return alpha_mse * loss_mse + loss1, loss_mse

def combined_prob1_mse_loss_100mult(criterion, y_pred, y_true, alpha_mse=100):
    loss1 = criterion(y_pred, y_true)
    loss_mse = alpha_mse * cal_mse_loss(y_pred, y_true, 1)
    return alpha_mse * loss_mse + loss1, loss_mse


def combined_mse_loss_200mult(criterion, y_pred, y_true, alpha_mse=200):
    loss1 = criterion(y_pred, y_true)
    loss_mse = alpha_mse * cal_mse_loss(y_pred, y_true)
    return alpha_mse * loss_mse + loss1, loss_mse

def cal_avg_mut_loss(y_pred, y_true):
    y_true_tensor = torch.tensor(y_true == 0, dtype=torch.float32, device=y_pred.device)
    
    y_true_mut = torch.mean(y_true_tensor)
    y_pred_mut = torch.mean(y_pred, axis=1)[0]
    loss_mse = torch.pow(y_true_mut - y_pred_mut, 2)
    return loss_mse

def combined_avg_mut_mse_loss1(criterion, y_pred, y_true):
    loss1 = criterion(y_pred, y_true)
    loss_mse = cal_avg_mut_loss(y_pred, y_true)
    return loss_mse +  loss1, loss_mse

def var_loss(label, prediction):
    # 计算误差
    error = label - prediction
    # 计算误差的方差
    var_error = torch.var(error, unbiased=True)
    return var_error

def combined_var_loss(y_pred, y_true, criterion):
    loss_main = criterion(y_pred, y_true)
    y_true_tensor = torch.tensor(y_true == 0).float().to(y_pred.device)    
    loss_auxiliary = var_loss(y_true_tensor, y_pred[:,0])
    alpha = dynamic_weight_adjustment(loss_main, loss_auxiliary)

    combined_loss = loss_main + alpha * loss_auxiliary
    return combined_loss, loss_main


def calc_moments(tensor):
    # 计算均值
    mean = torch.mean(tensor, dim=0)
    
    # 计算方差
    variance = torch.var(tensor, dim=0, unbiased=False)
    
    moments = torch.stack([mean, variance], dim=0)
    
    return moments

def calc_moments2(tensor):
    # 计算均值
    mean = torch.mean(tensor, dim=0)
    
    # 计算方差
    variance = torch.var(tensor, dim=0, unbiased=False) + 1e-6
    
    # 导致Nan的生成
    # 计算偏度
    skewness = torch.mean(((tensor - mean) ** 3) / (variance ** 1.5), dim=0)
    
    # 计算峰度
    kurtosis = torch.mean(((tensor - mean) ** 4) / (variance ** 2), dim=0) - 3

    # 构成向量
    moments = torch.stack([mean, variance, skewness, kurtosis], dim=0)
    
    return moments


def cal_nomut_monents_loss(y_pred, y_true):
    y_true_tensor = torch.tensor(y_true == 0).float().to(y_pred.device)    
    y_true_moments = calc_moments(y_true_tensor)
    y_pred_moments = calc_moments(y_pred[:,0])
    return y_pred_moments, y_true_moments

def cal_nclass_moments_loss(y_pred, y_true, moment=2, nclass=4):
    y_true_moments_list = []
    y_pred_moments_list = []

    if moment == 2:
        calc_moment = calc_moments
    elif moment == 4:
        calc_moment = calc_moments2
    else:
        sys.exit("Error: moment only 2 or 4")
    
    for i in range(nclass):
        # Create one-hot encoded tensors for the true labels
        y_true_tensor = (y_true == i).float().to(y_pred.device)    
        y_true_moments = calc_moment(y_true_tensor)
        
        # Extract the corresponding class predictions
        y_pred_class = y_pred[:, i]
        y_pred_moments = calc_moment(y_pred_class)

        y_true_moments_list.append(y_true_moments)
        y_pred_moments_list.append(y_pred_moments)
    
    # Stack all the moments for each class
    y_true_moments = torch.stack(y_true_moments_list, dim=0)
    y_pred_moments = torch.stack(y_pred_moments_list, dim=0)
    
    return y_pred_moments, y_true_moments


def dynamic_weight_adjustment(loss1, loss2, scale_factor=0.5):

    alpha = loss1 / (loss2 + 1e-8)  # 避免除以零
    return alpha * scale_factor

def combined_nomut_moments_loss(y_pred, y_true, criterion1, criterion2):
    loss_main = criterion1(y_pred, y_true)

    y_pred_moments, y_true_moments = cal_nomut_monents_loss(y_pred, y_true)
    loss_auxiliary = criterion2(y_pred_moments, y_true_moments)
    alpha = dynamic_weight_adjustment(loss_main, loss_auxiliary)

    combined_loss = loss_main + alpha * loss_auxiliary

    return combined_loss, loss_main

def combined_moments_loss(y_pred, y_true, criterion1, criterion2, 
                          cal_loss_func='nclass', 
                          moment=2,
                          nclass=4):
    loss_main = criterion1(y_pred, y_true)

    if cal_loss_func == 'nclass':
        cal_loss = cal_nclass_moments_loss
    else:
        sys.exit("Error, only nclass")

    y_pred_moments, y_true_moments = cal_loss(y_pred, y_true,moment=moment, nclass=nclass)
    loss_auxiliary = criterion2(y_pred_moments, y_true_moments)
    alpha = dynamic_weight_adjustment(loss_main, loss_auxiliary)

    combined_loss = loss_main + alpha * loss_auxiliary

    return combined_loss, loss_main

def combined_main_auxiliary(loss_main, loss_auxiliary, alpha):
    return loss_main + alpha * loss_auxiliary



"""
class auxiliary_Loss:
    def __init__(self,
                 loss_functions: List[str], 
                 pred, label, 
                 n_class: int) -> None:

        self.loss_init(loss_functions)

        self.n_class = n_class
        self.pred = pred
        self.label = label
        self.device = label.device

        self.pred_muts, self.obs_muts = self.init()

    
    def loss_init(self, loss_functions):
        for loss_function in loss_functions:
            if loss_function == 'MSE':
                self.mse_loss = nn.MSELoss()

    def __call__(self, pred, label):
        

    def init2(self):
        pred_muts = torch.mean(self.pred, axis=1)
        obs_muts = []
        for i in range(self.n_class):
            y_true_mut = torch.mean((self.label==i).float())
            obs_muts.append(y_true_mut)
        obs_muts = torch.tensor(obs_muts, device=self.device)

        return pred_muts, obs_muts

    def avg_muts_mes(self):
        return nn.MSELoss(self.pred_muts, self.obs_muts)

"""

class LossTracker:
    def __init__(self):
        self.cumulat_loss = 0.0  # cumulative loss

    def add_loss(self, batch_loss):
        self.cumulat_loss += batch_loss
    
    def reset(self):
        self.cumulat_loss = 0.0
    
    def report_total_losses(self):
        return self.cumulat_loss
    
    def has_loss(self):
        return self.cumulat_loss != 0.0

class LossTracker2:
    def __init__(self, n):
        """
        初始化 LossTracker 类。
        
        :param n: 批次数的阈值。当达到这个值时，存储损失并重置计数。
        """
        self.n = n  # 批次阈值
        self.batch_count = 0  # 当前批次计数
        self.cumulat_loss = 0.0  # cumulative loss
        self.stored_losses = []  # 存储的损失列表
        self.epoch_finish = False
    def add_loss(self, batch_loss):
        """
        添加损失并更新批次计数和当前损失。
        
        :param loss: 当前批次的损失值
        """
        if self.epoch_finish:
            print('Error: epoch end, plese re-init LossTracker')
            sys.exit()
        self.cumulat_loss += batch_loss
        self.batch_count += 1
        # 检查是否达到批次阈值
        if self.batch_count == self.n:
            self.stored_losses.append(self.cumulat_loss)
            self.batch_count = 0
    
    def epoch_end(self):
        """
        在每个 epoch 结束时重置计数和损失。
        """
        self.epoch_finish = True
        if self.batch_count > 0:
            self.stored_losses.append(self.cumulat_loss)
        self.batch_count = 0
    
    def reset(self):
        """
        重置所有记录的值。
        """
        self.batch_count = 0
        self.cumulat_loss = 0.0
        self.stored_losses = []
        self.epoch_finish = False
    
    def report_stored_losses(self):
        """
        获取存储的损失列表。
        
        :return: 存储的损失列表
        """
        return self.stored_losses
    
    def report_total_losses(self):
        total_losses = self.cumulat_loss
        if not self.epoch_finish:
            self.epoch_end()

        self.reset()
        return total_losses