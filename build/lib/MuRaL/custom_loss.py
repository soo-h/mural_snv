import sys
import numpy as np

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

def combined_avg_mut_mse_loss(criterion, y_pred, y_true):
    loss1 = criterion(y_pred, y_true)
    loss_mse = cal_avg_mut_loss(y_pred, y_true)
    return loss_mse +  loss1, loss_mse


class LossTracker:
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
    def add_loss(self, mean_loss_batch):
        """
        添加损失并更新批次计数和当前损失。
        
        :param loss: 当前批次的损失值
        """
        if self.epoch_finish:
            print('Error: epoch end, plese re-init LossTracker')
            sys.exit()
        self.cumulat_loss += mean_loss_batch
        self.batch_count += 1
        # 检查是否达到批次阈值
        if self.batch_count == self.n:
            self.stored_losses.append(self.cumulat_loss)
            self.cumulat_loss = 0.0
            self.batch_count = 0
    
    def epoch_end(self):
        """
        在每个 epoch 结束时重置计数和损失。
        """
        self.epoch_finish = True
        if self.batch_count > 0:
            self.stored_losses.append(self.cumulat_loss)
        self.current_loss = 0.0
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
    
    def report_mean_losses(self):
        if not self.epoch_finish:
            self.epoch_end

        if self.stored_losses:
            return np.mean(self.stored_losses)