import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys


def weights_init(m):
    """Initialize network layers"""
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight)
        
        if m.bias is not None:
            #nn.init.normal_(m.bias)
            nn.init.constant_(m.bias, 0)
        
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
            
        if m.bias is not None:
            #nn.init.normal_(m.bias)
            nn.init.constant_(m.bias, 0)
        
    elif classname.find('LSTM') != -1 or classname.find('GRU') != -1:
        for layer_p in m._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.xavier_uniform_(m.__getattr__(p))

import numpy as np
import matplotlib.pyplot as plt
from typing import List

def validate_inputs(outputs: List[np.ndarray], weights: List[float]) -> None:
    """
    Validate the validity of the model outputs and weights.

    Parameters:
        outputs (List[np.ndarray]): List of model outputs, each element is a 2D array with shape (batch, 4).
        weights (List[float]): List of corresponding model weights.

    Returns:
        None

    Raises:
        ValueError: If the inputs are invalid.
    """
    if len(outputs) != len(weights):
        raise ValueError("The length of outputs and weights lists must be the same.")
    
    if not all(output.shape == outputs[0].shape for output in outputs):
        raise ValueError("All model outputs must have the same shape.")

def compute_fused_output(outputs: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """
    Compute the weighted fused output.

    Parameters:
        outputs (List[np.ndarray]): List of model outputs.
        weights (List[float]): List of corresponding model weights.

    Returns:
        np.ndarray: The weighted fused output with shape (batch, 4).
    """
    fused_output = sum(weight * output for weight, output in zip(weights, outputs))
    return fused_output

def compute_contributions(outputs: List[np.ndarray], weights: List[float], fused_output: np.ndarray) -> List[np.ndarray]:
    """
    Compute the contribution ratio of each model in the final fused output.

    Parameters:
        outputs (List[np.ndarray]): List of model outputs.
        weights (List[float]): List of corresponding model weights.
        fused_output (np.ndarray): The weighted fused output.

    Returns:
        List[np.ndarray]: Contribution ratios of each model, each element has shape (batch, 4).
    """
    contributions = [(weight * output) / fused_output for weight, output in zip(weights, outputs)]
    return contributions

class ContributionsTracker:
    def __init__(self, model_names):
        self.model_names = model_names
        self.init(model_names)

    def init(self, model_names):
        self.model_contributions = {
            name: None for name in model_names
        }
        self.n = 0
    
    def convert_to_numpy(self, contribution):
        if contribution.is_cuda:
            contribution = contribution.cpu().numpy()
        else:
            contribution = contribution.numpy()
        return contribution
    
    def batch_number_record(self, num=None):
        if num is None:
            self.n += 1

    def save(self, name, contribution):
        contribution = self.convert_to_numpy(contribution)
        if self.model_contributions[name] is None:
            self.model_contributions[name] = np.zeros_like(contribution)
        self.model_contributions[name] += contribution 
    
    def reset(self):
        self.init(self.model_names)

    def report_mean_contribution(self, weights, print=print):
        if self.n == 0:
            print("No contributions to report.")
            return 

        for i, name in enumerate(self.model_names):
            avg_contribution = self.model_contributions[name] / self.n
            print(f"{name} contribution in validation(weights: {weights[i]}):", avg_contribution)

def compute_model_contributions(model_contributions, outputs: List[np.ndarray], weights: List[float], fused_output=None, model_names=None) -> None:
    """
    compute the contribution ratios of models in weighted fusion.

    Parameters:
        outputs (List[np.ndarray]): List of model outputs, each element is a 2D array with shape (batch, 4).
        weights (List[float]): List of corresponding model weights.

    Returns:
        None
    """
    validate_inputs(outputs, weights)
    ## change 2024.9.30##
    if fused_output==None:
        fused_output = compute_fused_output(outputs, weights)

    fused_output = compute_fused_output(outputs, weights) # 分母改为各部分softmax的加和
    ## 
    contributions = compute_contributions(outputs, weights, fused_output)

    assert model_names == model_contributions.model_names

    model_contributions.batch_number_record()

    # Plot the contribution ratio for each model
    for i, contribution in enumerate(contributions):
        avg_contribution = contribution.mean(axis=0)  # Compute the average contribution over the batch
        #print(f"{model_names[i]} contribution in validation(weights: {weights[i]}):", avg_contribution)
        model_contributions.save(model_names[i], avg_contribution)
    return model_contributions


def model_predict_m(model, dataloader, criterion, device, n_class, distal=True, print=print):
    """Do model prediction using dataloader"""
    import time
    model.to(device)
    model.eval()
    
    pred_y = torch.empty(0, n_class).to(device)        
    total_loss = 0
    local_loss = 0
    distal_loss = 0
    distal_loss2 = 0
    batch_count = 0
    step_time = time.time()
    with torch.no_grad():
        for y, cont_x, cat_x, distal_x in dataloader:
            batch_count += 1
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)
        
            if distal:
                preds = model.forward((cont_x, cat_x), distal_x)
            else:
                preds = model.forward(cont_x, cat_x)
            if isinstance(preds, tuple):
                if len(preds) == 3:
                    preds_local, preds_distal, preds = preds
                    local_loss += criterion(preds_local, y.long().squeeze(1)).item()
                    distal_loss += criterion(preds_distal, y.long().squeeze(1)).item()
                    outputs = [preds_local, preds_distal]
                    weights = [0.5, 0.5]
                    model_names = ['local', 'distal']  
                elif len(preds) >= 4:
                    if len(preds) == 5:
                        preds, _ = preds[:-1], preds[-1]
                    preds_local, preds_distal, preds_distal2, preds = preds
                    local_loss += criterion(preds_local, y.long().squeeze(1)).item()
                    distal_loss += criterion(preds_distal, y.long().squeeze(1)).item()
                    distal_loss2 += criterion(preds_distal2, y.long().squeeze(1)).item()
                    outputs = [preds_local, preds_distal, preds_distal2]
                    weights = [0.5, 0.25,0.25]
                    weights2 = [1/3, 1/3, 1/3]
                    model_names = ['local', 'distal1', 'distal2']
                    contri_tracker1 = ContributionsTracker(model_names)
                    contri_tracker1 = compute_model_contributions(contri_tracker1, outputs, weights2, preds, model_names)

                contri_tracker = ContributionsTracker(model_names)
                contri_tracker = compute_model_contributions(contri_tracker, outputs, weights, preds, model_names)

            pred_y = torch.cat((pred_y, preds), dim=0)
                
            loss = criterion(preds, y.long().squeeze(1))
            total_loss += loss.item()
            
            if device == torch.device('cpu'):
                if  np.random.uniform(0,1) < 0.0001:
                    #print('in the model_predict_m:', device)
                    sys.stdout.flush()
            
    # time view
    print(f"Batch Number: {batch_count}; prediction Time of {batch_count} batch: {(time.time()-step_time) / 60} min")
    if not local_loss:
        return pred_y, total_loss
    
    contri_tracker.report_mean_contribution(weights, print=print)
    if len(weights) == 3:
        contri_tracker1.report_mean_contribution(weights2, print=print)

    sys.stdout.flush()
    if distal_loss2:
        return pred_y, total_loss, local_loss, distal_loss, distal_loss2
    if local_loss:
        return pred_y, total_loss, local_loss, distal_loss
    
    sys.exit("Error: return not run! check if status")



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss









def run_time_view_model_predict_m(model, dataloader, criterion, device, n_class, distal=True):
    """Do model prediction using dataloader"""
    import time
    model.to(device)
    model.eval()
    
    pred_y = torch.empty(0, n_class).to(device)        
    local_loss = 0
    distal_loss = 0
    distal_loss2 = 0
    total_loss = 0
    batch_count = 0
    get_batch_time_recode = 0
    get_batch_predict_recode = 0
    step_time = time.time()
    get_batch_time = time.time()

    with torch.no_grad():
        for y, cont_x, cat_x, distal_x in dataloader:
            get_batch_time_recode += time.time() - get_batch_time
            batch_count += 1
            
            if batch_count % 500 == 0:
                print("get 500 batch used time: ", get_batch_time_recode)
                get_batch_time_recode = 0
            
            batch_predict_time = time.time()
            cat_x = cat_x.to(device)
            cont_x = cont_x.to(device)
            distal_x = distal_x.to(device)
            y  = y.to(device)
        
            if distal:
                preds = model.forward((cont_x, cat_x), distal_x)
            else:
                preds = model.forward(cont_x, cat_x)

            if isinstance(preds, tuple):
                if len(preds) == 3:
                    preds_local, preds_distal, preds = preds
                    local_loss += criterion(preds_local, y.long().squeeze(1)).item()
                    distal_loss += criterion(preds_distal, y.long().squeeze(1)).item()
                elif len(preds) >= 4:
                    if len(preds) == 5:
                        preds, _ = preds[:-1], preds[-1]
                    preds_local, preds_distal, preds_distal2, preds = preds
                    local_loss += criterion(preds_local, y.long().squeeze(1)).item()
                    distal_loss += criterion(preds_distal, y.long().squeeze(1)).item()
                    distal_loss2 += criterion(preds_distal2, y.long().squeeze(1)).item()
                
            pred_y = torch.cat((pred_y, preds), dim=0)
                
            loss = criterion(preds, y.long().squeeze(1))
            total_loss += loss.item()

            if batch_count % 500 == 0:
                print(f"Batch Number: {batch_count}; Mean Time of 500 batch: {(time.time()-step_time)}")
                step_time = time.time()
 
            get_batch_predict_recode += time.time() - batch_predict_time
            if batch_count % 500 == 0:
                print("training 500 batch used time:", get_batch_predict_recode)
                get_batch_predict_recode = 0
            get_batch_time = time.time()

            if device == torch.device('cpu'):
                if  np.random.uniform(0,1) < 0.0001:
                    #print('in the model_predict_m:', device)
                    sys.stdout.flush()
            
    # time view
    print(f"Batch Number: {batch_count}; prediction Time of {batch_count} batch: {(time.time()-step_time)}")
    sys.stdout.flush()

    if distal_loss2:
        return pred_y, total_loss, local_loss, distal_loss, distal_loss2
    if local_loss:
        return pred_y, total_loss, local_loss, distal_loss
    return pred_y, total_loss

