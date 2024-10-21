import torch
import time
import sys

import torch.nn as nn
from MuRaL.evaluation.observer import Observer, TimeMinor, GradMinor, LossMinor, PredsRecoder, ContributionMinor

class TrainerSubject:
    def __init__(self):
        self.observers = []
        self.metrics = {}

    def register_observer(self, observer: Observer):
        self.observers.append(observer)

    def remove_observer(self, observer: Observer):
        self.observers.remove(observer)

    def notify_observers(self, **kwargs):
        for observer in self.observers:
            indicator = observer.update(**kwargs)
            if indicator:
                self.metrics.update(indicator)


class Trainer(TrainerSubject):
    def __init__(self, model, optimizer, scheduler, loss_calculator, criterion, device, config, observer=None, train_strategy=None, printer=print) -> None:

        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        self.LossCalculator = loss_calculator
        self.printer = printer
        self.train_strategy = train_strategy

        if observer is None:
            self.observer = [TimeMinor(out_after_n_batch=1000), GradMinor(out_after_n_batch=1000), LossMinor()]
        else:
            self.observer = observer

        for observer in self.observer:
            self.register_observer(observer)
        
        self.valid_preds_recoder = PredsRecoder()
        self.contribution_minor = ContributionMinor()
        self.metrics = {}

    def train_step(self, data_loader):
        self.model.train()
        batch_total_time = time_tmp = time.time()
        batch_count = 0
        #for y, cont_x, cat_x, distal_x in data_loader:
        for batch in data_loader:
            batch_load_time = time.time() - time_tmp
            batch_count += 1
            sample_number = batch[0].shape[0]

            # load data
            time_tmp = time.time()

            batch = self.load_to_device(batch, self.device)
            #label, inputs = split_batch(batch)
            label, preds = model_train(batch, self.model, self.train_strategy)
            losses = self.LossCalculator.calc_loss(preds, label, self.criterion)
            loss = self.LossCalculator.extract_total_loss()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10, error_if_nonfinite=False)
            self.notify_observers(model=self.model)
            self.optimizer.step()
            batch_train_time = time.time() - time_tmp

            # updata lr
            self.update_lr()
            batch_total_time = time.time() - batch_total_time
            # record log
            self.notify_observers(losses=losses, 
                                  sample_number = sample_number,
                                  batch_train_time=batch_train_time, 
                                  batch_load_time=batch_load_time, 
                                  batch_total_time=batch_total_time)
            batch_total_time = time_tmp = time.time()
        
        self.notify_observers(train_step_finish = True)
        self.update_lr()
        sys.stdout.flush()

    def valid_step(self, dataloader_valid):
        self.register_observer(self.valid_preds_recoder)
        self.register_observer(self.contribution_minor)
        self.model.eval()
        valid_step_time = time.time()
        with torch.no_grad():
            for batch in dataloader_valid:
                batch = self.load_to_device(batch, self.device)
                label, valid_preds = model_predict(batch, self.model, self.train_strategy)
                losses = self.LossCalculator.calc_loss(valid_preds, label, self.criterion)
                #valid_pred = self.LossCalculator.extract_pred(valid_preds)
                sample_number = batch[0].shape[0]

                self.notify_observers(losses = losses,
                                      sample_number = sample_number,
                                      valid_preds = valid_preds)
            
            self.notify_observers(valid_step_finish = True)
        valid_step_time = time.time() - valid_step_time
        self.printer(f"Validation used time: {valid_step_time / 60} mins")
        valid_preds = self.valid_preds_recoder.output()

        self.remove_observer(self.valid_preds_recoder)
        self.remove_observer(self.contribution_minor)
        return valid_preds
                

    def update_lr(self):
        if self.config['lr_scheduler'] != 'ROP':
            self.scheduler.step()
            if self.optimizer.param_groups[0]['lr'] < self.config['min_lr']:
                self.printer("optimizer.param_groups[0]:", self.optimizer.param_groups[0]['lr'])
                for g in self.optimizer.param_groups:
                    g['lr'] = self.config['restart_lr']
        if self.config['lr_scheduler'] == 'ROP':
            self.scheduler.step(self.metrics['current_valid_loss'])

    def load_to_device(self, batch, device):
        if isinstance(batch, dict):
            return {k: v.to(device) for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            return [v.to(device) for v in batch]
        else:
            return batch.to(device)



def model_train(batch, model, strategy=None):
    model_train_strategy = {
        'segment_soft_label': model_train_with_segment_soft_label,
    }

    if strategy is None:
        y, cont_x, cat_x, distal_x = batch
        return y.long().squeeze(), model.forward((cont_x, cat_x), distal_x)
    
    if strategy not in model_train_strategy:
        sys.exit(f"Error: <{strategy}> unsupported model train strategy!")
    else:
        label, preds = model_train_strategy[strategy](batch, model)
        return label, preds

def model_train_with_segment_soft_label(batch, model):
    y, cont_x, cat_x, distal_x, soft_label = batch
    return (y.long().squeeze(), soft_label), model.forward((cont_x, cat_x), distal_x)


def model_predict(batch, model, strategy=None):
    model_train_strategy = {
        'segment_soft_label': model_train_with_segment_soft_label,
    }

    if strategy is None:
        y, cont_x, cat_x, distal_x = batch
        return y.long().squeeze(), model.forward((cont_x, cat_x), distal_x)
    else:
        label, preds = model_train_strategy[strategy](batch, model)
        return label, preds
 
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

class TorchBackendManager:
    def __init__(self, use_dilation=False, input_size_fixed=True, printer=None):
        self.use_dilation = use_dilation
        self.input_size_fixed = input_size_fixed

        if printer is not None:
            self.printer = printer
        else:
            self.printer = print

    def set_torch_backends(self):
        """Configure PyTorch backend settings based on dilation usage."""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = not self.use_dilation
        torch.backends.cudnn.benchmark = not self.input_size_fixed
        self.display_torch_backends_info()

    def display_torch_backends_info(self):
        """Display the current PyTorch backend settings."""
        self.printer(f"TF32 Matmul Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        self.printer(f"CUDNN Benchmark Enabled: {torch.backends.cudnn.benchmark}")
        self.printer(f"CUDNN TF32 Enabled: {torch.backends.cudnn.allow_tf32}")
        self.printer(f"CUDNN Deterministic: {torch.backends.cudnn.deterministic}")

    def display_torch_device_info(self):
        """Display the current PyTorch device information."""
        self.printer("torch._C._cuda_getDeviceCount():", torch._C._cuda_getDeviceCount())
        self.printer("torch.cuda.device_count(): ", torch.cuda.device_count())


