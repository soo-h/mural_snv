import torch
import time
import sys

import torch.nn as nn
from MuRaL.evaluation.observer import Observer, TimeMinor, GradMinor, LossMinor, PredsRecoder, ContributionMinor, SubModelPredResRecoder, ContributionMinor2
from MuRaL.training.train import TrainerSubject, get_inputs_labels, model_train




class Predictor(TrainerSubject):
    def __init__(self, model, loss_calculator, criterion, device, config, observer=None, train_strategy=None, printer=print, detach=False) -> None:

        super().__init__()

        self.model = model
        self.criterion = criterion
        self.device = device
        self.config = config
        self.LossCalculator = loss_calculator
        self.printer = printer
        self.train_strategy = train_strategy
        self.detach = detach

        if observer is None:
            self.observer = [TimeMinor(out_after_n_batch=1000),LossMinor()]
        else:
            self.observer = observer

        for observer in self.observer:
            self.register_observer(observer)
        
        self.valid_preds_recoder = PredsRecoder()
        self.each_model_preds_recoder = SubModelPredResRecoder()
        self.contribution_minor = ContributionMinor2()
        self.contribution_minor_split_mut_type = ContributionMinor2()
        self.metrics = {}

    def predict(self, dataloader_test):
        self.register_observer(self.valid_preds_recoder)
        self.register_observer(self.contribution_minor)
        self.model.eval()
        valid_step_time = time.time()
        with torch.no_grad():
            for batch in dataloader_test:
                batch = self.load_to_device(batch, self.device)
                label, inputs = get_inputs_labels(batch, self.train_strategy)
                valid_preds = model_predict(inputs, self.model, self.detach, self.train_strategy)
                #label, valid_preds = model_predict(batch, self.model, self.train_strategy)
                losses = self.LossCalculator.calc_loss(valid_preds, label, self.criterion)
                #valid_pred = self.LossCalculator.extract_pred(valid_preds)
                sample_number = batch[0].shape[0]

                self.notify_observers(losses = losses,
                                      sample_number = sample_number,
                                      valid_preds = valid_preds,
                                      label=label)
            
            self.notify_observers(valid_step_finish = True)
        valid_step_time = time.time() - valid_step_time
        self.printer(f"Validation used time: {valid_step_time / 60} mins")
        valid_preds = self.valid_preds_recoder.output()

        self.remove_observer(self.valid_preds_recoder)
        self.remove_observer(self.contribution_minor)
        return valid_preds
    
    def predict_each_model(self, dataloader_test):
        self.register_observer(self.each_model_preds_recoder)
        self.register_observer(self.contribution_minor_split_mut_type)
        self.model.eval()
        valid_step_time = time.time()
        with torch.no_grad():
            for batch in dataloader_test:
                batch = self.load_to_device(batch, self.device)
                label, inputs = get_inputs_labels(batch, self.train_strategy)
                valid_preds = model_predict(inputs, self.model, self.detach, self.train_strategy)
                #label, valid_preds = model_predict(batch, self.model, self.train_strategy)
                losses = self.LossCalculator.calc_loss(valid_preds, label, self.criterion)
                #valid_pred = self.LossCalculator.extract_pred(valid_preds)
                sample_number = batch[0].shape[0]

                self.notify_observers(losses = losses,
                                      sample_number = sample_number,
                                      valid_preds = valid_preds,
                                      label = label
                                      )
            
            self.notify_observers(valid_step_finish = True)
        valid_step_time = time.time() - valid_step_time
        self.printer(f"Validation used time: {valid_step_time / 60} mins")
        valid_preds = self.each_model_preds_recoder.output()

        self.remove_observer(self.each_model_preds_recoder)
        self.remove_observer(self.contribution_minor_split_mut_type)
        return valid_preds
                

    def load_to_device(self, batch, device):
        if isinstance(batch, dict):
            return {k: v.to(device) for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            return [v.to(device) for v in batch]
        else:
            return batch.to(device)

def model_predict(batch, model, detach, strategy):
    if detach:
        cont_x, cat_x, distal_x = batch
        return model.predict((cont_x, cat_x), distal_x)
    else:
        #return model.forward((cont_x, cat_x), distal_x)
        return model_train(batch, model, strategy)

