import time
import numpy as np
import sys
import torch
from MuRaL.evaluation.evaluation import print_gradients, print_gradient_norms


class Observer:
    def update(self, **kwargs):
        raise NotImplementedError


class TimeMinor(Observer):
    def __init__(self, out_after_n_batch, dataset_clss='Training', printer=print):
        self.batch_load_times = []
        self.batch_train_times = []
        self.batch_total_times = []
        self.conter = 0
        self.out_after_n_batch = out_after_n_batch
        self.dataset_clss = dataset_clss
        self.printer = printer
    
    def record_batch_load(self, time):
        self.batch_load_times.append(time)

    def record_batch_train_times(self, time):
        self.batch_train_times.append(time)

    def record_batch_total_time(self, time):
        self.batch_total_times.append(time)

    def out_batch_times(self):
        self.printer(f"{self.dataset_clss} load {self.out_after_n_batch} batch used {np.sum(self.batch_load_times) / 60} min")
        self.printer(f"{self.dataset_clss} train {self.out_after_n_batch} batch used {np.sum(self.batch_train_times) / 60} min")
        self.printer(f"{self.dataset_clss} after {self.out_after_n_batch} batch used {np.sum(self.batch_total_times) / 60} min")
        self.reset()

    def reset(self):
        self.conter = 0
        self.batch_load_times = []
        self.batch_train_times = []
        self.batch_total_times = []

    def update(self, **kwargs):
        counter = False
        if 'batch_load_time' in kwargs:
            self.record_batch_load(kwargs['batch_load_time'])
            counter = True

        if 'batch_train_time' in kwargs:
            self.record_batch_train_times(kwargs['batch_train_time'])
            counter = True
            
        if 'batch_total_time' in kwargs: 
            self.record_batch_total_time(kwargs['batch_total_time'])
            counter = True

        if counter:
            self.conter += 1
            if self.conter == self.out_after_n_batch:
                self.out_batch_times()

class GradMinor(Observer):
    def __init__(self, out_after_n_batch, first_epoch=True, printer=print):
        self.grad_norms = []
        self.counter = 0
        self.out_after_n_batch = out_after_n_batch
        if first_epoch:
            self.out_epoch = 5
        else:
            self.out_epoch = 0
        self.printer = printer
    
    def out_grad(self, model):
        print_gradients(model, print=self.printer)
        print_gradient_norms(model, print=self.printer)
        self.reset()

    def reset(self):
        self.counter = 0

    def update(self, **kwargs):
        if 'model' in kwargs:
            self.counter += 1
            if self.out_epoch:
                self.out_grad(kwargs['model'])
                self.out_epoch -= 1

            if self.counter == self.out_after_n_batch:
                self.out_grad(kwargs['model'])
        
class LossMinorStrategy:
    def __init__(self):
        self.loss = []

    def record(self, loss):
        raise NotImplementedError("Must implement record method.")

    def reset(self):
        raise NotImplementedError("Must implement record method.")

    def get_total_loss(self):
        raise NotImplementedError("Must implement record method.")

class OnlyCombinedLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.printer = printer

    def record(self, loss):
        self.loss.append(loss.item())
    
    def reset(self):
        self.loss.clear()
    
    def out_mean_loss(self, dataset_class, sample_number):
        self.printer(f"{dataset_class} Total Loss: {np.sum(self.loss) / sample_number}")
        return {'loss' : self.loss}
    
class LocalDistalCombinedLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.local_loss = []
        self.distal_loss = []
        self.printer = printer 

    def record(self, losses):
        if len(losses) != 3:
            raise ValueError("Expected losses to be a tuple of length 3.")
        self.local_loss.append(losses[0].item())
        self.distal_loss.append(losses[1].item())
        self.loss.append(losses[2].item())  
    
    def reset(self):
        self.local_loss.clear()
        self.distal_loss.clear()
        self.loss.clear()
    
    def out_mean_loss(self, dataset_class, sample_number):
        self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}")
        self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}")
        self.printer(f"{dataset_class} Total Loss : {np.sum(self.loss) / sample_number}")
        return {'loss' : self.loss}

    def get_total_loss(self):
        return super().get_total_loss()

class LocalMidDistalCombinedLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.local_loss = []
        self.distal_loss = []
        self.mid_loss = []
        self.printer = printer

    def record(self, loss):
        self.local_loss.append(loss[0].item())
        self.mid_loss.append(loss[1].item())
        self.distal_loss.append(loss[2].item())
        self.loss.append(loss[3].item())

    def reset(self):
        self.loss.clear()
        self.local_loss.clear()
        self.distal_loss.clear()
        self.mid_loss.clear()
    
    def out_mean_loss(self, dataset_class, sample_number):
        self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}")
        self.printer(f"{dataset_class} Mid Loss: {np.sum(self.mid_loss) / sample_number}")
        self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}")
        self.printer(f"{dataset_class} Total Loss : {np.sum(self.loss) / sample_number}")
        return {'loss' : self.loss}

class LocalMidDistalCombinedDecoderLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print):
        super().__init__()
        self.local_loss = []
        self.distal_loss = []
        self.mid_loss = []
        self.decoder_loss = []
        self.printer = printer

    def record(self, loss):
        self.local_loss.append(loss[0].item())
        self.mid_loss.append(loss[1].item())
        self.distal_loss.append(loss[2].item())
        self.loss.append(loss[3].item())
        self.decoder_loss.append(loss[4].item())
    
    def reset(self):
        self.local_loss.clear()
        self.mid_loss.clear()
        self.distal_loss.clear()
        self.decoder_loss.clear()
        self.loss.clear()
        
    
    def out_mean_loss(self, dataset_class, sample_number):
        self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}")
        self.printer(f"{dataset_class} Mid Loss: {np.sum(self.mid_loss) / sample_number}")
        self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}")
        self.printer(f"{dataset_class} construct Loss : {np.sum(self.decoder_loss) / sample_number}")
        self.printer(f"{dataset_class} Total Loss : {np.sum(self.loss) / sample_number}")
        return {'loss' : self.loss}

class AdaptiveLossMinorStrategy(LossMinorStrategy):
    def __init__(self, printer=print) -> None:
        super().__init__()
        self._record_init()
        self.printer = printer
    
    def _record_init(self):
        self.loss = []
        self.local_loss = []
        self.distal_loss = []
        self.mid_loss = []
        self.decoder_loss = []
 
    
    def record(self, loss):

        if isinstance(loss, torch.Tensor):
            self.loss.append(loss.item())
        elif isinstance(loss, tuple):
            if len(loss) == 3:
                self.local_loss.append(loss[0].item())
                self.distal_loss.append(loss[1].item())
                self.loss.append(loss[2].item())
            elif len(loss) == 4:
                self.local_loss.append(loss[0].item())
                self.mid_loss.append(loss[1].item())
                self.distal_loss.append(loss[2].item())
                self.loss.append(loss[3].item())
            elif len(loss) == 5:
                self.local_loss.append(loss[0].item())
                self.mid_loss.append(loss[1].item())
                self.distal_loss.append(loss[2].item())
                self.loss.append(loss[3].item())
                self.decoder_loss.append(loss[4].item())
            else:
                self.printer("Warming: Loss Record unnormal")
    def reset(self):
        self._record_init()

    def out_mean_loss(self, dataset_class, sample_number):
        if self.local_loss:
            self.printer(f"{dataset_class} Local Loss: {np.sum(self.local_loss) / sample_number}")
        if self.mid_loss:
            self.printer(f"{dataset_class} Mid Loss: {np.sum(self.mid_loss) / sample_number}")
        if self.distal_loss:
            self.printer(f"{dataset_class} Distsal Loss: {np.sum(self.distal_loss) / sample_number}")
        if self.decoder_loss:
            self.printer(f"{dataset_class} construct Loss : {np.sum(self.decoder_loss) / sample_number}")
        if self.loss:
            loss = np.sum(self.loss) / sample_number
            self.printer(f"{dataset_class} Total Loss : {loss}") 
        return {'loss' : loss}
    
class LossMinor(Observer):
    strategy_map = {
        'OnlyCombined': OnlyCombinedLossMinorStrategy,
        'LocalDistalCombined': LocalDistalCombinedLossMinorStrategy,
        'LocalMidDistalCombined': LocalMidDistalCombinedLossMinorStrategy,
        'LocalMidDistalCombinedDecoder': LocalMidDistalCombinedDecoderLossMinorStrategy,  # Add as needed
    }

    def __init__(self, calc_loss_strategy_name=None, printer=print):
        if calc_loss_strategy_name is None:
            self.loss_strategy = AdaptiveLossMinorStrategy()

        else:
            if calc_loss_strategy_name not in self.strategy_map:
                sys.exit("Error: Unsupported strategy name")
            self.loss_strategy = self.strategy_map[calc_loss_strategy_name]()
        self.printer = printer
        self.sample_number = 0

    def record_loss(self, loss):
        self.loss_strategy.record(loss)
    
    def record_sample_number(self, sample_number):
        self.sample_number += sample_number

    def reset(self):
        self.loss_strategy.reset()
        self.sample_number = 0

    def out_mean_losses(self, dataset_class, sample_number):
        minor_dict = self.loss_strategy.out_mean_loss(dataset_class, sample_number)
        self.reset()
        return minor_dict

    def update(self, **kwargs):
        if 'losses' in kwargs:
            self.record_loss(kwargs['losses'])
            self.record_sample_number(kwargs['sample_number'])
        if 'train_step_finish' in kwargs:
            minor_dict = self.out_mean_losses(dataset_class="Training", sample_number=self.sample_number)
            return {}

        if 'valid_step_finish' in kwargs:
            minor_dict = self.out_mean_losses(dataset_class="Validation", sample_number=self.sample_number)
            return {'valid_loss': minor_dict['loss']}

        


# class LossMinorStrategy:
#     def __init__(self, printer=print):
#         self.losses = {}
#         self.printer = printer

#     def record(self, loss):
#         raise NotImplementedError("Must implement record method.")

#     def reset(self):
#         self.losses.clear()

#     def out_mean_loss(self, dataset_class, sample_number):
#         for loss_type, loss_values in self.losses.items():
#             if loss_values:
#                 mean_loss = np.sum(loss_values) / sample_number
#                 self.printer(f"{dataset_class} {loss_type}: {mean_loss}")

# class OnlyCombinedLossMinorStrategy(LossMinorStrategy):
#     def record(self, loss):
#         self.losses.setdefault('Total Loss', []).append(loss.item())

# class LocalDistalCombinedLossMinorStrategy(LossMinorStrategy):
#     def record(self, losses):
#         if len(losses) != 3:
#             raise ValueError("Expected losses to be a tuple of length 3.")
#         self.losses.setdefault('Local Loss', []).append(losses[0].item())
#         self.losses.setdefault('Distal Loss', []).append(losses[1].item())
#         self.losses.setdefault('Total Loss', []).append(losses[2].item())

# class LocalMidDistalCombinedLossMinorStrategy(LossMinorStrategy):
#     def record(self, loss):
#         if len(loss) != 4:
#             raise ValueError("Expected losses to be a tuple of length 4.")
#         self.losses.setdefault('Local Loss', []).append(loss[0].item())
#         self.losses.setdefault('Mid Loss', []).append(loss[1].item())
#         self.losses.setdefault('Distal Loss', []).append(loss[2].item())
#         self.losses.setdefault('Total Loss', []).append(loss[3].item())

# class LocalMidDistalCombinedDecoderLossMinorStrategy(LossMinorStrategy):
#     def record(self, loss):
#         if len(loss) != 5:
#             raise ValueError("Expected losses to be a tuple of length 5.")
#         self.losses.setdefault('Local Loss', []).append(loss[0].item())
#         self.losses.setdefault('Mid Loss', []).append(loss[1].item())
#         self.losses.setdefault('Distal Loss', []).append(loss[2].item())
#         self.losses.setdefault('Construct Loss', []).append(loss[3].item())
#         self.losses.setdefault('Decoder Loss', []).append(loss[4].item())

# class AdaptiveLossMinorStrategy(LossMinorStrategy):
#     def _record_init(self):
#         self.losses = {}

#     def record(self, loss):
#         if isinstance(loss, torch.Tensor):
#             self.losses.setdefault('Total Loss', []).append(loss.item())
#         elif isinstance(loss, tuple):
#             for i, key in enumerate(['Local Loss', 'Mid Loss', 'Distal Loss', 'Total Loss', 'Decoder Loss']):
#                 if i < len(loss):
#                     self.losses.setdefault(key, []).append(loss[i].item())
#                 else:
#                     self.printer("Warning: Loss record is abnormal.")


class ModelSaverObserve(Observer):
    def __init__(self, model_saver, printer=print):
        self.model_saver = model_saver
        self.printer = printer

    def update(self, **kwargs):
        if 'epoch_finish' in kwargs:
            self.model_saver.save_model(kwargs['epoch_finish'])
            self.printer(f"Model saved at epoch {kwargs['epoch_finish']}")

class PredsRecoder(Observer):

    def __init__(self) -> None:
        super().__init__()

        self.preds = None
    
    def reset(self):
        self.preds = None
    
    def recode(self, preds):
        preds = self.extract_preds(preds)

        if self.preds is None:
            self.preds = preds
        else:
            self.preds = torch.cat([self.preds, preds], dim=0)
        
    def extract_preds(self, preds):
        if isinstance(preds, tuple):
            if len(preds) == 3:
                preds_local, preds_distal, pred = preds
            if len(preds) == 4:
                preds_local, preds_mid, preds_distal, pred = preds
            if len(preds) == 5:
                preds_local, preds_mid, preds_distal, pred, loss_construct = preds
        return pred

    
    def output(self):
        preds = self.preds
        self.reset()
        return preds

    
    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            preds = kwargs['valid_preds']
            self.recode(preds)


class ContributionMinor(Observer):
    def __init__(self, printer=print):
        self.mean_contributions = None
        self.calc_var_contributions = None
        self.printer = printer

    def record(self, preds_each_model: dict):

        mean_contributions = self.calc_mean_contribution(preds_each_model)
        var_contributions = self.calc_var_contribution(preds_each_model)

        if self.mean_contributions is None:
            self.mean_contributions = mean_contributions
            self.var_contributions = var_contributions
        else:
            self.update_contributions('mean_contribution', mean_contributions)
            self.update_contributions('var_contribution', var_contributions)
    
    def calc_mean_contribution(self, preds_each_model: dict):
        mean_contributions = {}
        fused_preds = self.convert_to_numpy(preds_each_model['fused_pred'])
        # Note: each model output should after softmax or sigmoid to ensure each element large than 0
        for model_name, preds in preds_each_model.items():
            if model_name == 'fused_pred':
                continue
            preds = self.convert_to_numpy(preds)
            contribution = np.mean(preds / fused_preds, axis=0)
            mean_contributions[model_name] = [contribution]
        return mean_contributions

    def calc_var_contribution(self, preds_each_model: dict):
        var_contributions = {}

        for model_name, preds in preds_each_model.items():
            if model_name == 'fused_pred':
                continue
            preds = self.convert_to_numpy(preds)
            contribution = np.var(preds, axis=0)
            var_contributions[model_name] = [contribution]
        return var_contributions 
    
    def update_contributions(self, contribution_name , contributions):

        if contribution_name == 'mean_contribution':
            for model_name, contribution in contributions.items():
                self.mean_contributions[model_name] = np.concatenate([self.mean_contributions[model_name], contributions[model_name]])

        elif contribution_name == 'var_contribution':
            for model_name, contribution in contributions.items():
                self.var_contributions[model_name] = np.concatenate([self.var_contributions[model_name], contributions[model_name]])
        else:
            raise ValueError("Error: For ContributionMinor, contribution_name must be 'mean_contribution' or 'var_contribution'")
    
    def convert_to_numpy(self, contribution):
        if contribution.is_cuda:
            contribution = contribution.cpu().numpy()
        else:
            contribution = contribution.numpy()
        return contribution
    
    def extract_sub_model_preds(self, preds):
        if isinstance(preds, tuple):
            if len(preds) == 3:
                preds_local, preds_distal, pred = preds
                return {
                    'local': preds_local,
                    'distal': preds_distal,
                    'fused_pred': pred
                }
            if len(preds) == 4:
                preds_local, preds_mid, preds_distal, pred = preds

            if len(preds) == 5:
                preds_local, preds_mid, preds_distal, pred, loss_construct = preds

            return {
                    'local': preds_local,
                    'mid': preds_mid,
                    'distal': preds_distal,
                    'fused_pred': pred
                }
        return None

    def reset(self):
        self.mean_contributions = None
        self.var_contributions = None
    
    def out_contribution(self):
        if self.mean_contributions is None:
            self.printer("Only one preds out, No contribution to report.")
            return

        for model_name, contribution in self.mean_contributions.items():
            self.printer(f"{model_name} Mean abs contribution in validation:", np.mean(contribution, axis=0))

        for model_name, contribution in self.var_contributions.items():
            self.printer(f"{model_name} Mean Var in validation:", np.mean(contribution, axis=0))

    def update(self, **kwargs):
        if 'valid_preds' in kwargs:
            preds = kwargs['valid_preds']
            preds_each_model = self.extract_sub_model_preds(preds)
            if preds_each_model is not None:
                self.record(preds_each_model)

        if 'valid_step_finish' in kwargs:
            self.out_contribution()
            self.reset()