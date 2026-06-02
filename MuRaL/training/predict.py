import torch
import time

import torch.nn.functional as F
from MuRaL.evaluation.observer import Observer, TimeMinor, GradMinor, LossMinor, PredsRecoder, MuRRecoder, ContributionMinor, SubModelPredResRecoder, ContributionMinor2, DirMDNRecoder, GammaMDNRecoder
from MuRaL.training.train import TrainerSubject, get_inputs_labels, model_train_register




class Predictor(TrainerSubject):
    def __init__(self, model, loss_calculator, criterion, device, config, observer=None, train_strategy=None, printer=print, detach=False, collect_mu_r=False, collect_evidence=False, collect_gamma_mdn=False) -> None:

        super().__init__()

        self.model = model
        self.criterion = criterion
        self.device = device
        self.config = config
        self.LossCalculator = loss_calculator
        self.printer = printer
        self.train_strategy = train_strategy
        self.detach = detach
        self.collect_mu_r = collect_mu_r
        self.collect_evidence = collect_evidence
        self.collect_gamma_mdn = collect_gamma_mdn

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
        if collect_mu_r:
            self._mu_r_recoder = MuRRecoder()
        if collect_evidence:
            self._dir_mdn_recoder = DirMDNRecoder()
        if collect_gamma_mdn:
            self._gamma_mdn_recoder = GammaMDNRecoder()

    def predict(self, dataloader_test):
        self.register_observer(self.valid_preds_recoder)
        self.register_observer(self.contribution_minor)
        if self.collect_mu_r:
            self.register_observer(self._mu_r_recoder)
        if self.collect_evidence:
            self.register_observer(self._dir_mdn_recoder)
        if self.collect_gamma_mdn:
            self.register_observer(self._gamma_mdn_recoder)
        self.model.eval()
        valid_step_time = time.time()
        with torch.no_grad():
            for batch in dataloader_test:
                batch = self.load_to_device(batch, self.device)
                label, inputs, sample_weight = get_inputs_labels(batch, self.train_strategy)
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
        if self.collect_mu_r:
            self.remove_observer(self._mu_r_recoder)
        if self.collect_evidence:
            self.remove_observer(self._dir_mdn_recoder)
        if self.collect_gamma_mdn:
            self.remove_observer(self._gamma_mdn_recoder)
        return valid_preds
    
    def predict_each_model(self, dataloader_test):
        self.register_observer(self.each_model_preds_recoder)
        self.register_observer(self.contribution_minor_split_mut_type)
        self.model.eval()
        valid_step_time = time.time()
        with torch.no_grad():
            for batch in dataloader_test:
                batch = self.load_to_device(batch, self.device)
                label, inputs, sample_weight = get_inputs_labels(batch, self.train_strategy)
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

    def get_mu_r(self):
        """返回收集的 mu 和 r（已激活的正值）。

        仅当 collect_mu_r=True 且模型为 NB variant 时返回有效值。
        CE 模型返回 (None, None)。
        """
        if hasattr(self, '_mu_r_recoder'):
            return self._mu_r_recoder.output()
        return None, None

    def get_evidence(self):
        """返回收集的 evidence 向量。

        仅当 collect_evidence=True 且模型为 DirMDN variant 时返回有效值。
        """
        if hasattr(self, '_dir_mdn_recoder'):
            return self._dir_mdn_recoder.output()
        return None

    def get_dir_mdn_components(self):
        """返回未激活的 DirMDN 分量 (pi_logits, alpha_raw)。

        - pi_logits: [B, K], raw (需 softmax 激活)
        - alpha_raw: [B, K, C], raw (需 softplus 激活)
        仅当 collect_evidence=True 且模型为 DirMDN variant 时返回有效值。
        """
        if hasattr(self, '_dir_mdn_recoder'):
            return self._dir_mdn_recoder.get_components()
        return None, None

    def get_pi_entropy(self):
        """返回收集的 pi_entropy 向量。

        仅当 collect_gamma_mdn=True 且模型为 Gamma MDN variant 时返回有效值。
        """
        if hasattr(self, '_gamma_mdn_recoder'):
            return self._gamma_mdn_recoder.output()
        return None

    def get_gamma_mdn_components(self):
        """返回未激活的 Gamma MDN 分量 (pi_logits, alpha_raw, beta_raw)。

        - pi_logits: [B, K], raw (需 softmax 激活)
        - alpha_raw:  [B, K, C], raw (需 softplus 激活)
        - beta_raw:   [B, K, C], raw (需 softplus 激活)
        仅当 collect_gamma_mdn=True 且模型为 Gamma MDN variant 时返回有效值。
        """
        if hasattr(self, '_gamma_mdn_recoder'):
            return self._gamma_mdn_recoder.get_components()
        return None, None, None

def model_predict(batch, model, detach, strategy):
    if detach:
        cont_x, cat_x, distal_x = batch
        return model.predict((cont_x, cat_x), distal_x)
    else:
        #return model.forward((cont_x, cat_x), distal_x)
        model_train = model_train_register(strategy)
        return model_train(batch, model)

class BayesianPredictor(TrainerSubject):
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
        self.num_monte_carlo = self.config.get('num_monte_carlo', 10)
        self.model_predict = model_train_register(self.train_strategy)

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

        pred_y_ensemble = torch.empty(0, self.config['n_class']).to(self.device)
        pred_y_uncertain = torch.empty(0, self.config['n_class']).to(self.device)

        with torch.no_grad():
            for batch in dataloader_test:
                batch = self.load_to_device(batch, self.device)
                label, inputs, sample_weight = get_inputs_labels(batch, self.train_strategy)
                pred_results = []
                output_mc = []
                for mc_run in range(int(self.num_monte_carlo)):
                    valid_preds = self.model_predict(inputs, self.model)
                    pred_results.append(valid_preds)
                    final_pred = self._extract_final_pred(valid_preds)
                    output_mc.append(F.softmax(final_pred, dim=1))
                pred_results = self._merge_mc_outputs(pred_results, mode="mean")
                output_mc = torch.stack(output_mc)
                means = output_mc.mean(axis=0)
                stds = output_mc.std(axis=0)
                pred_y_ensemble = torch.cat((pred_y_ensemble, means), dim=0)
                pred_y_uncertain = torch.cat((pred_y_uncertain, stds), dim=0)

                losses = self.LossCalculator.calc_loss(valid_preds, label, self.criterion)
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

        return pred_y_ensemble, pred_y_uncertain
    
    def load_to_device(self, batch, device):
        if isinstance(batch, dict):
            return {k: v.to(device) for k, v in batch.items()}
        elif isinstance(batch, (tuple, list)):
            return [v.to(device) for v in batch]
        else:
            return batch.to(device)
    def _merge_mc_outputs(self, outputs, mode="mean"):
        if mode not in ["mean"]:
            raise ValueError(f"mode {mode} not supported")
        agg_fn = torch.mean
        outputs = [o[0] for o in outputs if len(o) == 2] 
        if isinstance(outputs[0], dict):
            return {k: 
                    agg_fn(torch.stack([o[k] for o in outputs]), dim=0) 
                    for k in outputs[0].keys()}, None
        elif isinstance(outputs[0], torch.Tensor):
            return agg_fn(torch.stack(outputs), dim=0), None
        
    def _extract_final_pred(self, preds):
        preds = preds[0] if len(preds) == 2 else preds
        if isinstance(preds, dict):
            return preds['out']
        else:
            assert isinstance(preds, torch.Tensor) , "preds must be a torch.Tensor or a dict with key 'out'"
            return preds