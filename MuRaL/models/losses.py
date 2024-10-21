import torch
import sys


class LossFactory():
    def __init__(self) -> None:
        pass

    def create_loss(self, loss_name=None):
        if loss_name is None:
            return torch.nn.CrossEntropyLoss(reduction='sum')

def compute_local_distal_loss(preds, y, criterion):
    preds_local, preds_distal, preds = preds
    loss_local = criterion(preds_local, y)
    loss_distal = criterion(preds_distal, y)
    loss = criterion(preds, y)
    return loss_local, loss_distal, loss

def compute_complex_loss_with_decoder(preds, y, criterion):
    preds_local, preds_mid, preds_distal, preds, loss_construct = preds
    loss_local = criterion(preds_local, y)
    loss_mid = criterion(preds_mid, y)
    loss_distal = criterion(preds_distal, y)
    if isinstance(loss_construct, tuple):
        loss_construct, loss_descrim = loss_construct

    loss = criterion(preds, y)
    return loss_local, loss_mid, loss_distal, loss, loss_construct

def compute_mid_distal_loss(preds, y, criterion):
    """处理中间和远程预测的损失计算"""
    preds_local, preds_mid, preds_distal, preds = preds
    loss_local = criterion(preds_local, y)
    loss_mid = criterion(preds_mid, y)
    loss_distal = criterion(preds_distal, y)
    loss = criterion(preds, y)
    return loss_local, loss_mid, loss_distal, loss    

class LossCalcStrategyFactory:
    @staticmethod
    def get_loss_strategy(strategy_name=None):
        loss_calc_startegy_name = {
            'OnlyCombined': OnlyCombinedLossStrategy,
            'LocalDistalCombined': LocalDistalCombinedLossStrategy,
            'LocalMidDistalCombined': LocalMidDistalCombinedLossStrategy,
            'LocalMidDistalCombinedDecoder': LocalMidDistalCombinedDecoderLossStrategy,
            'segment_soft_label' : SegmentCombinedLossStrategy
            
        }
        # adapt Original test
        if strategy_name is None:
            return AdaptiveLossStrategy()
        else:
            return loss_calc_startegy_name[strategy_name]()


class AdaptiveLossStrategy():
    def __init__(self) -> None:
        self.loss = None 
    
    def calc_loss(self, preds, y, criterion):
        if isinstance(preds, tuple):
            if len(preds) == 3:
                res = compute_local_distal_loss(preds, y, criterion) 
                self.loss = res[-1]
                return res

            elif len(preds) >= 4:
                if len(preds) == 5:
                    res = compute_complex_loss_with_decoder(preds, y, criterion)
                    self.loss = res[-2]
                    return res
                else:
                    res = compute_mid_distal_loss(preds, y, criterion)
                    self.loss = res[-1]
                    return res
            else:
                sys.exit("Error: unconsider situation, check AdaptiveLossStrategy")
        else:
            res = criterion(preds, y)
            return res
    
    def extract_total_loss(self):
        return self.loss
    
    def extract_pred(self, preds):
        if isinstance(preds, tuple):
            if len(preds) == 3:
                preds_local, preds_distal, preds = preds
            if len(preds) == 4:
                preds_local, preds_mid, preds_distal, preds = preds
            if len(preds) == 5:
                preds_local, preds_mid, preds_distal, preds, loss_construct = preds

        return preds


class OnlyCombinedLossStrategy():
    def __init__(self) -> None:
        self.loss = None

    def calc_loss(self, preds, y, criterion):
        loss = criterion(preds, y)
        self.loss = loss
        return loss
    
    def extract_total_loss(self):
        return self.loss

class LocalDistalCombinedLossStrategy():
    def __init__(self) -> None:
        self.loss = None

    def calc_loss(self, preds, y, criterion):
        loss_local, loss_distal, loss = compute_local_distal_loss(preds, y, criterion)
        self.loss = loss
        return loss_local, loss_distal, loss
    
    def extract_total_loss(self):
        return self.loss

class LocalMidDistalCombinedLossStrategy():
    def __init__(self) -> None:
        self.loss = None

    def calc_loss(self, preds, y, criterion):
        loss_local, loss_mid, loss_distal, loss = compute_mid_distal_loss(preds, y, criterion)
        self.loss = loss
        return loss_local, loss_mid, loss_distal, loss

    def extract_total_loss(self):
        return self.loss

class LocalMidDistalCombinedDecoderLossStrategy():
    def __init__(self) -> None:
        self.loss = None

    def calc_loss(self, preds, y, criterion):
        loss_local, loss_mid, loss_distal, loss, loss_construct = compute_complex_loss_with_decoder(preds, y, criterion)
        self.loss = loss
        return loss_local, loss_mid, loss_distal, loss, loss_construct
    def extract_total_loss(self):
        return self.loss



class SegmentCombinedLossStrategy():
    def __init__(self) -> None:
        self.loss = None
    
    def calc_loss(self, preds, labels, criterion):


        return loss_local, loss_mid, loss_distal, loss
    

    def extract_total_loss(self):
        return self.loss
        

