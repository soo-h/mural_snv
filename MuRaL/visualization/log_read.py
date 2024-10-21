import re
import sys
import numpy as np
import pandas as pd

class TimeCost:
    def __init__(self):
        self._preprocess_times = {'train': None, 'valid': None}
        self._epoch_times = []
        self.total_batch_time = None
        self._get_batch_time = 0
        self._train_batch_time = 0

    @property
    def preprocess_times(self):
        return self._preprocess_times

    @property
    def epoch_times(self):
        return self._epoch_times

    @property
    def get_batch_time(self):
        return self._get_batch_time

    @property
    def train_batch_time(self):
        return self._train_batch_time

    @preprocess_times.setter
    def preprocess_times(self, times):
        self._preprocess_times.update(times)

    @epoch_times.setter
    def epoch_times(self, times):
        self._epoch_times.extend(times)


class Metrics:
    def __init__(self):
        self._kmer_metrics = {}
        self._regional_metrics = {}
        self._loss_metrics = {}

    @property
    def kmer_metrics(self):
        return self._kmer_metrics

    @property
    def regional_metrics(self):
        return self._regional_metrics

    @property
    def loss_metrics(self):
        return self._loss_metrics

    
class LogParser:
    def __init__(self, file_path):
        self.time_cost = TimeCost()
        self.metrics = Metrics()
        self.file_path = file_path
        self.contri_dict = {}
        self.read_log()

    def read_log(self):
        with open(self.file_path, 'r') as file:
            for line in file:
                self.process_line(line.strip())

    def extract_preprocess_time(self, line):
        parts = line.split()
        value = float(parts[-1])
        key = 'train' if 'training' in parts else 'valid' if 'validation' in parts else None
        if key:
            return {key: value}
        else:
            print(f'Error: preprocess time info not in line <{line}>!')
            sys.exit(1)
    
    def extract_used_time(self, line):
        parts = line.split()
        value = float(parts[-1])
        return value

    def extract_loss(self, line):
        try:
            key, value = line.split(':')
        except:
            print(f"Warming: {line} can not be split by ':' ")
            return None
        
        value = float(value)
        if 'Training Loss' in key:
            key = 'train_loss' 
        elif 'fdiri_cal' in key: 
            key = 'valid_loss_fdiri_cal'
        elif 'Validation Loss' in key:
            key = 'valid_loss'
        elif 'Validation local Loss' in key:
            key = 'valid_local_loss'
        elif 'Validation distal Loss' in key:
            key = 'valid_distal_loss'
        return key, value
    
    def extract_kmer_correlation(self, line):
        line = line.strip()
        prefix = line.split()[0]
        if 'after' in line:
            key = prefix + "_fdiri_cal"
        else:
            key = prefix + "_all"
        correlation_values = [float(x) for x in re.findall(r'\d+\.\d+', line)]
        return key, correlation_values

    def extract_regional_correlation(self, line):
        line = line.strip()
        if 'after' in line:
            key = re.findall(r'(\d+)bp', line)[0] + 'bp_valid_fdiri_cal'
        else:
            key = re.findall(r'(\d+)bp', line)[0] + 'bp_valid'
        values = [float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', line)]
        return key, values
    
    def extract_contribution_each_model(self, line):
        line = line.strip()
        # Extract the contribution type (e.g., 'local', 'distal1', 'distal2')
        key_match = re.search(r'([a-zA-Z]+\d*) contribution', line)
        if key_match:
            key = key_match.group(1)
        else:
            raise ValueError("Invalid contribution line format: missing contribution type.")

        # Extract the weight associated with this contribution
        weight_match = re.search(r'weights:\s*([\d.]+)', line)
        if weight_match:
            weight = float(weight_match.group(1))
        else:
            raise ValueError("Invalid contribution line format: missing weight.")

        # Extract the contribution values
        values = [float(x) for x in re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?', line)]

        return key, weight, values


    def process_line(self, line):
        if "used time" in line:
            if 'preprocess' in line:
                self.time_cost.preprocess_times = self.extract_preprocess_time(line)       
            elif line.startswith('get'):
                self.time_cost._get_batch_time += self.extract_used_time(line)
            elif line.startswith('training'):
                self.time_cost._train_batch_time += self.extract_used_time(line)
        
            elif line.startswith('Epoch'):
                self.time_cost.epoch_times = [float(line.split()[-2].split(':')[-1])]

        elif "mer" in line:
            key, values = self.extract_kmer_correlation(line)
            if key not in self.metrics.kmer_metrics:
                self.metrics.kmer_metrics[key] = []
            self.metrics.kmer_metrics[key].append(values) 
            
        elif "regional" in line and "score" not in line:
            key, values = self.extract_regional_correlation(line)
            if key not in self.metrics.regional_metrics:
                self.metrics.regional_metrics[key] = []
            self.metrics.regional_metrics[key].append(values)
        
        elif "Loss" in line and 'LossTracker' not in line and "BCE" not in line:
            out = self.extract_loss(line)
            if out is None:
                return 0
            key,value = out

            if key not in self.metrics.loss_metrics:
                self.metrics.loss_metrics[key] = []
            self.metrics.loss_metrics[key].append(value)
        
        elif "contribution" in line:
            key, weights, values = self.extract_contribution_each_model(line)
            key = key+ str(weights)
            if key not in self.contri_dict:
                self.contri_dict[key] = []
            self.contri_dict[key].append(values) 

def main():
    demo_file = '/public/home/songhui/project/Mural/model_structure/Unet/Unet_sequence_fc/distal1k_localAndUnet_batch16.seqFc_distal16000.out'
    data_parser = LogParser(demo_file)

    print(data_parser.time_cost.epoch_times)
    print(data_parser.time_cost.get_batch_time)
    print(data_parser.time_cost.train_batch_time)
    print(data_parser.metrics.kmer_metrics)
    print(data_parser.metrics.regional_metrics)
    print(data_parser.metrics.loss_metrics)

if __name__ == "__main__":
    main()

