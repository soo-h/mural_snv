import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from pybedtools import BedTool

import sys
import argparse
import textwrap
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.append('/public/home/songhui/project/Mural/Mural_repo/MuRaL_112/model_utils')
from model_config import ModelFactory


import pandas as pd
import numpy as np
import pickle

import os
import time
import datetime

from MuRaL.utils.gpu_utils import get_available_gpu, check_cuda_id
from MuRaL.models.nn_models import *
from MuRaL.models.nn_utils import *
from MuRaL.scripts.training import set_torch_backends
from MuRaL.evaluation.evaluation import *
from MuRaL._version import __version__

# from MuRaL.custom_dataloader import MyDataLoader
from MuRaL.data.preprocessing import get_position_info, generate_data_batches, to_np
from MuRaL.data.data_preprocess_pipeline import DatasetPreprocessor
from MuRaL.data.dataset import dict_to_tuple_collate
from pynvml import *

from MuRaL.models.losses import LossFactory, LossCalcStrategyFactory
from MuRaL.evaluation.observer import TimeMinor, LossMinor
from MuRaL.training.predict import Predictor, BayesianPredictor

from MuRaL.utils.config_utils import read_bnn_config, read_feature_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

def parse_arguments(parser):
    """
    Parse parameters from the command line
    """ 
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    Bayes_args = parser.add_argument_group('Bayes-related arguments')
    optional.title = 'Other arguments' 
    
    required.add_argument('--ref_genome', type=str, metavar='FILE', default='',  
                          required=True, help=textwrap.dedent("""
                          File path of the reference genome in FASTA format.""").strip())
    
    required.add_argument('--test_data', type=str, metavar='FILE', required=True,
                          help= textwrap.dedent("""
                          File path of the data to do prediction, in BED format.""").strip())
    
    required.add_argument('--model_path', type=str, metavar='FILE', required=True,
                          help=textwrap.dedent("""
                          File path of the trained model.
                          """ ).strip())
        
    required.add_argument('--model_config_path', type=str, metavar='FILE', required=True,
                          help=textwrap.dedent("""
                          File path for the configurations of the trained model.
                          """ ).strip()) 
                            
    optional.add_argument('--distal_encoding', 
                            type=str,
                            default=None)

    optional.add_argument('--calc_loss_strategy_name', type=str, default=None)

    optional.add_argument('--distal_order', metavar='INT', default=1, 
                          help=textwrap.dedent("""
                          Default: 1.
                          """ ).strip())

    optional.add_argument('--cuda_id', type=str, metavar='STR', default=None, 
                          help=textwrap.dedent("""
                          Which GPU device to be used. Default: '0'. 
                          """ ).strip())
    
    optional.add_argument('--pred_file', type=str, metavar='FILE', default='pred.tsv.gz', help=textwrap.dedent("""
                          Name of the output file for prediction results.
                          Default: 'pred.tsv.gz'.
                          """ ).strip())
        
    optional.add_argument('--calibrator_path', type=str, metavar='FILE', default='',help=textwrap.dedent("""
                          File path for the paired calibrator of the trained model.
                          """ ).strip())
    
    optional.add_argument('--bw_paths', type=str, metavar='FILE', default=None,
                          help=textwrap.dedent("""
                          File path for a list of BigWig files for non-sequence 
                          features such as the coverage track. Default: None.""").strip())
    optional.add_argument('--n_h5_files', metavar='INT', default=1, 
                          help=textwrap.dedent("""
                          Number of HDF5 files for each BED file. When the BED file has many
                          positions and the distal radius is large, increasing the value for 
                          --n_h5_files files can reduce the time for generating HDF5 files.
                          Default: 1.
                          """ ).strip())
    
    optional.add_argument('--pred_time_view', default=False, action='store_true',  
                          help=textwrap.dedent("""
                          Check pred time of each part. Default: False.
                          """).strip())

    optional.add_argument('--use_single_base_task', 
                            nargs='?',
                            const=True,
                            default=False,
                            help=textwrap.dedent("""
                            use ref seq information.
                            """).strip())

    optional.add_argument('--with_h5', default=False, action='store_true',  
                          help=textwrap.dedent("""
                          Generate HDF5 file for the BED file. Default: False.
                          """).strip())

    optional.add_argument('--h5f_path', type=str, default=None,
                    help=textwrap.dedent("""
                    Specify the folder to generate HDF5. Default: Folder containing the BED file.""").strip())

    optional.add_argument('--save_each_model_preds', default=False, action='store_true',
                    help=textwrap.dedent("""
                    Out each model prediction result.""").strip())


    optional.add_argument('--cpu_only', default=False, action='store_true',  
                          help=textwrap.dedent("""
                          Only use CPU computing. Default: False.
                          """).strip())
    
    # optional.add_argument('--custom_dataloader', default=False, action='store_true',  
    #                       help=textwrap.dedent("""
    #                       Specify the way to construct DataLoaer, while allocw mutlti cpu for one trial, add this paramater. Default: False.
    #                       """ ).strip())
    
    optional.add_argument('--segment_center', type=int, metavar='INT', default=10000,
                          help=textwrap.dedent("""
                          The maximum encoding unit of the sequence. It affects trade-off 
                          between RAM memory and preprocessing speed. It is recommended to use 300k.
                          Default: 300000.""" ).strip())

    optional.add_argument('--pred_batch_size', type=int, metavar='INT', default=16, 
                          help=textwrap.dedent("""
                          Size of mini batches for prediction. Default: 16.
                          """ ).strip())
    
    optional.add_argument('--kmer_corr', type=int, metavar='INT', default=[], nargs='+',
                          help=textwrap.dedent("""
                          Calculate k-mer correlations with observed variants in 5th column.
                          Accept one or more odd positive integers for k-mers, e.g., "3 5 7".
                          Default: no value.
                          """ ).strip())
    
    optional.add_argument('--region_corr', type=int, metavar='INT', default=[], nargs='+',
                          help=textwrap.dedent("""
                          Calculate region correlations with observed variants in 5th column.
                          Accept one or more positive integers for window size (bp), 
                          e.g., "10000 50000". Default: no value.
                          """ ).strip())
    optional.add_argument('--use_dilation', 
                            default=False, 
                            action='store_true',  
                            help=textwrap.dedent("""Add this parameter if dilation is used in the model.""" ))
    
    optional.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(__version__))

    optional.add_argument('--feature_config', default=None, type=str, help=textwrap.dedent("""
                          Path to the JSON file containing feature configuration.
                          Default: None
                          """).strip()
                          )

    optional.add_argument('--recurrent', default=False, action='store_true',
                        help='Use per-site sample weights from BED name field for evaluation. Default: False.')

    Bayes_args.add_argument('--use_bayesian', default=False, action='store_true',
                          help=textwrap.dedent("""
                          Use Bayesian model. Default: False.
                          """ ).strip())
    Bayes_args.add_argument("--bnn_config", default=None, type=str, help=textwrap.dedent("""
                          Path to the JSON file containing Bayesian model configuration. Default: None
                          """).strip()
                          )
    
    parser._action_groups.append(optional)
    
    if len(sys.argv) == 1:
        parser.parse_args(['--help'])
    else:
        args = parser.parse_args()

    return args

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="""
    Overview
    -------- 
    This tool uses a trained MuRaL model to do prediction for the sites in the 
    input BED file.
    
    * Input data 
    Required input files for prediction include the reference FASTA file, 
    a BED-formatted data file and a trained model. The BED file is organized 
    in the same way as that for training. The 5th column can be set to '0' 
    if no observed mutations for the sites in the prediction BED. The 
    model-related files for input are 'model' and 'model.config.pkl', which 
    are generated at the training step. The file 'model.fdiri_cal.pkl', which 
    is for calibrating predicted mutation rates, is optional. If the input BED
    file has many sites (e.g. many millions), it is recommended to split it
    into smaller files (e.g. 1 million each) for parallel processing.
   
    * Output data 
    The output of `mural_predict` is a tab-separated file containing the 
    sequence coordinates and the predicted probabilities for all possible 
    mutation types. Usually, the 'prob0' column stores probabilities for the 
    non-mutated class and other 'probX' columns for mutated classes. 
   
    Some example lines of a prediction output file are shown below:
    chrom   start   end    strand mut_type  prob0   prob1   prob2   prob3
    chr1    10006   10007   -       0       0.9797  0.003134 0.01444 0.002724
    chr1    10007   10008   +       0       0.9849  0.005517 0.00707 0.002520
    chr1    10008   10009   +       0       0.9817  0.004801 0.01006 0.003399
    chr1    10012   10013   -       0       0.9711  0.004898 0.02029 0.003746

    Command line examples
    ---------------------
    1. The following command will predict mutation rates for all sites in 
    'testing.bed.gz' using model files under the 'checkpoint_6/' folder 
    and save prediction results into 'testing.ckpt6.fdiri.tsv.gz'. For most
    models, as prediction tasks usually won't take long, it is recommended to 
    set '--cpu_only' for using only CPUs and not generating HDF5 files.
    If the input BED file has many sites (e.g. many millions), it is recommended 
    to spilt it into smaller files (e.g. 1 million each) for parallel processing.
    
        mural_predict --ref_genome seq.fa --test_data testing.bed.gz \\
        --model_path checkpoint_6/model \\
        --model_config_path checkpoint_6/model.config.pkl \\
        --calibrator_path checkpoint_6/model.fdiri_cal.pkl \\
        --pred_file testing.ckpt6.fdiri.tsv.gz \\
        --cpu_only \\
        > test.out 2> test.err
    """) 
    
    args = parse_arguments(parser)

    use_dilation = args.use_dilation
    set_torch_backends(use_dilation)

    pred_batch_size = args.pred_batch_size
    sampled_segments = 1
    # Output file path
    pred_file = args.pred_file
    cpu_only = args.cpu_only

    # Get saved model-related files
    model_path = args.model_path
    model_config_path = args.model_config_path
    calibrator_path = args.calibrator_path
    
    kmer_corr = args.kmer_corr
    region_corr = args.region_corr

    # Load model config (hyperparameters)
    if model_config_path != '':
        with open(model_config_path, 'rb') as fconfig:
            config = pickle.load(fconfig)
    else:
        print('Error: no model config file provided!')
        sys.exit()



    # Set hyperparameters
    if not config.get('distal_order'):
        config['distal_order'] = args.distal_order
    if 'without_bw_distal' not in config:
        config['without_bw_distal'] = False
    without_bw_distal = False 
    n_class = config['n_class']
    print("without_bw_distal: ", without_bw_distal)

    if args.use_bayesian:
        # dependency bayesian modules
        from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
        # compatibility check and read config
        assert int(config['model_no']) == 127 , "Only model_no 127 (MuRaL_Hybrid) is supported for Bayesian training currently."
        const_bnn_prior_parameters = read_bnn_config(args.bnn_config)
    
    # set segment_center   
    if not args.segment_center:
        args.segment_center = segment_center = config['segment_center']
    else:
        segment_center = args.segment_center
    
    # Print command line
    cuda_id = args.cuda_id
    print(' '.join(sys.argv))
    for k,v in vars(args).items():
        print("{0}: {1}".format(k,v))
   
    out_each_model_preds = args.save_each_model_preds
    
    start_time = time.time()
    print('Start time:', datetime.datetime.now())
    sys.stdout.flush()
    
    mix_loss = config.get('mix_loss')
    
    preprocess_config = {
        'segment_center': segment_center,
        'local_radius' : config['local_radius'],
        'local_order' : config['local_order'],
        'distal_radius' : config['distal_radius'],
        'distal_order' : config['distal_order'],
        'h5f_path' : args.h5f_path,
        'seq_only' : config['seq_only'],
        'n_h5_files' : args.n_h5_files,
        'without_bw_distal' : without_bw_distal,
        'bw_paths' : args.bw_paths,
    }

    for k,v in preprocess_config.items():
        print("{0}: {1}".format(k,v))

    feature_config = read_feature_config(args.feature_config)
    use_segment_task = True if len(feature_config['features']) > 2 else False
    print("use_segment_task:", use_segment_task)

    preprocess_config.update(feature_config)


    calc_loss_strategy_name = "AvgStepMutAndKmerMutUseInLocal" if args.calc_loss_strategy_name is None else args.calc_loss_strategy_name
    print("calc_loss_strategy_name:", calc_loss_strategy_name)
    preprocessor_pipline = DatasetPreprocessor(preprocess_config, use_h5=args.with_h5)
    dataset_test = preprocessor_pipline.preprocess_dataset(args.test_data, args.ref_genome, use_segment_task=use_segment_task)
    segmentLoader_test = DataLoader(dataset_test, 1, shuffle=False, pin_memory=False,  collate_fn=dict_to_tuple_collate)
    dataloader= generate_data_batches(segmentLoader_test, sampled_segments, pred_batch_size, shuffle=False, use_segment_task=use_segment_task)

    data_local_test = dataset_test.data_local.reset_index(drop=True)
    sys.stdout.flush()
    
    if cpu_only:
        device = torch.device('cpu')
    else:
        if cuda_id == None:
            cuda_id = get_available_gpu(1)
        else:
            check_cuda_id(cuda_id)
        print('CUDA: ', torch.cuda.is_available())
        if torch.cuda.is_available():
            print('using'  , 'cuda:'+cuda_id)
        device = torch.device('cuda:'+cuda_id if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(f'cuda:{cuda_id}')


    # model config
    # 2025.12.14, 当前该参数在model_factory中直接定义，后续考虑优化为自动生成或在config文件中定义
    model_factory = ModelFactory(config, args)
    model = model_factory.create_model(config['model_no'])

    model_state = torch.load(model_path, map_location=device)




    # Loss function
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    loss_calculator = LossCalcStrategyFactory.get_loss_strategy(calc_loss_strategy_name, avg_mut_loss_strategy=mix_loss)

    # Set prob names for mutation types

    Observer = [TimeMinor(out_after_n_batch=1000), 
                LossMinor(calc_loss_strategy_name, printer=print)]

    if args.use_bayesian:
        config.update(const_bnn_prior_parameters)
        dnn_to_bnn(model, const_bnn_prior_parameters)
        # Load the saved model object
        model.load_state_dict(model_state)
        model.to(device)
        predictor = BayesianPredictor(model, loss_calculator, criterion, device, config, 
                      observer=Observer, printer=print, train_strategy=calc_loss_strategy_name)
    else:
        model.load_state_dict(model_state)
        model.to(device)
        predictor = Predictor(model, loss_calculator, criterion, device, config, 
                      observer=Observer, printer=print, train_strategy=calc_loss_strategy_name)

    print('model:')
    print(model)
    del model_state
    torch.cuda.empty_cache() 
    

    prob_names = ['prob'+str(i) for i in range(n_class)]

    if out_each_model_preds:
        assert args.use_bayesian == False, "args.use_bayesian must be True when out_each_model_preds is False"
        pred_dict = predictor.predict_each_model(dataloader)
        out_each_model_preds = [
            pd.DataFrame(
                data=to_np(preds),
                columns=[f"{name}_{prob_name}" for prob_name in prob_names]
                )
                for name, preds in pred_dict.items() if name != 'out'
                ]
        out_each_model_preds = pd.concat(out_each_model_preds, axis=1)
        chr_pos_ = get_position_info(BedTool(args.test_data), segment_center)
        chr_pos_.columns = ['chrom', 'start', 'end', 'strand']
        mut_type = data_local_test['mut_type']
        out_each_model_preds = pd.concat([chr_pos_, mut_type, out_each_model_preds], axis=1)
        out_each_model_preds.sort_values(['chrom', 'start'], inplace=True)
        out_each_model_preds.reset_index(drop=True, inplace=True)
        pred_file_each_model = pred_file.split('.bed.tsv.gz')[0] + '_each_model.tsv.gz'
        out_each_model_preds.to_csv(pred_file_each_model, sep='\t', float_format='%.4g', index=False)

        pred_y = pred_dict['out']
        y_prob = pd.DataFrame(data=to_np(F.softmax(pred_y, dim=1)), columns=prob_names)

        dfs = [data_local_test, y_prob]
    else:
        if args.use_bayesian:
            pred_y, pred_y_std = predictor.predict(dataloader)
            y_prob = pd.DataFrame(data=to_np(pred_y), columns=prob_names)
            prob_std_names = ['prob_std' + str(i) for i in range(n_class)]
            pred_y_std = pd.DataFrame(data=to_np(pred_y_std), columns=prob_std_names)
            all_prob_names = prob_names + prob_std_names
            dfs = [data_local_test, y_prob, pred_y_std]

            print('pred_y:', pred_y[1:10])
            print('pred_y_std:', pred_y_std[1:10])
        else:
            pred_y = predictor.predict(dataloader)
            print('pred_y:', F.softmax(pred_y[1:10], dim=1))
            y_prob = pd.DataFrame(data=to_np(F.softmax(pred_y, dim=1)), columns=prob_names)
            all_prob_names = prob_names
            dfs = [data_local_test, y_prob]
            # Print some data for debugging
            for i in range(1, n_class):
                print('min and max of pred_y: type', i, np.min(to_np(F.softmax(pred_y, dim=1))[:,i]), np.max(to_np(F.softmax(pred_y, dim=1))[:,i]))
        
    # Get the predicted probabilities, as the returns of model are logits    
    
    # Do probability calibration using saved calibrator
    if calibrator_path != '':
        with open(calibrator_path, 'rb') as fcal:   
            print('using calibrator for scaling ...')
            calibr = pickle.load(fcal)         
            prob_cal = calibr.predict_proba(y_prob.to_numpy())  
            y_prob = pd.DataFrame(data=np.copy(prob_cal), columns=prob_names)

    # Combine data
    use_obs_count = args.recurrent

    data_and_prob = pd.concat(dfs, axis=1)

    # Write the prediction
    chr_pos = get_position_info(BedTool(args.test_data), segment_center)
    # 验证mut_type一致性
    assert (chr_pos['mut_type'].astype(int).values ==
        data_and_prob['mut_type'].astype(int).values).all(), \
            'ERROR: mut_type mismatch between position info and prediction data. ' \
                'BED file or data pipeline may have inconsistent ordering.'
    pred_df = pd.concat((chr_pos, y_prob), axis=1)
    pred_df.sort_values(['chrom', 'start'], inplace=True)
    pred_df.reset_index(drop=True, inplace=True)
    # 输出文件包含info列
    pred_df.to_csv(pred_file, sep='\t', float_format='%.4g', index=False)
    
    #do k-mer evaluation
    if len(kmer_corr) > 0:
        modes = [i%2 for i in kmer_corr]

        if sum(modes) != len(kmer_corr) or min(kmer_corr) < 0:
            print('Warning: please provide odd positive mumbers for k-mer lengths', kmer_corr, '. No k-mer correlation was calculated.')
        else:
            for kmer in kmer_corr:
                print(str(kmer)+'mer correlation: ', freq_kmer_comp_multi(data_and_prob, kmer, n_class, use_obs_count))
   
    # Calculate regional correlations for a few window sizes
    #for win_size in [10000, 50000, 200000]:
    if len(region_corr) > 0:
        if min(region_corr) <=0:
            print('Warning: please provide  positive mumbers for window sizes. No regional correlation was calculated.')
        else:      
            pred_df.sort_values(['chrom', 'start'], inplace=True)
            
            for win_size in region_corr:
                corr = corr_calc_sub(pred_df, win_size, prob_names, use_obs_count)
                print('regional corr:', str(win_size)+'bp', corr)

    print('Total time used: %s seconds' % (time.time() - start_time))
    
if __name__ == "__main__":
    main()
