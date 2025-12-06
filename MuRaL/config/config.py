

calc_strategies = [
    'AvgSegMutUseInLocal', 
    'AvgSegmentLabel_withGAN', 
    'AvgSegmentLabel_withGAN2',
    'AvgSegAndKmerMut',
]

segment_calc_method = ['SegMutRate', 'SegMut', 'AvgSegMutAndKmerMut', None]
soft_label_dict_keys = ['segment_id', 'segment_avg_mut', 'segment_avg_kmer_mut']
segment_task = [True, False]
return_strategy = [SegmentTaskReturnStrategy, SegmentMutFreqReturnStrategy, SegmentAvgAndKmerReturnStrategy, None]
strategy = ['AvgSegMutAndNucSkewUseInLocal']


if calc_strategy in ['AvgSegmentLabel_withGAN', 'AvgSegmentLabel_withGAN2']:
    method = 'SegMut'
elif calc_strategy == 'AvgSegMutUseInLocal':
    method = 'SegMutRate'
elif calc_strategy == 'AvgSegAndKmerMut':
    method = 'AvgSegMutAndKmerMut'
else:
    method = None


"""
calc_strategies
            ######## data preparation ########
     |----> segment_calc_method
        |----> prepare_soft_label( or segment task; How to get information from the segment)
        |----> CombinedDatasetNPv2 
                |----> return_strategy(How to return the data used to train the model)
                        return_strategy = 
                        [SegmentTaskReturnStrategy, 
                        SegmentMutFreqReturnStrategy, 
                        SegmentAvgAndKmerReturnStrategy, 
                        None]

            ######## model training ########
    |----> create_model(How to create the model)
                |----> combined segment Task model
                
                |----> parser data used training(choice parser strategy to adapt model)
                        |---> part1: parser data from data loader
                        |---> part2: classify data to training data and label
                        |---> part3: input training data to model
                        |---> part4: input label to loss_calculator

                |----> loss_calculator(How to calculate the loss)
                        loss_calc_startegy_name = []
                
                |----> optimizer(How to optimize the model)

                |----> Observer(which minor shoud be record in the training process)
                        observer_name = []

            ######## model validation ########



                                              
"""

