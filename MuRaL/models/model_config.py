
import sys
from MuRaL.models.nn_models import Network0, Network1, Network2, Network3

def model_choice(model_no, config, emb_dims, distal_order, n_class, n_cont, in_channels):
    if isinstance(model_no, str):
        try:
            model_no = int(model_no)
        except ValueError:
            pass
    elif model_no == 151:
        sys.path.append('/public/home/songhui/project/Mural/Mural_repo/MuRaL_112/model_utils/')
        from model_fusion_arg import Network3_ARG_condition
        model_config = {
            'pooling_kind' : 'max',
            'embeding_avg_mutations' : False,
            'embeding_nuc_skew' : False,
            'no_of_nuc_skew' : 14, # feature map to local seq len
            'use_local_fc2': True,
            'use_local_fc3': True,
            'local_model_name': 'AverageMutationModel_add2DCNN',
            # add 2026.1.21
            'local_fc2_name': 'hidden_with_relu',
            'local_fc3_name': 'ConvModelDrop',
        }

        model_specify_config = {
            'fused_type' : 'logit',
            'n_arg_features' : 23,
            'arg_hidden_dim' : 128,
            'arg_out_dim' : 64,
            'arg_dropout' : [0.2, 0.1, 0.1]
        }

        model_config.update(model_specify_config)

        model = Network3_ARG_condition(emb_dims, no_of_cont=config['no_of_cont'], lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']],
                  emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']],
                  in_channels=4, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'],
                  distal_radius=config['distal_radius'], distal_order=distal_order,
                  distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'],
                  config=model_config)

    elif model_no == '151_nb':
        from MuRaL.models.nb_model import Network3_ARG_condition_NB

        model_config = {
            'pooling_kind' : 'max',
            'embeding_avg_mutations' : False,
            'embeding_nuc_skew' : False,
            'no_of_nuc_skew' : 14,
            'use_local_fc2': True,
            'use_local_fc3': True,
            'local_model_name': 'AverageMutationModel_add2DCNN',
            'local_fc2_name': 'hidden_with_relu',
            'local_fc3_name': 'ConvModelDrop',
        }

        model_specify_config = {
            'fused_type' : 'logit',
            'n_arg_features' : 23,
            'arg_hidden_dim' : 128,
            'arg_out_dim' : 64,
            'arg_dropout' : [0.2, 0.1, 0.1]
        }

        model_config.update(model_specify_config)

        model = Network3_ARG_condition_NB(emb_dims, no_of_cont=config['no_of_cont'], lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']],
                  emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']],
                  in_channels=4, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'],
                  distal_radius=config['distal_radius'], distal_order=distal_order,
                  distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'],
                  config=model_config)

    elif model_no == '151_nb_v2':
        from MuRaL.models.nb_model_v2 import Network3_ARG_condition_NBv2

        model_config = {
            'pooling_kind' : 'max',
            'embeding_avg_mutations' : False,
            'embeding_nuc_skew' : False,
            'no_of_nuc_skew' : 14,
            'use_local_fc2': True,
            'use_local_fc3': True,
            'local_model_name': 'AverageMutationModel_add2DCNN',
            'local_fc2_name': 'hidden_with_relu',
            'local_fc3_name': 'ConvModelDrop',
        }

        model_specify_config = {
            'fused_type' : 'logit',
            'n_arg_features' : 23,
            'arg_hidden_dim' : 128,
            'arg_out_dim' : 64,
            'arg_dropout' : [0.2, 0.1, 0.1]
        }

        model_config.update(model_specify_config)

        model = Network3_ARG_condition_NBv2(emb_dims, no_of_cont=config['no_of_cont'], lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']],
                  emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']],
                  in_channels=4, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'],
                  distal_radius=config['distal_radius'], distal_order=distal_order,
                  distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'],
                  config=model_config)

    elif model_no == '151_nb_v3':
        from MuRaL.models.nb_model_v3 import Network3_ARG_condition_NBv3

        model_config = {
            'pooling_kind' : 'max',
            'embeding_avg_mutations' : False,
            'embeding_nuc_skew' : False,
            'no_of_nuc_skew' : 14,
            'use_local_fc2': True,
            'use_local_fc3': True,
            'local_model_name': 'AverageMutationModel_add2DCNN',
            'local_fc2_name': 'hidden_with_relu',
            'local_fc3_name': 'ConvModelDrop',
        }

        model_specify_config = {
            'fused_type' : 'logit',
            'n_arg_features' : 23,
            'arg_hidden_dim' : 128,
            'arg_out_dim' : 64,
            'arg_dropout' : [0.2, 0.1, 0.1]
        }

        model_config.update(model_specify_config)

        model = Network3_ARG_condition_NBv3(emb_dims, no_of_cont=config['no_of_cont'], lin_layer_sizes=[config['local_hidden1_size'], config['local_hidden2_size']],
                  emb_dropout=config['emb_dropout'], lin_layer_dropouts=[config['local_dropout'], config['local_dropout']],
                  in_channels=4, out_channels=config['CNN_out_channels'], kernel_size=config['CNN_kernel_size'],
                  distal_radius=config['distal_radius'], distal_order=distal_order,
                  distal_fc_dropout=config['distal_fc_dropout'], n_class=n_class, emb_padding_idx=4**config['local_order'],
                  config=model_config)

    else:
        print('Error: no model selected!')
        sys.exit() 
    return model

class ModelFactory:
    def __init__(self, config, args) -> None:
        self.emb_dims  = config['emb_dims']
        self.emb_dropout = config['emb_dropout']
        self.n_cont = config['no_of_cont']
        self.lin_layer_sizes = [config['local_hidden1_size'], config['local_hidden2_size']]
        self.lin_layer_dropouts = [config['local_dropout'], config['local_dropout']]
        self.n_class = args.n_class if not config.get('n_class') else config['n_class']
        self.emb_padding_idx = 4 ** config['local_order']

        self.out_channels = config['CNN_out_channels']
        self.kernel_size = config['CNN_kernel_size']


        self.distal_order = args.distal_order if not config.get('distal_order') else config['distal_order']
        without_bw_distal = config.get('without_bw_distal')
        if without_bw_distal:
            self.in_channels = 4 ** self.distal_order
        else:
            self.in_channels = 4 ** self.distal_order + self.n_cont
        
        self.config = config
        
    def create_model(self, model_no):
        if isinstance(model_no, str):
            try:
                model_no = int(model_no)
            except ValueError:
                pass
        if model_no == 0:
            model = Network0(self.emb_dims, no_of_cont=self.n_cont, 
                             lin_layer_sizes=self.lin_layer_sizes, 
                             emb_dropout=self.emb_dropout, 
                             lin_layer_dropouts=self.lin_layer_dropouts, 
                             n_class=self.n_class, 
                             emb_padding_idx=self.emb_padding_idx)

        elif model_no == 1:
            model = Network1(in_channels=self.in_channels, 
                             out_channels=self.out_channels, 
                             kernel_size=self.kernel_size,  
                             distal_radius=self.config['distal_radius'], 
                             distal_order=self.distal_order, 
                             distal_fc_dropout=self.config['distal_fc_dropout'], 
                             n_class=self.n_class)
        elif model_no == 2:
            model = Network2(self.emb_dims, 
                             no_of_cont=self.n_cont, 
                             lin_layer_sizes=self.lin_layer_sizes,
                             emb_dropout=self.emb_dropout, 
                             lin_layer_dropouts=self.lin_layer_dropouts,
                             in_channels=self.in_channels, 
                             out_channels=self.out_channels, 
                             kernel_size=self.kernel_size, 
                             distal_radius=self.config['distal_radius'], 
                             distal_order=self.distal_order, 
                             distal_fc_dropout=self.config['distal_fc_dropout'], 
                             n_class=self.n_class, 
                             emb_padding_idx=self.emb_padding_idx)
        elif model_no == 3:
            model = Network3(self.emb_dims, 
                             no_of_cont=self.n_cont, 
                             lin_layer_sizes=self.lin_layer_sizes,
                             emb_dropout=self.emb_dropout,
                             lin_layer_dropouts=self.lin_layer_dropouts,
                             in_channels=self.in_channels, 
                             out_channels=self.out_channels,
                             kernel_size=self.kernel_size, 
                             distal_radius=self.config['distal_radius'], 
                             distal_order=self.distal_order, 
                             distal_fc_dropout=self.config['distal_fc_dropout'], 
                             n_class=self.n_class, 
                             emb_padding_idx=self.emb_padding_idx)
        else:
            model = model_choice(model_no, 
                                 self.config, 
                                 self.emb_dims, 
                                 self.distal_order, 
                                 self.n_class, 
                                 self.n_cont,
                                 self.in_channels)
        
        return model
