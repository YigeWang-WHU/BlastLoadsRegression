import os
import argparse
from utils.utils import may_mkdir
from utils.utils import str2bool, time_str
class Config(object):
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--set_seed', type=str2bool, default=False)
        parser.add_argument('-d', '--sys_device_ids', type=eval, default=())
        ## dataset parameter
        parser.add_argument('--raw_data', type=str, default='')
        parser.add_argument('--data_root', type=str, default='')
        parser.add_argument('--dataset', type=str, default='Beyer1986v1',
                choices=['Beyer1986v1', 'Beyer1986v2', 'Beyer1986v3', 'data_FF', 'values_FAB', 'data_tot'])
        parser.add_argument('--batch_size', type=int, default=4)
        parser.add_argument('--workers', type=int, default=2)
        parser.add_argument('--portion_train', type=float, default=0.7)
        parser.add_argument('--portion_val', type=float, default=0.15)
        # model
        parser.add_argument('--num_input', type=int, default=3)
        parser.add_argument('--num_output', type=int, default=2)
        parser.add_argument('--num_h', type=int, default=1)
        parser.add_argument('--sgd_weight_decay', type=float, default=0.0005)
        parser.add_argument('--sgd_momentum', type=float, default=0.9)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--neurons_per_hlayer', type=int, nargs='+',
                            default=[3,8])
        parser.add_argument('--epochs_per_decay', type=int, nargs='+',
                            default=[10, 15])
        parser.add_argument('--staircase_decay_multiple_factor', type=float,
                            default=0.1)
        parser.add_argument('--total_epochs', type=int, default=10)
        # utils
        parser.add_argument('--ckpt_file', type=str, default='')
        parser.add_argument('--load_model_weight', type=str2bool, default=False)
        parser.add_argument('--test_only', type=str2bool, default=False)
        parser.add_argument('--exp_dir', type=str, default='')
        parser.add_argument('--log_to_file', type=str2bool, default=True)
        parser.add_argument('--steps_per_log', type=int, default=2)
        parser.add_argument('--epochs_per_val', type=int, default=10)
        parser.add_argument('--epochs_per_save', type=int, default=50)
        parser.add_argument('--run', type=int, default=1)
        args = parser.parse_args()

        self.sys_device_ids = args.sys_device_ids
        # random
        self.set_seed = args.set_seed
        if self.set_seed:
            self.rand_seed = 0
        else:
            self.rand_seed = None
        # run time index
        self.run = args.run
        # Dataset #
        self.data_root =args.data_root
        self.dataset_name = args.dataset
        self.raw_data = args.raw_data
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.input_size = args.num_input
        self.output_size = args.num_output
        self.portion_train = args.portion_train
        self.portion_val = args.portion_val
        # optimization
        self.sgd_momentum = args.sgd_momentum
        self.sgd_weight_decay = args.sgd_weight_decay
        self.lr = args.lr
        self.epochs_per_decay = args.epochs_per_decay
        self.staircase_decay_multiple_factor = args.staircase_decay_multiple_factor
        self.total_epochs = args.total_epochs

        # utils
        self.ckpt_file = args.ckpt_file
        self.load_model_weight = args.load_model_weight
        if self.load_model_weight:
            if self.ckpt_file == '':
                print ('Please input the ckpt_file if you want to resume training')
                raise ValueError
        self.test_only = args.test_only
        self.exp_dir = args.exp_dir
        self.log_to_file = args.log_to_file
        self.steps_per_log = args.steps_per_log
        self.epochs_per_val = args.epochs_per_val
        self.epochs_per_save = args.epochs_per_save
        self.run = args.run

        # for model
        model_kwargs = dict()
        model_kwargs['input_size'] = self.input_size
        model_kwargs['output_size'] = self.output_size
        model_kwargs['num_h'] = args.num_h
        model_kwargs['hidden_size'] = args.neurons_per_hlayer
        self.model_kwargs = model_kwargs
        # for evaluation

        if self.exp_dir == '':
            self.exp_dir = os.path.join('exp',
                '{}'.format(self.dataset_name),
                'run{}'.format(self.run))
        self.stdout_file = os.path.join(self.exp_dir, \
            'log', 'stdout_{}.txt'.format(time_str()))
        self.stderr_file = os.path.join(self.exp_dir, \
            'log', 'stderr_{}.txt'.format(time_str()))
        may_mkdir(self.stdout_file)

