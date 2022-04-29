'''
Entire training pipeline logic.
'''

from __init__ import *
import torch.nn.functional as F

# Internal imports.
import msynth.utils.daw_synth as daw_synth
import loss_single as loss
import utils
from msynth.models.fmsynth import FMNet,Synth
import pdb

class MyTrainPipeline(torch.nn.Module):
    '''
    Wrapper around most of the training iteration such that DataParallel can be leveraged.
    '''

    def __init__(self, train_args, logger, networks, optimizer, device):
        super().__init__()
        self.train_args = train_args
        self.logger = logger
        self.networks = torch.nn.ModuleList(networks)
        self.analysis_net = networks[0]
        self.device = device
        self.phase = None  # Assigned only by set_phase().
        self.losses = None  # Instantiated only by set_phase().
        self.optimizer = optimizer

    def set_phase(self, phase):
        self.phase = phase
        self.losses = loss.MyLosses(self.train_args, self.logger, phase)

        if phase == 'train':
            self.train()
            for net in self.networks:
                if net is not None:
                    net.train()
            torch.set_grad_enabled(True)

        else:
            self.eval()
            for net in self.networks:
                if net is not None:
                    net.eval()
            torch.set_grad_enabled(False)

    def get_prepared_values(self,analysis_output):
        carrier_weight = analysis_output[:127]
        mod_weight = analysis_output[127:127*2]
        carrier_slope = analysis_output[127*2:127*3]
        carrier_offset = analysis_output[127*2:127*4]
        mod_slope = analysis_output[127*4:127*5]
        mod_offset = analysis_output[127*5:127*6]
        phase_offset = analysis_output[127*6:127*6+1]
        test = analysis_output[126*7+1:]
        self.min_slope=-2.0
        self.max_slope=8.0


        carrier_env_slope = 2.0 ** carrier_slope * (self.max_slope - self.min_slope)+ self.min_slope
        mod_env_slope = 2.0 ** mod_slope * (self.max_slope - self.min_slope)+ self.min_slope
        carrier_env_offset = carrier_offset / 2
        mod_env_offset = mod_offset / 2
        phase_offset = 2 * np.pi * phase_offset
        analysis_o = torch.cat([carrier_weight, mod_weight,carrier_env_slope, carrier_env_offset,mod_env_slope,mod_env_offset,phase_offset,test])
        return analysis_o

    

    def forward(self, data_retval, cur_step, total_step):
        '''
        Handles one parallel iteration of the training or validation phase.
        Executes the models and calculates the per-example losses.
        This is all done in a parallelized manner to minimize unnecessary communication.
        :param data_retval (dict): Data loader elements.
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :return (model_retval, loss_retval)
            model_retval (dict): All output information.
            loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        '''
        soundfont_path = self.train_args.soundfont_path
        sample_rate = self.train_args.sample_rate
        anadaw_interval = self.train_args.anadaw_interval

        # Prepare data.
        control_matrix = data_retval['control_matrix']
        daw_waveform = data_retval['daw_waveform']
        daw_melspec = data_retval['mel_spec']
        control_matrix = control_matrix.to(self.device)
        daw_waveform = daw_waveform.to(self.device)
        (B, P, T) = control_matrix.shape
        (B, S) = daw_waveform.shape
        

        # Run model to perform one-step processing for direct supervision.
        (analysis_output, _) = self.analysis_net(daw_melspec)


        assert analysis_output.shape == control_matrix.shape
    
        # Optional cycle supervision: perform analysis from predicted waveform, and also perform
        # synthesis from predicted control matrix with both neural proxy and DAW.

        # TODO: how to make this differentiable? use custom step function!
        # control_matrix_output = (analysis_output > 0.0).type(torch.float32)
        # TODO: sigmoid is out of domain!
        #control_matrix_output = torch.sigmoid(analysis_o)

        # Run model again to perform two-step processing.
        """(anasyn_cycle, _) = self.synthesis_net(control_matrix_output)
        (synana_cycle, _) = self.analysis_net(synthesis_output)
        assert anasyn_cycle.shape == daw_waveform.shape
        assert synana_cycle.shape == control_matrix.shape"""

        model_retval = dict()
        model_retval['analysis_output'] = torch.nn.functional.sigmoid(analysis_output)  # (B, P, T).

        loss_retval = self.losses.per_example(data_retval, model_retval)

        return (model_retval, loss_retval)

    def process_entire_batch(self, data_retval, model_retval, loss_retval, cur_step, total_step):
        '''
        Finalizes the training step. Calculates all losses.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :param cur_step (int): Current data loader index.
        :param total_step (int): Cumulative data loader index, including all previous epochs.
        :return loss_retval (dict): All loss information.
        '''
        loss_retval = self.losses.entire_batch(data_retval, model_retval, loss_retval)

        return loss_retval