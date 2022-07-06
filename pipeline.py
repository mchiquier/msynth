'''
Entire training pipeline logic.
'''

from __init__ import *


# Internal imports.
import daw_synth
import loss
import utils


class MyTrainPipeline(torch.nn.Module):
    '''
    Wrapper around most of the training iteration such that DataParallel can be leveraged.
    '''

    def __init__(self, train_args, logger, networks, device):
        super().__init__()
        self.train_args = train_args
        self.logger = logger
        self.networks = torch.nn.ModuleList(networks)
        self.analysis_net = networks[0]
        self.synthesis_net = networks[1]
        self.device = device
        self.phase = None  # Assigned only by set_phase().
        self.losses = None  # Instantiated only by set_phase().

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
        timecell_ms = self.train_args.timecell_ms
        anadaw_interval = self.train_args.anadaw_interval

        # Prepare data.
        control_matrix = data_retval['control_matrix']
        daw_waveform = data_retval['daw_waveform']
        control_matrix = control_matrix.to(self.device)
        daw_waveform = daw_waveform.to(self.device)
        (B, P, T) = control_matrix.shape
        (B, S) = daw_waveform.shape

        # Run model to perform one-step processing for direct supervision.
        (analysis_output, _) = self.analysis_net(daw_waveform)
        (synthesis_output, _) = self.synthesis_net(control_matrix)
        assert analysis_output.shape == control_matrix.shape
        assert synthesis_output.shape == daw_waveform.shape

        # Optional cycle supervision: perform analysis from predicted waveform, and also perform
        # synthesis from predicted control matrix with both neural proxy and DAW.

        # TODO: how to make this differentiable? use custom step function!
        # control_matrix_output = (analysis_output > 0.0).type(torch.float32)
        # TODO: sigmoid is out of domain!
        control_matrix_output = torch.sigmoid(analysis_output)

        # Run model again to perform two-step processing.
        (anasyn_cycle, _) = self.synthesis_net(control_matrix_output)
        (synana_cycle, _) = self.analysis_net(synthesis_output)
        assert anasyn_cycle.shape == daw_waveform.shape
        assert synana_cycle.shape == control_matrix.shape

        all_anadaw_cycle = []

        # Using the DAW is quite expensive, so only do it once in a while.
        if cur_step % anadaw_interval == 0:

            for i in range(B):

                # Binarize analysis output (= logits, not probits) into control matrix for DAW.
                cur_analysis_output_numpy = analysis_output[i].detach().cpu().numpy()
                tp_matrix_output = (cur_analysis_output_numpy > 0.0).astype(np.float32)
                
                # Synthesize for cycle DAW reconstruction.
                event_seq = daw_synth.tp_matrix_to_event_seq(
                    tp_matrix_output, timecell_ms)
                anadaw_cycle = daw_synth.event_seq_to_waveform(
                    soundfont_path, event_seq, sample_rate, True, S)
                
                # Update lists to keep.
                anadaw_cycle = torch.tensor(anadaw_cycle, device=daw_waveform.device)
                all_anadaw_cycle.append(anadaw_cycle)

            anadaw_cycle = torch.stack(all_anadaw_cycle)
            assert anadaw_cycle.shape == daw_waveform.shape
            has_anadaw_cycle = torch.tensor(True, device=daw_waveform.device)

        else:

            anadaw_cycle = torch.zeros_like(anasyn_cycle)
            has_anadaw_cycle = torch.tensor(False, device=daw_waveform.device)

        model_retval = dict()
        model_retval['analysis_output'] = analysis_output  # (B, P, T).
        model_retval['synthesis_output'] = synthesis_output  # (B, S).
        model_retval['anasyn_cycle'] = anasyn_cycle  # (B, S).
        model_retval['synana_cycle'] = synana_cycle  # (B, P, T).
        model_retval['anadaw_cycle'] = anadaw_cycle  # (B, S).
        model_retval['has_anadaw_cycle'] = has_anadaw_cycle  # (B).

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
