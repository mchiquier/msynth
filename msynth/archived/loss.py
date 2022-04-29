
from __init__ import *
import pdb

class MyLosses():
    '''
    Wrapper around the loss functionality such that DataParallel can be leveraged.
    '''

    def __init__(self, train_args, logger, phase):
        self.train_args = train_args
        self.logger = logger
        self.phase = phase
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.l1_loss = torch.nn.L1Loss(reduction='mean')
        self.l2_loss = torch.nn.MSELoss(reduction='mean')

    def my_analysis_loss(self, control_matrix_output, control_matrix_target):
        '''
        :param control_matrix_output (B, C, T) tensor.
        :param control_matrix_target (B, C, T) tensor.
        :return loss_analysis (tensor).
        '''
        loss_analysis = self.l2_loss(control_matrix_output, control_matrix_target)
        #pdb.set_trace()
        return loss_analysis

    def my_synthesis_loss(self, waveform_output, waveform_target):
        '''
        :param waveform_output (B, S) tensor.
        :param waveform_target (B, S) tensor.
        :return loss_synthesis (tensor).
        '''

        if self.train_args.waveform_loss_mode == 'samples':

            loss_synthesis = self.l2_loss(waveform_output, waveform_target)
            loss_synthesis *= 16.0  # Because waveform values are relatively tiny.

        else:

            n_fft = self.train_args.n_fft_freq
            hop_length = self.train_args.n_fft_freq // 4
            stft_output = torch.stft(
                waveform_output, n_fft=n_fft, hop_length=hop_length, return_complex=True)
            stft_target = torch.stft(
                waveform_target, n_fft=n_fft, hop_length=hop_length, return_complex=True)

            # Force removal of highest frequency components to avoid glitches.
            stft_target[:, -self.train_args.n_fft_freq // 32:] = 0.0

            if self.train_args.waveform_loss_mode == 'stft_real_imag':

                stft_real_output = stft_output.real
                stft_real_target = stft_target.real
                stft_imag_output = stft_output.imag
                stft_imag_target = stft_target.imag

                loss_synthesis = self.l2_loss(stft_real_output, stft_real_target) + \
                    self.l2_loss(stft_imag_output, stft_imag_target)
                loss_synthesis /= 16.0  # Because STFT values are relatively large.

            elif self.train_args.waveform_loss_mode == 'stft_mag':

                stft_mag_output = stft_output.abs()
                stft_mag_target = stft_target.abs()

                loss_synthesis = self.l2_loss(stft_mag_output, stft_mag_target)
                loss_synthesis /= 16.0  # Because STFT values are relatively large.

        return loss_synthesis

    def per_example(self, data_retval, model_retval):
        '''
        Loss calculations that *can* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :return loss_retval (dict): Preliminary loss information.
        '''
        (B, C, T) = data_retval['control_matrix'].shape
        (B, S) = data_retval['daw_waveform'].shape

        loss_analysis = []
        loss_synthesis = []

        # Loop over every example.
        for i in range(B):

            control_matrix_target = data_retval['control_matrix'][i:i + 1]
            waveform_target = data_retval['daw_waveform'][i:i + 1]
            control_matrix_output = model_retval['analysis_output'][i:i + 1]
            waveform_output = model_retval['synthesis_output'][i:i + 1]

            # Calculate loss terms.
            cur_analysis = self.my_analysis_loss(control_matrix_output, control_matrix_target)
            cur_synthesis = self.my_synthesis_loss(waveform_output, waveform_target)

            # Update lists.
            if self.train_args.analysis_lw > 0.0:
                loss_analysis.append(cur_analysis)
            if self.train_args.synthesis_lw > 0.0:
                loss_synthesis.append(cur_synthesis)

        # Sum & return losses + other informative metrics across batch size within this GPU.
        # Prefer sum over mean to be consistent with dataset size.
        loss_analysis = torch.sum(torch.stack(loss_analysis)) if len(loss_analysis) else None
        loss_synthesis = torch.sum(torch.stack(loss_synthesis)) if len(loss_synthesis) else None

        # Return results.
        loss_retval = dict()
        loss_retval['analysis'] = loss_analysis
        loss_retval['synthesis'] = loss_synthesis
        return loss_retval

    def entire_batch(self, data_retval, model_retval, loss_retval):
        '''
        Loss calculations that *cannot* be performed independently for each example within a batch.
        :param data_retval (dict): Data loader elements.
        :param model_retval (dict): All network output information.
        :param loss_retval (dict): Preliminary loss information (per-example, but not batch-wide).
        :return loss_retval (dict): All loss information.
        '''

        # Sum all terms across batch size.
        # Prefer sum over mean to be consistent with dataset size.
        for (k, v) in loss_retval.items():
            if torch.is_tensor(v):
                loss_retval[k] = torch.sum(v)

        # Obtain total loss.
        loss_total =  loss_retval['synthesis'] * self.train_args.synthesis_lw #loss_retval['analysis'] * self.train_args.analysis_lw +l
            
        print(loss_total)
        # Convert loss terms (just not the total) to floats for logging.
        for (k, v) in loss_retval.items():
            if torch.is_tensor(v):
                loss_retval[k] = v.item()

        # Report all loss values.
        self.logger.report_scalar(
            self.phase + '/loss_total', loss_total.item(), remember=True)
        self.logger.report_scalar(
            self.phase + '/loss_analysis', loss_retval['analysis'], remember=True)
        self.logger.report_scalar(
            self.phase + '/loss_synthesis', loss_retval['synthesis'], remember=True)

        # Return results, i.e. append to the existing loss_retval dictionary.
        loss_retval['total'] = loss_total
        return loss_retval