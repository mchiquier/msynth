
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
        print(control_matrix_output, control_matrix_target, control_matrix_target.shape)
        return loss_analysis

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
        loss_total =  loss_retval['analysis'] #* self.train_args.analysis_lw 
            
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

        # Return results, i.e. append to the existing loss_retval dictionary.
        loss_retval['total'] = loss_total
        return loss_retval

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

            # Calculate loss terms.
            cur_analysis = self.my_analysis_loss(control_matrix_output, control_matrix_target)

            # Update lists.
            if self.train_args.analysis_lw > 0.0:
                loss_analysis.append(cur_analysis)

        # Sum & return losses + other informative metrics across batch size within this GPU.
        # Prefer sum over mean to be consistent with dataset size.
        loss_analysis = torch.sum(torch.stack(loss_analysis)) if len(loss_analysis) else None

        # Return results.
        loss_retval = dict()
        loss_retval['analysis'] = loss_analysis
        return loss_retval

   