'''
Neural network architecture description.
Created by Basile Van Hoorick, Feb 2022.
'''

# Internal imports.
import msynth.models.perceiverbackbone as pb
import torch
import utils


class MySpecAnalysisModel(torch.nn.Module):
    '''
    Converts audio waveforms to control matrix representations.
    '''

    def __init__(self, logger, num_input_freq, num_input_timebin, num_controls, num_timecells, samples_per_frame,
                 output_pos_enc):
        '''
        :param num_samples (int) = S: Waveform length.
        :param num_controls (int) = C: Number of event control types (e.g. pitches for instruments).
        :param num_timecells (int) = T: Discretized duration of control matrix.
        '''
        super().__init__()
        self.logger = logger
        self.num_input_freq = num_input_freq
        self.num_input_timebin = num_input_timebin
        self.num_controls = num_controls
        self.num_timecells = num_timecells
        self.samples_per_frame = samples_per_frame

        self.model = pb.MyPerceiverBackbone(
            logger, (num_input_freq, num_input_timebin), (num_timecells,num_controls), samples_per_frame,
            output_pos_enc) 

    def forward(self, waveform):
        '''
        :param waveform (B, S) tensor.
        :return (control_matrix, control_matrix).
            control_matrix (B, C, T) tensor.
            last_hidden_state (B, L, D) tensor.
        '''

        (control_matrix, last_hidden_state) = self.model(waveform)

        return (control_matrix, last_hidden_state)

class MySynthesisModel(torch.nn.Module):
    '''
    Converts control matrix representations to audio waveforms.
    '''

    def __init__(self, logger, num_samples, num_controls, num_timecells, samples_per_frame,
                 output_pos_enc):
        '''
        :param num_samples (int) = S: Waveform length.
        :param num_controls (int) = C: Number of event control types (e.g. pitches for instruments).
        :param num_timecells (int) = T: Discretized duration of control matrix.
        '''
        super().__init__()
        self.logger = logger
        self.num_samples = num_samples
        self.num_controls = num_controls
        self.num_timecells = num_timecells

        self.model = pb.MyPerceiverBackbone(
            logger, (num_controls, num_timecells), (num_samples, ), samples_per_frame,
            output_pos_enc)

    def forward(self, control_matrix):
        '''
        :param control_matrix (B, C, T) tensor.
        :return (waveform, last_hidden_state).
            waveform (B, S) tensor.
            last_hidden_state (B, L, D) tensor.
        '''

        (waveform, last_hidden_state) = self.model(control_matrix)

        return (waveform, last_hidden_state)


if __name__ == '__main__':

    num_samples = 41000*6
    num_controls = 127*8+4
    num_timecells = 1 #1
    samples_per_frame = 60

    for output_pos_enc in ['fourier', 'trainable']:

        print()
        print('output_pos_enc:', output_pos_enc)

        """analysis = MyAnalysisModel(
            None, num_samples, num_controls, num_timecells, samples_per_frame, output_pos_enc)

        input_waveform = torch.randn(4, num_samples)
        (output_control_matrix, last_hidden_state) = analysis(input_waveform)

        print('input_waveform:', input_waveform.shape)
        print('output_control_matrix:', output_control_matrix.shape)
        print('last_hidden_state:', last_hidden_state.shape)
        print()"""

        synthesis = MySynthesisModel(
            None, num_samples, num_controls, num_timecells, samples_per_frame, output_pos_enc)

        input_control_matrix = torch.randn(4, num_controls, num_timecells)
        (output_waveform, last_hidden_state) = synthesis(input_control_matrix)

        print('input_control_matrix:', input_control_matrix.shape)
        print('output_waveform:', output_waveform.shape)
        print('last_hidden_state:', last_hidden_state.shape)
        print()

    pass

