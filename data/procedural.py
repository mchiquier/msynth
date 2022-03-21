'''
Procedural / random melody / music generation as a data source.
Created by Basile Van Hoorick, Mar 2022.
'''

from __init__ import *

# Internal imports.
import daw_synth
import utils


def random_tp_matrix(duration_ms, timecell_ms, pitch_bounds):
    '''
    :param duration_ms (int): Total duration of the generated melody.
    :param timecell_ms (int): Duration of one timecell (i.e. element in the horizontal axis) [ms].
    :param pitch_bounds ((int, int) tuple): pitch_min and pitch_max.
    :return tp_matrix (P, T) numpy array: Timecell / frequency binary description of melody.
    '''
    P = 128
    chord_prob = 0.2

    T = duration_ms // timecell_ms
    tp_matrix = np.zeros((P, T), dtype=np.uint8)
    num_notes = np.random.randint(2, 13)
    existing_starts = set()

    for i in range(num_notes):
        # Add random simultaneous starts of multiple notes.
        found = False
        while not found:
            if i > 0 and np.random.rand() < chord_prob:
                start = np.random.choice(list(existing_starts))
                if start + 1 < T - 2:
                    found = True
                    end = np.random.randint(start + 1, T - 2)
            else:
                start, end = np.random.randint(1, T - 2, 2)
                found = True

        start, end = min(start, end), max(start, end) + 1
        pitch = np.random.randint(pitch_bounds[0], pitch_bounds[1] + 1)

        for t in range(start, end):
            tp_matrix[pitch, t] = 1

        existing_starts.add(start)

    return tp_matrix


class MyRandomMelodyDataset(torch.utils.data.Dataset):
    '''
    Dataset with procedurally generated melodies.
    '''

    def __init__(self, logger, phase, soundfont_path=None, sample_rate=24000, duration_ms=4000,
                 timecell_ms=50, pitch_bounds=[69 - 21, 69 + 15], data_clip_mode='random',
                 clip_interval_ms=2000):
        '''
        :param logger (MyLogger).
        :param phase (str): train / val_aug / val_noaug / test.
        :param soundfont_path (list of str): Path(s) to soundfont preset file(s).
        :param sample_rate (int): Sample rate (Fs) in Hz.
        :param duration_ms (int): Length of all clips (waveforms).
        :param timecell_ms (int): Control matrix time resolution.
        :param pitch_bounds (int, int): Min and max pitch respectively, as MIDI note indices
            (inclusive). Default is C3 to C6.
        :param data_clip_mode (str): When a track is longer than the desired duration, how to select
            subclips (random / sequential).
            NOTE: Unused for now.
        :param clip_interval_ms (int): If data_clip_mode is sequential, stride size across data
            loader elements returned subsequently.
            NOTE: Unused for now.
        '''

        if not isinstance(soundfont_path, list):
            soundfont_path = [soundfont_path]

        dset_size = 44000 if 'train' in phase else 4400

        # Precisely fixed number to avoid model errors; trim or pad data with zeros whenever needed.
        desired_samples = int(duration_ms / 1000.0 * sample_rate)

        self.logger = logger
        self.phase = phase
        self.soundfont_path = soundfont_path
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.timecell_ms = timecell_ms
        self.dset_size = dset_size
        self.pitch_bounds = pitch_bounds
        self.data_clip_mode = data_clip_mode
        self.clip_interval_ms = clip_interval_ms
        self.desired_samples = desired_samples
        self.num_instruments = len(soundfont_path)

    def __len__(self):
        return self.dset_size

    def __getitem__(self, index):

        # TODO: support multiple soundfonts -- this loop does nothing yet.
        for i in range(self.num_instruments):

            tp_matrix = procedural.random_tp_matrix(
                self.duration_ms, self.timecell_ms, self.pitch_bounds)
            event_seq = daw_synth.tp_matrix_to_event_seq(
                tp_matrix, self.timecell_ms)
            daw_waveform = daw_synth.event_seq_to_waveform(
                self.soundfont_path[i], event_seq, self.sample_rate, True, self.desired_samples)

        # TODO: This is where we combine multiple instruments and effects in the future.
        control_matrix = tp_matrix

        # Return results.
        data_retval = dict()
        data_retval['control_matrix'] = control_matrix.astype(np.float32)
        # data_retval['event_seq'] = event_seq
        data_retval['daw_waveform'] = daw_waveform.astype(np.float32)

        # TODO: find way to make event_seq across batch elements and return.

        return data_retval
