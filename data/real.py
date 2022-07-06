'''
Real melody / music interfacing as a data source.
Created by Basile Van Hoorick, Mar 2022.
'''

from __init__ import *

# Internal imports.
import daw_synth
import utils


class MyMIDIFileDataset(torch.utils.data.Dataset):
    '''
    Dataset from existing MIDI files.
    '''

    def __init__(self, data_path, logger, phase, soundfont_path=None, sample_rate=24000,
                 duration_ms=4000, timecell_ms=50, pitch_bounds=[69 - 21, 69 + 15],
                 data_clip_mode='random', clip_interval_ms=2000):
        '''
        :param data_path (str): Dataset root or split path.
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

        if os.path.isfile(data_path):
            all_fps = [data_path]

        else:
            # TODO
            split_dp = os.path.join(data_path, phase)
            if not os.path.exists(split_dp):
                split_dp = data_path
            raise NotImplementedError()

        real_size = len(all_fps)
        all_tracks = []
        
        if data_clip_mode == 'random':
            # Reuse every track a fixed number of times.
            virtual_multiplier = 16
            virtual_size = real_size * virtual_multiplier

        elif data_clip_mode == 'sequential':
            # Cover every track with a certain amount of clips according to the specified interval.
            num_total_clips = 0
            
            for i in range(real_size):
                (track_event_seq, metadata) = daw_synth.read_midi_file(all_fps[i])
                all_tracks.append(track_event_seq)
                track_ms = int(metadata['length'] * 1000.0)
                cur_uses = int(np.ceil((track_ms - duration_ms) / clip_interval_ms))
                num_total_clips += cur_uses
            
            virtual_size = num_total_clips

        # Precisely fixed number to avoid model errors; trim or pad data with zeros whenever needed.
        desired_samples = int(duration_ms / 1000.0 * sample_rate)

        self.logger = logger
        self.phase = phase
        self.soundfont_path = soundfont_path
        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.timecell_ms = timecell_ms
        self.pitch_bounds = pitch_bounds
        self.data_clip_mode = data_clip_mode
        self.clip_interval_ms = clip_interval_ms
        self.all_fps = all_fps
        self.all_tracks = all_tracks
        self.real_size = real_size
        self.virtual_size = virtual_size
        self.desired_samples = desired_samples
        self.num_instruments = len(soundfont_path)

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, index):

        if self.data_clip_mode == 'random':
            file_idx = index % self.real_size
            src_fp = self.all_fps[file_idx]
            within_idx = -1
        
        elif self.data_clip_mode == 'sequential':
            # TODO
            file_idx = 0
            src_fp = self.all_fps[file_idx]
            within_idx = index

        # TODO: support multiple soundfonts -- this loop does nothing yet.
        for i in range(self.num_instruments):

            # Convert MIDI file => track event sequence => track control matrix
            #                   => clip control matrix  => clip event sequence.
            track_event_seq = self.all_tracks[file_idx]
            track_tp_matrix = daw_synth.event_seq_to_tp_matrix(
                track_event_seq, 'auto', self.timecell_ms)
        
            (P, T) = track_tp_matrix.shape
            desired_timecells = int(round(self.duration_ms / self.timecell_ms))

            if self.data_clip_mode == 'random':
                clip_start = 0
                clip_end = desired_timecells
                
                # Select random clip if song is too long.
                if T > desired_timecells:
                    clip_start = np.random.randint(T - desired_timecells)
                    clip_end = clip_start + desired_timecells
                    clip_tp_matrix = track_tp_matrix[:, clip_start:clip_end]

            elif self.data_clip_mode == 'sequential':
                clip_start = int(round(within_idx * self.clip_interval_ms / self.timecell_ms))
                clip_end = clip_start + desired_timecells
                clip_tp_matrix = track_tp_matrix[:, clip_start:clip_end]

            # Pad control matrix at the end if it is too short.
            (P, T) = clip_tp_matrix.shape
            if T < desired_timecells:
                clip_tp_matrix = np.pad(clip_tp_matrix, ((0, 0), (0, desired_timecells - T)))
                T = desired_timecells

            clip_event_seq = daw_synth.tp_matrix_to_event_seq(
                clip_tp_matrix, self.timecell_ms)
            daw_waveform = daw_synth.event_seq_to_waveform(
                self.soundfont_path[i], clip_event_seq,
                self.sample_rate, True, self.desired_samples)

        # TODO: This is where we combine multiple instruments and effects in the future.
        control_matrix = clip_tp_matrix

        # Return results.
        data_retval = dict()
        data_retval['index'] = index
        data_retval['file_idx'] = file_idx
        data_retval['src_fp'] = src_fp
        data_retval['within_idx'] = within_idx
        data_retval['control_matrix'] = control_matrix.astype(np.float32)
        # data_retval['event_seq'] = clip_event_seq
        data_retval['daw_waveform'] = daw_waveform.astype(np.float32)
        data_retval['clip_start'] = clip_start * self.timecell_ms / 1000.0
        data_retval['clip_end'] = clip_end * self.timecell_ms / 1000.0

        # TODO: find way to make event_seq across batch elements and return.

        return data_retval


class MyAudioFileDataset(torch.utils.data.Dataset):
    '''
    Dataset from existing audio (WAV) files, i.e. no MIDI information available.
    '''

    def __init__(self, data_path, logger, phase):
        '''
        X
        '''

        pass
