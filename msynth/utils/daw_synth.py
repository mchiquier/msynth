'''
Utilities related to synthesis via DAW / existing libraries specifically.
Created by Basile Van Hoorick, Feb 2022.
'''

from __init__ import *

# Library imports.
import fluidsynth
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.signal
import mido
# Internal imports.
import pdb


def read_midi_file(midi_path, max_iters=-1):
    '''
    Converts a MIDI file to an event sequence.
    :param midi_path (str): Path to MIDI file.
    :return event_seq (list of (str, int) tuples): List of DAW commands encoded as performance events.
    '''
    event_seq = []
    cur_iters = 0
    
    # https://mido.readthedocs.io/en/latest/
    mid = mido.MidiFile(midi_path)
    for msg in mid:

        # NOTE: Take care to interpret silent note_on messages as note_off!
        if msg.type == 'note_on' and msg.velocity > 0:
            event_seq.append(('NOTE_ON', msg.note))
        
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            event_seq.append(('NOTE_OFF', msg.note))
        
        if msg.time > 0:
            # TODO: Check if this is always seconds, could be ticks or beats!
            event_seq.append(('TIME_SHIFT', int(msg.time * 1000.0)))

        cur_iters += 1
        if cur_iters <= 40:
            print(msg)
        if max_iters > 0 and cur_iters >= max_iters:
            break
        
        pass

    return event_seq


def tp_matrix_to_event_seq(tp_matrix, timecell_ms):
    '''
    Converts a dense time/pitch matrix into a sequence of events.
    :param tp_matrix (P, T) numpy array: Timecell / frequency binary description of melody.
    :param timecell_ms (int): Duration of one timecell (i.e. element in the horizontal axis) [ms].
    :return event_seq (list of (str, int) tuples): List of DAW commands encoded as performance events.
    '''
    # NOTE: In MIDI, pitch index 69 = note A4 = frequency 440 Hz.
    (P, T) = tp_matrix.shape
    assert P == 128

    # Indicates whether a note is currently active, and since when.
    note_status = np.ones(P, dtype=np.int32) * (-1)
    event_seq = []

    # Iterate temporally from start to end first.
    for t in range(0, T):
        for p in range(0, P):

            if tp_matrix[p, t]:
                if note_status[p] == -1:
                    # From inactive to active.
                    event_seq.append(('NOTE_ON', p))
                    note_status[p] = t

            else:
                if note_status[p] != -1:
                    # From active to inactive.
                    event_seq.append(('NOTE_OFF', p))
                    note_status[p] = -1

        # Pass some time (merge durations into a single command whenever possible).
        if len(event_seq) != 0 and event_seq[-1][0] == 'TIME_SHIFT':
            event_seq[-1] = ('TIME_SHIFT', event_seq[-1][1] + timecell_ms)
        else:
            event_seq.append(('TIME_SHIFT', timecell_ms))

    return event_seq


def event_seq_to_tp_matrix(event_seq, duration_ms, timecell_ms):
    '''
    Converts an event sequence to a time/pitch matrix.
    :param event_seq (list of (str, int) tuples): List of DAW commands encoded as performance events.
    :param return (P, T) numpy array: Timecell / frequency binary description of melody.
    '''
    P = 128
    
    if duration_ms == 'auto':
        # Accumulate all time shifts to determine encompassing control matrix size.
        auto_duration = sum([event[1] for event in event_seq if event[0] == 'TIME_SHIFT']) + \
            timecell_ms * 10
        T = int(round(auto_duration / timecell_ms))
    else:
        T = int(round(duration_ms / timecell_ms))
    
    tp_matrix = np.zeros((P, T), dtype=np.float32)

    for p in range(0, P):
        passed_ms = 0
        timecell_idx = 0
        note_on_start = -1

        for event in event_seq:
            command = event[0]  # NOTE_ON / NOTE_OFF / TIME_SHIFT.
            parameter = int(event[1])  # duration [ms] / note [pitch].

            if command == 'NOTE_ON' and parameter == p:
                note_on_start = timecell_idx

            elif command == 'NOTE_OFF' and parameter == p:
                assert note_on_start != -1, \
                    'Invalid event sequence detected (NOTE OFF appears without preceding NOTE ON).'
                if timecell_idx > note_on_start:
                    tp_matrix[p, note_on_start:timecell_idx] = 1.0
                note_on_start = -1

            elif command == 'TIME_SHIFT':
                passed_ms += parameter
                timecell_idx = int(round(passed_ms / timecell_ms))
                
                if timecell_idx >= T:
                    break

        # Fill row to the end in case a note off command never came.
        if note_on_start != -1:
            for t in range(note_on_start, T):
                tp_matrix[p, t] = 1.0

    return tp_matrix


def event_seq_to_waveform(soundfont_path, event_seq, Fs, mono, desired_samples):
    '''
    Uses pyfluidsynth to synthesize the given sequence of events into audio samples.
    :param soundfont_path (str): Path to SoundFont instrument file.
    :param event_seq (list of (str, int) tuples): List of DAW commands encoded as performance
        events.
    :param Fs (int): Sample rate [Hz].
    :return samples (T) int16 numpy array.
    NOTE: The length T is determined by the total sum of TIME_SHIFT values in event_seq.
    '''
    # QUICKSTART:
    # program_select(track, soundfontid, banknum, presetnum)
    # noteon(track, midinum, velocity)
    # noteoff(track, midinum)
    # get_samples(len)

    # https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    # with utils.suppress_stdout():
    if 1:

        # Instantiate synthesizer with a single soundfont instrument.
        fl = fluidsynth.Synth()
        sfid = fl.sfload(soundfont_path)
        fl.program_select(0, sfid, 0, 0)
        amplify = 4.0  # Some soundfonts are quieter than others.

        # Run through list of events and encode them into sound.
        samples_list = []

        for event in event_seq:
            command = event[0]  # NOTE_ON / NOTE_OFF / TIME_SHIFT.
            parameter = int(event[1])  # duration [ms] / note [pitch].

            if command == 'NOTE_ON':
                midinum = parameter
                velocity = 75
                fl.noteon(0, midinum, velocity)

            elif command == 'NOTE_OFF':
                midinum = parameter
                fl.noteoff(0, midinum)

            elif command == 'TIME_SHIFT':
                duration = parameter / 1000.0  # in seconds.
                num_samples = int(duration * Fs)
                cur_samples = fl.get_samples(num_samples)
                cur_samples = cur_samples.astype(np.int16)
                samples_list.append(cur_samples)

        fl.delete()
        samples = np.concatenate(samples_list, axis=0)

        # Stereo to mono.
        if mono:
            samples = (samples[::2] + samples[1::2]) / 2.0

        # Normalize.
        samples = samples * amplify / 32768.0

        if samples.shape[0] < desired_samples:
            samples = np.pad(samples, ((0, desired_samples - samples.shape[0])))
        else:
            samples = samples[:desired_samples]

        assert samples.shape == (desired_samples, )

        return samples