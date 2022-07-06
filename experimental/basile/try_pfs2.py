# BVH, Feb 2022.

import os
import pathlib
import sys
for base_dir in [os.getcwd(), str(pathlib.Path(os.getcwd()).parent)]:
    print(base_dir)
    sys.path.append(base_dir)
    sys.path.append(os.path.join(base_dir, 'experimental/'))
    sys.path.append(os.path.join(base_dir, 'third_party/'))
    sys.path.append(os.path.join(base_dir, 'third_party/pyfluidsynth'))
    sys.path.append(os.path.join(base_dir, 'third_party/pyfluidsynth/test'))

from __init__ import *

# Library imports.
import time
import numpy
import pyaudio
import fluidsynth

'''
QUICKSTART:
program_select(track, soundfontid, banknum, presetnum)
noteon(track, midinum, velocity)
noteoff(track, midinum)
get_samples(len)
'''

if __name__ == '__main__':

    Fs = 44100

    pa = pyaudio.PyAudio()
    strm = pa.open(
        format = pyaudio.paInt16,
        channels = 2, 
        rate = Fs, 
        output = True)

    s = []

    fl = fluidsynth.Synth()

    # Initial silence is 1 second
    s = numpy.append(s, fl.get_samples(Fs * 1))

    sfid = fl.sfload("third_party/pyfluidsynth/test/example.sf2")
    fl.program_select(0, sfid, 0, 0)

    fl.noteon(0, 60, 60)
    fl.noteon(0, 67, 60)
    fl.noteon(0, 76, 60)

    print(np.unique(fl.get_samples(Fs * 1)))

    # Chord is held for 2 seconds
    s = numpy.append(s, fl.get_samples(Fs * 2))

    fl.noteoff(0, 60)
    fl.noteoff(0, 67)
    fl.noteoff(0, 76)

    # Decay of chord is held for 1 second
    s = numpy.append(s, fl.get_samples(Fs * 1))

    fl.delete()

    # s = s[::2]
    x = s[::2]
    print('x:', x.shape, x.dtype, x.min(), x.max(), x.mean())   

    samps = fluidsynth.raw_audio_string(s)

    print(s.shape, len(samps), s.shape[0] / Fs, len(samps) / Fs)

    strm.write(samps)
