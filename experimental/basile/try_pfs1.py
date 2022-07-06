# BVH, Feb 2022.

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'experimental/'))
sys.path.append(os.path.join(os.getcwd(), 'third_party/'))
sys.path.append(os.path.join(os.getcwd(), 'third_party/pyfluidsynth'))
sys.path.append(os.path.join(os.getcwd(), 'third_party/pyfluidsynth/test'))

from __init__ import *

# Library imports.
import time
import fluidsynth


'''
QUICKSTART:
program_select(track, soundfontid, banknum, presetnum)
noteon(track, midinum, velocity)
noteoff(track, midinum)
get_samples(len)
'''

if __name__ == '__main__':

    fs = fluidsynth.Synth()
    fs.start()

    sfid = fs.sfload("third_party/pyfluidsynth/test/example.sf2")
    fs.program_select(0, sfid, 0, 0)

    fs.noteon(0, 60, 60)
    fs.noteon(0, 67, 60)
    fs.noteon(0, 76, 60)

    time.sleep(2.0)

    fs.noteoff(0, 60)
    fs.noteoff(0, 67)
    fs.noteoff(0, 76)

    time.sleep(1.0)

    fs.delete()
