import time
import fluidsynth
import fluidsynth
import numpy 
from scipy.io.wavfile import write
import os
import pdb
import numpy as np


s = []
fs = fluidsynth.Synth()
#fs.set_reverb(roomsize=1.0, damping=1.0, width=100.0, level=1.0)
#fs.set_chorus(nr=10, level=10.0, speed=1.0, depth=0.5, type=1)
## Your installation of FluidSynth may require a different driver.
## Use something like:
# fs.start(driver="pulseaudio")
print(os.listdir("../"))
sfid = fs.sfload("../soundfonts/piano.sf2")
fs.program_select(0, sfid, 0, 0) #pick the instrument

fs.noteon(0, 44, 120) #channel, pitch, velocity
for i in range(15):
    fs.pitch_bend(0, i*512)
    s = numpy.append(s, fs.get_samples(int(44100 *0.05))) #how long I want the note to be 

#s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 44) #which note do I want to turn off 
print(s.shape)
print(np.unique(s))
fs.noteon(0, 42, 120) #channel, pitch, velocity
for i in range(15):
    fs.pitch_bend(0, i*512)
    s = numpy.append(s, fs.get_samples(int(44100 *0.05))) 
#s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 42)
"""
print(s.shape)
fs.noteon(0, 40, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 40) 
print(s.shape)
fs.noteon(0, 42, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 42)
fs.noteon(0, 44, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 44)
fs.noteon(0, 44, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 44)
fs.noteon(0, 44, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 44)
fs.noteon(0, 44, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 44)

fs.noteon(0, 42, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 42)
fs.noteon(0, 42, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 42)
fs.noteon(0, 42, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 42)
fs.noteon(0, 42, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 42)

fs.noteon(0, 44, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 44)
fs.noteon(0, 47, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 47)
fs.noteon(0, 47, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 47)
fs.noteon(0, 47, 120) #channel, pitch, velocity
s = numpy.append(s, fs.get_samples(int(44100 *0.33))) #how long I want the note to be 
fs.noteoff(0, 47)"""

print(s.shape)
print(np.unique(s))
fs.delete()
if not os.path.isdir("rendered_audio"):
    os.mkdir("rendered_audio")
write("rendered_audio/examplechorus_pitchshifttest.wav", 44100, s.astype(numpy.int16))

print('Starting playback')



