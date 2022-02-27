import librosa
import matplotlib.pyplot as plt
import json
from PIL import Image
import torch
from torchvision import transforms
from librosa.filters import mel as librosa_mel_fn
import msynth.efficientnet_pytorch.model as modeleff
from torch.autograd import Variable
import torchaudio 
import einops 
import os
import soundfile as sf

torch.autograd.set_detect_anomaly(True)
def sampler(controls): #(16,1,4)
    sampled = torch.bmm(controls,
            samples)    
    synthesized_audio = einops.rearrange(sampled, 'b d s -> d (b s)')#(1,465696)
    return synthesized_audio

model = modeleff.EfficientNet.from_name('efficientnet-b0')
model._change_in_channels(1)
        

list_of_notes = ["47.wav","44.wav","42.wav","40.wav"]
song=["mary.wav"]

ground_truth = torch.tensor([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,0],
             [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
             [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,0,0],
             [1,0,0,0],[1,0,0,0],[1,0,0,0]]).type(torch.DoubleTensor)
audio_d,sr = torchaudio.load("../data/musicfiles/" + list_of_notes[0])
audio_b,sr = torchaudio.load("../data/musicfiles/" + list_of_notes[1])
audio_a,sr = torchaudio.load("../data/musicfiles/" + list_of_notes[2])
audio_g,sr = torchaudio.load("../data/musicfiles/" + list_of_notes[3])
mary,sr = torchaudio.load("../data/musicfiles/" + song[0])

samples = torch.cat([torch.unsqueeze(audio_d,dim=0),torch.unsqueeze(audio_b,dim=0),
    torch.unsqueeze(audio_a,dim=0),torch.unsqueeze(audio_g,dim=0)],dim=0) #(4,1,29106)
samples = samples.permute((1,0,2)).repeat(16,1,1).type(torch.DoubleTensor) #(16,4,29106)

resynthesized_mary = sampler(torch.unsqueeze(ground_truth,dim=1))

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

ground_truth = torch.unsqueeze(ground_truth, dim=2) #(16,4,1)

loss = torch.nn.MSELoss()

mel_basis = librosa.filters.mel(44100,n_fft=441*4,n_mels=128)
mel_basis = torch.from_numpy(mel_basis).float()

stft_mary = torch.stft(resynthesized_mary.type(torch.FloatTensor),n_fft=441*4,hop_length=441*2)
real_part, imag_part = stft_mary.unbind(-1)
magnitude_mary = torch.sqrt(real_part ** 2 + imag_part ** 2)
mel_output = torch.matmul(mel_basis, magnitude_mary)
log_mel_spec_mary = torch.log10(torch.clamp(mel_output, min=1e-5))#(1,128,529)


ground_truth=ground_truth.permute((0,2,1)) #(16,1,4)


for i in range(100):

    optimizer.zero_grad()
    feats = model.extract_features(torch.unsqueeze(log_mel_spec_mary,dim=0))[0] #(4,1,16)
    feats=feats.permute((2,1,0)) #(16,1,4)
    output = torch.nn.functional.sigmoid(feats*10).type(torch.DoubleTensor)
    synthesized_audio = sampler(output)
    stft_concat = torch.stft(synthesized_audio,n_fft=441*4,hop_length=441*2)
    real_part, imag_part = stft_concat.unbind(-1)
    magnitude_concat = torch.sqrt(real_part ** 2 + imag_part ** 2)
    mel_output = torch.matmul(mel_basis, magnitude_concat.float())
    log_mel_spec_concat = torch.log10(torch.clamp(mel_output, min=1e-5))#(1,128,529)

    loss_val = loss(log_mel_spec_concat,log_mel_spec_mary)
    loss_val.backward()
    """
    
    plt.figure()
    plt.imshow(log_mel_spec_concat[0].detach().numpy())
    plt.savefig("melspec_concat.png")
    
    plt.figure()
    plt.imshow(log_mel_spec_mary[0])
    plt.savefig("melspec_mary.png")"""
    
    optimizer.step()
    print(loss_val)

if os.path.isdir("reconstruction"):
    os.mkdir("reconstruction")
sf.write("reconstruction/resynthesized_mary.wav", resynthesized_mary.permute(1,0).detach().numpy(), 44100)

sf.write("reconstruction/synthesized_audio.wav", synthesized_audio.permute(1,0).detach().numpy(), 44100)


