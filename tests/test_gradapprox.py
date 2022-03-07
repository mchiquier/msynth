import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import json
from PIL import Image 
import torch
from librosa.filters import mel as librosa_mel_fn
import msynth.efficientnet_pytorch.model as modeleff
from torch.autograd import Variable
import torchaudio 
import einops 
import pdb
import os
import soundfile as sf
import wandb


def sampler(controls, samples): #(16,1,4)
    sampled = torch.bmm(controls,
            samples)    
    synthesized_audio = einops.rearrange(sampled, 'b d s -> d (b s)')#(1,465696)
    return synthesized_audio

def get_melspec(audio):
    mel_basis = librosa.filters.mel(44100,n_fft=441*4,n_mels=128)
    mel_basis = torch.from_numpy(mel_basis).float()
    stft = torch.stft(audio,n_fft=441*4,hop_length=441*2)
    real_part, imag_part = stft.unbind(-1)
    magnitude_concat = torch.sqrt(real_part ** 2 + imag_part ** 2)
    mel_output = torch.matmul(mel_basis, magnitude_concat.float())
    log_mel_spec_concat = torch.log10(torch.clamp(mel_output, min=1e-5))#(1,128,529)
    return log_mel_spec_concat

class FunctionWithNumericalGrad(torch.autograd.Function):
    
    @staticmethod
    def forward(cxt, synth_params, samples, epsilon):
        cxt.save_for_backward(synth_params,samples, epsilon)
        # do forward process
        synth_signal = sampler(synth_params, samples)

        return synth_signal

    @staticmethod
    def backward(cxt, grad_output): 


        synth_params = cxt.saved_tensors[0]
        samples = cxt.saved_tensors[1]
        epsilon = cxt.saved_tensors[2].numpy()

        # generate random vector of +-1s, #delta_k = rademacher(synth_params.shape).numpy()
        delta_k= np.random.randint(0,high=2,size=synth_params.shape)
        delta_k[delta_k==0]=-1
        
        # synth +- forward and backward passes
        J_plus = sampler(synth_params + delta_k, samples)
        J_minus = sampler(synth_params - delta_k, samples)
        grad_synth = J_plus - J_minus 

        grad_synth = grad_synth.reshape(-1)
        delta_k = delta_k.reshape(-1) 
        print(delta_k)

        # loop for each element of synth_params
        params_sublist = []
        for sub_p_idx in range(len(delta_k)):
            grad_s = grad_synth / (2 * epsilon * delta_k[sub_p_idx])
            params_sublist.append(torch.sum(grad_output[0] * grad_s))

        grad_synth = torch.tensor(params_sublist).reshape(synth_params.shape)

        #for i in range(len(params_sublist)):
        #    wandb.log({"param_" + str(i) + "_predicted_grad" :  params_sublist[i]})

        return grad_synth, None, None

def load_samples_and_resynthesized(path_to_musicfiles, num_notes):
    list_of_notes = ["47.wav","44.wav","42.wav","40.wav"]
    song=["mary.wav"]

    
    ground_truth = torch.tensor([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,0],
        [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
        [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,0,0],
        [1,0,0,0],[1,0,0,0],[1,0,0,0]]).type(torch.DoubleTensor)

    ground_truth = ground_truth[:num_notes]

        
    audio_d,sr = torchaudio.load(path_to_musicfiles + list_of_notes[0])
    audio_b,sr = torchaudio.load(path_to_musicfiles + list_of_notes[1])
    audio_a,sr = torchaudio.load(path_to_musicfiles + list_of_notes[2])
    audio_g,sr = torchaudio.load(path_to_musicfiles + list_of_notes[3])

    samples = torch.cat([torch.unsqueeze(audio_d,dim=0),torch.unsqueeze(audio_b,dim=0),
    torch.unsqueeze(audio_a,dim=0),torch.unsqueeze(audio_g,dim=0)],dim=0) #(4,1,29106)
    samples = samples.permute((1,0,2)).repeat(num_notes,1,1).type(torch.DoubleTensor) #(16,4,29106)
    song=["mary.wav"]
    mary,sr = torchaudio.load("../data/musicfiles/" + song[0])
    resynthesized_mary = sampler(torch.unsqueeze(ground_truth,dim=1),samples)

    return samples, resynthesized_mary, ground_truth 

def train():

    num_notes=1
    epsilon=torch.tensor([0.0001])
    wandb.init(name="epsilon_" + str(epsilon.numpy()[0]))
    samples, resynthesized_mary, ground_truth = load_samples_and_resynthesized("../data/musicfiles/", num_notes)
    model = modeleff.EfficientNet.from_name('efficientnet-b0',num_notes=num_notes) 
        
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    ground_truth = torch.unsqueeze(ground_truth, dim=2) #(16,4,1)
    ground_truth_matrix =ground_truth.permute((0,2,1)) #(16,1,4)

    loss = torch.nn.MSELoss()
    log_mel_spec_mary = get_melspec(resynthesized_mary.type(torch.FloatTensor))

    for i in range(300):

        optimizer.zero_grad()
        

        feats = model.extract_features(torch.unsqueeze(log_mel_spec_mary,dim=0))[0] #(4,1,16)
        feats=feats.permute((2,1,0)) #(16,1,4)
        output = torch.nn.functional.sigmoid(feats*10).type(torch.DoubleTensor)

        output_reshaped = output.reshape(-1)
        ground_truth_reshaped = ground_truth_matrix.reshape(-1)
        for i in range(len(output_reshaped)):
            wandb.log({"param_" + str(i) + "_predicted" :  output_reshaped[i]})
            wandb.log({"param_" + str(i) + "_gt" :  ground_truth_reshaped[i]})

        synthesized_audio = FunctionWithNumericalGrad.apply(output, samples, epsilon)
        
        log_mel_spec_concat = get_melspec(synthesized_audio)

        loss_val = loss(log_mel_spec_concat,log_mel_spec_mary)
        loss_val.backward()

        optimizer.step()

        wandb.log({"loss":loss_val.detach().cpu().numpy()})

        print("Loss is: ", loss_val)

    if not os.path.isdir("reconstruction"):
        os.mkdir("reconstruction")

    sf.write("reconstruction/resynthesized_mary.wav", resynthesized_mary.permute(1,0).detach().numpy(), 44100)
    sf.write("reconstruction/synthesized_audio.wav", synthesized_audio.permute(1,0).detach().numpy(), 44100)

train()