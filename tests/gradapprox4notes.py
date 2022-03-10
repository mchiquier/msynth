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
    mel_basis = torch.from_numpy(mel_basis).float().cuda()
    stft = torch.stft(audio,n_fft=441*4,hop_length=441*2)
    real_part, imag_part = stft.unbind(-1)
    magnitude_concat = torch.sqrt(real_part ** 2 + imag_part ** 2)
    mel_output = torch.matmul(mel_basis, magnitude_concat.float())
    log_mel_spec_concat = torch.log10(torch.clamp(mel_output, min=1e-5))#(1,128,529)
    return log_mel_spec_concat

def synthesize(path_to_musicfiles, controls):
    """This function returns the synthesized audio from the given matrix. 
    It also returns the samples used to do that. This is useful because later we will want a samples 
    matrix that has the same first dimension as the controls matrix, as thats what gets given to 
    the sampler. So synthesize is now used to generate the ground truth, but the samples generated are 
    reapplied at inference time when the model has predicted its own controls"""
    list_of_notes = ["47.wav","44.wav","42.wav","40.wav"]
    audio_d,sr = torchaudio.load(path_to_musicfiles + list_of_notes[0])
    audio_b,sr = torchaudio.load(path_to_musicfiles + list_of_notes[1])
    audio_a,sr = torchaudio.load(path_to_musicfiles + list_of_notes[2])
    audio_g,sr = torchaudio.load(path_to_musicfiles + list_of_notes[3])
    samples = torch.cat([torch.unsqueeze(audio_d,dim=0),torch.unsqueeze(audio_b,dim=0),
    torch.unsqueeze(audio_a,dim=0),torch.unsqueeze(audio_g,dim=0)],dim=0) #(4,1,29106)
    samples = samples.permute((1,0,2)).type(torch.cuda.DoubleTensor).repeat(controls.shape[0],1,1)#(16,4,29106)
   
    resynthesized = sampler(torch.unsqueeze(controls,dim=1),samples) 
    return resynthesized, samples

class FunctionWithNumericalGrad(torch.autograd.Function):
    
    @staticmethod
    def forward(cxt, synth_params, samples, epsilon, num_avg):
        cxt.save_for_backward(synth_params,samples, epsilon, num_avg)
        # do forward process
        synth_signal = sampler(synth_params, samples)

        return synth_signal

    @staticmethod
    def backward(cxt, grad_output): 

        synth_params = cxt.saved_tensors[0]
        samples = cxt.saved_tensors[1]
        epsilon = cxt.saved_tensors[2].numpy()
        num_avg = cxt.saved_tensors[3].numpy()[0]

        # generate random vector of +-1s, #delta_k = rademacher(synth_params.shape).numpy()
        list_of_grads = []
        for i in range(num_avg):
            delta_k= np.random.randint(0,high=2,size=synth_params.shape)
            delta_k[delta_k==0]=-1
            
            # synth +- forward and backward passes
            updated_plus = torch.min(synth_params + epsilon*delta_k, torch.tensor([1.0]))
            updated_minus = torch.max(synth_params - epsilon*delta_k, torch.tensor([0.0]))
            J_plus = sampler(updated_plus, samples)
            J_minus = sampler(updated_minus, samples) 
            grad_synth = J_plus - J_minus 

            grad_synth = grad_synth.reshape(-1)
            delta_k = delta_k.reshape(-1) 

            # loop for each element of synth_params
            params_sublist = []
            for sub_p_idx in range(len(delta_k)):
                grad_s = grad_synth / (2 * epsilon * delta_k[sub_p_idx])
                params_sublist.append(torch.sum(grad_output[0] * grad_s))

            grad_synth = torch.tensor(params_sublist).reshape(synth_params.shape)
            list_of_grads.append(torch.unsqueeze(grad_synth,dim=0))

        grad_synth = torch.mean(torch.cat(list_of_grads),axis=0) 

        return grad_synth, None, None, None


def train():

    log_wandb = False
    theconfig = {"epsilon": 0.001, 
    "num_notes": 16, 
    "num_avg": 20, 
    "learning_rate": 0.001, 
    "recurrent": True, 
    "recurrent_aggregate": False,
    "apply_softmax": False,
    "apply_sigmoid": True,
    "real_gradient": True,
    "freq_version": True,
    "num_iterations": 600}
    
    if log_wandb:
        wandb.init(config=theconfig)
        config = wandb.config
        num_notes= int(config.num_notes)
        num_avg=int(config.num_avg)
        epsilon=config.epsilon
        recurrent = config.recurrent
        learning_rate=config.learning_rate
        recurrent = config.recurrent
        freq_version = config.freq_version
        apply_softmax = config.apply_softmax
        apply_sigmoid = config.apply_sigmoid
        real_gradient = config.real_gradient
        num_iterations = config.num_iterations
    else:
        config = theconfig
        num_notes= int(config["num_notes"])
        num_avg=int(config["num_avg"])
        epsilon=config["epsilon"]
        recurrent = config["recurrent"]
        freq_version = config["freq_version"]
        learning_rate=config["learning_rate"]
        apply_sigmoid=config["apply_sigmoid"]
        recurrent = config["recurrent"]
        apply_softmax = config["apply_softmax"]
        real_gradient = config["real_gradient"]
        num_iterations = config["num_iterations"]


    ground_truth = torch.tensor([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,0],
        [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
        [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,0,0],
        [1,0,0,0],[1,0,0,0],[1,0,0,0]]).type(torch.cuda.DoubleTensor)
    
 
    resynthesized_mary, samples = synthesize("../data/musicfiles/", ground_truth[:num_notes])
    log_mel_spec_mary = get_melspec(resynthesized_mary.type(torch.cuda.FloatTensor))

    """if log_wandb: 
        wandb.log({"ground-truth pitch-time":wandb.Table(data=ground_truth.detach().cpu().tolist(), columns=["47","44","42","40"])})
        wandb.log({"ground_truth audio": wandb.Audio(resynthesized_mary[0].detach().cpu().numpy(), caption="ground truth audio", sample_rate=44100)})
        wandb.log({"ground-truth melspec": wandb.Image(log_mel_spec_mary[0].cpu().detach().numpy(), caption="GT Melspec")})"""
    
    if recurrent:
        model = modeleff.EfficientNet.from_name('efficientnet-b0',num_notes=1, freqversion = freq_version).cuda()
    else:
        model = modeleff.EfficientNet.from_name('efficientnet-b0',num_notes=num_notes,freqversion = freq_version).cuda()

    sigmoid = torch.nn.Sigmoid().cuda() 
    softmax = torch.nn.Softmax(dim=2).cuda()
        
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss = torch.nn.MSELoss()
    current_loss = torch.tensor([10000.0]).cuda()

    if not os.path.isdir("reconstruction"):
        os.mkdir("reconstruction")

    for k in range(num_iterations): 
        print(k)
        
        if recurrent: 
            list_of_audio=[]   
            list_of_controls=[]        
            for j in range(16):
                optimizer.zero_grad()

                thenote, samples = synthesize("../data/musicfiles/", ground_truth[j:j+1])
                log_mel_spec_note = get_melspec(thenote.type(torch.cuda.FloatTensor))
                controls = model.extract_features(torch.unsqueeze(log_mel_spec_note,dim=0).cuda())[0] 
                if freq_version:
                    controls=controls.permute((2,0,1)) 
                else:
                    controls = controls.reshape(4,1,1)
                    controls=controls.permute((2,1,0))
                if apply_sigmoid:
                    controls = sigmoid(controls).type(torch.cuda.DoubleTensor)
                if apply_softmax:
                    controls = softmax(controls).type(torch.cuda.DoubleTensor)

                if real_gradient:
                    synthesized_audio = sampler(controls, samples.cuda())
                else:
                    synthesized_audio = FunctionWithNumericalGrad.apply(controls, samples.cuda(), torch.tensor([epsilon]).cuda(), torch.tensor([num_avg]).cuda())
            

                log_mel_spec_concat = get_melspec(synthesized_audio)

                if log_wandb and k%20==0:
                        wandb.log({"predicted pitch-time" + str(j):wandb.Table(data=controls[:,0,:].detach().cpu().tolist(), columns=["47","44","42","40"])})
                        wandb.log({"difference pitch-time"+ str(j):wandb.Table(data=(torch.unsqueeze(ground_truth[j],dim=0)-controls[:,0,:]).detach().cpu().tolist(), columns=["47","44","42","40"])})
                        wandb.log({"reconstructed audio"+ str(j): wandb.Audio(synthesized_audio[0].detach().cpu().numpy(), caption="reconstructed audio " + str(k), sample_rate=44100)})
                        wandb.log({"predicted melspec"+ str(j): wandb.Image(log_mel_spec_concat[0].cpu().detach().numpy(), caption="Predicted Melspec  " + str(k))})
                        wandb.log({"difference melspec" + str(j): wandb.Image(log_mel_spec_concat[0].cpu().detach().numpy()-log_mel_spec_note[0].cpu().detach().numpy(), caption="Predicted Melspec  " + str(k))})
                        wandb.log({"ground-truth pitch-time" + str(j):wandb.Table(data=torch.unsqueeze(ground_truth[j],dim=0).detach().cpu().tolist(), columns=["47","44","42","40"])})
                        wandb.log({"ground_truth audio" + str(j): wandb.Audio(thenote[0].detach().cpu().numpy(), caption="ground truth audio", sample_rate=44100)})
                        wandb.log({"ground-truth melspec" + str(j): wandb.Image(log_mel_spec_note[0].cpu().detach().numpy(), caption="GT Melspec")})
                loss_val = loss(log_mel_spec_concat,log_mel_spec_note.cuda())
                loss_val.backward()

                optimizer.step()
                
                if log_wandb: 
                    wandb.log({"reconstruction loss" : loss_val.detach().cpu().numpy(),"iteration": k})

                print("Loss is: ", loss_val)

                if loss_val < current_loss: 
                    print("Saved")
                    sf.write("reconstruction/resynthesized_mary.wav", resynthesized_mary.permute(1,0).cpu().detach().numpy(), 44100)
                    sf.write("reconstruction/synthesized_audio.wav", synthesized_audio.permute(1,0).cpu().detach().numpy(), 44100)
                    current_loss = loss_val

train()
