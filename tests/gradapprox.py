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
    samples = samples.permute((1,0,2)).type(torch.DoubleTensor).repeat(controls.shape[0],1,1)#(16,4,29106)
   
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

        #for i in range(len(params_sublist)):
        #    wandb.log({"param_" + str(i) + "_predicted_grad" :  params_sublist[i]})

        return grad_synth, None, None, None



def train():
   
    theconfig = {"epsilon": 0.001, "num_notes": 16, "num_avg": 20, "learning_rate": 0.001, "recurrent": False, "recurrent_aggreegate": False}
    
    wandb.init(config=theconfig)

    config = wandb.config
    num_notes= int(config.num_notes)
    num_avg=int(config.num_avg)
    epsilon=config.epsilon
    recurrent = config.recurrent
    learning_rate=config.learning_rate
    recurrent = False

    sigmoid = torch.nn.Sigmoid() 
    softmax = torch.nn.Softmax(dim=2)

    ground_truth = torch.tensor([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,0],
        [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
        [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,0,0],
        [1,0,0,0],[1,0,0,0],[1,0,0,0]]).type(torch.DoubleTensor)
    

    resynthesized_mary, samples = synthesize("../data/musicfiles/", ground_truth[:num_notes])
    if recurrent:
        model = modeleff.EfficientNet.from_name('efficientnet-b0',num_notes=1) 
    else:
        model = modeleff.EfficientNet.from_name('efficientnet-b0',num_notes=num_notes)
        
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    ground_truth_u = torch.unsqueeze(ground_truth, dim=2) #(16,4,1)
    ground_truth_matrix =ground_truth_u.permute((0,2,1)) #(16,1,4)

    loss = torch.nn.MSELoss()
    log_mel_spec_mary = get_melspec(resynthesized_mary.type(torch.FloatTensor))
    current_loss = torch.tensor([10000.0])

    if not os.path.isdir("reconstruction"):
        os.mkdir("reconstruction")

    for k in range(500): 
        print(k)

        if recurrent: 
            optimizer.zero_grad()
            list_of_output=[]
            for j in range(16):
        
                #thenote = load_resynthesized_note("../data/musicfiles/", j)
                thenote, samples = synthesize("../data/musicfiles/", torch.unsqueeze(ground_truth[j],dim=0))
                log_mel_spec_note = get_melspec(thenote.type(torch.FloatTensor))
                feats = model.extract_features(torch.unsqueeze(log_mel_spec_note,dim=0))[0] #(4,1,16)
                feats=feats.permute((2,1,0)) #(16,1,4)
                output = sigmoid(feats).type(torch.DoubleTensor)
                #output = softmax(output)
                output_reshaped = output.reshape(-1)
                ground_truth_reshaped = ground_truth_matrix.reshape(-1)

                #for i in range(len(output_reshaped)):
                #print(ground_truth[j][i],output_reshaped[i])
                #    wandb.log({"param_" + str(i) + "_predicted" :  output_reshaped[i], "iteration": k})
                #    print(j,i)
                #    wandb.log({"param_" + str(i) + "_gt" :  ground_truth_reshaped[i], "iteration": k})
                #print("end")
                curr_audio = sampler(output, samples)
                #curr_audio = FunctionWithNumericalGrad.apply(output, samples, torch.tensor([epsilon]), torch.tensor([num_avg]))
            
                list_of_output.append(curr_audio)
            synthesized_audio = torch.cat(list_of_output,dim=1)
        
        else: 
            optimizer.zero_grad()
        

            feats = model.extract_features(torch.unsqueeze(log_mel_spec_mary,dim=0))[0] #(4,1,16)
            feats=feats.permute((2,1,0)) #(16,1,4)
            output = sigmoid(feats*10).type(torch.DoubleTensor)
            #output = softmax(output)
            output_reshaped = output.reshape(-1)
            ground_truth_reshaped = ground_truth_matrix.reshape(-1)
            for i in range(len(output_reshaped)):
                wandb.log({"param_" + str(i) + "_predicted" :  output_reshaped[i], "iteration": k})
                wandb.log({"param_" + str(i) + "_gt" :  ground_truth_reshaped[i], "iteration": k})

            synthesized_audio = sampler(output, samples)
            #synthesized_audio = FunctionWithNumericalGrad.apply(output, samples, torch.tensor([epsilon]), torch.tensor([num_avg]))
        
        log_mel_spec_concat = get_melspec(synthesized_audio)

        loss_val = loss(log_mel_spec_concat,log_mel_spec_mary)
        loss_val.backward()

        optimizer.step()

        wandb.log({"loss":loss_val.detach().cpu().numpy(),"iteration": k})

        print("Loss is: ", loss_val)

        if loss_val < current_loss: 
            print("saved")
            sf.write("reconstruction/resynthesized_mary.wav", resynthesized_mary.permute(1,0).detach().numpy(), 44100)
            sf.write("reconstruction/synthesized_audio.wav", synthesized_audio.permute(1,0).detach().numpy(), 44100)
            current_loss = loss_val

def train_recurrent():

   
    theconfig = {"epsilon": 0.0001, "num_notes": 16, "num_avg": 10, "learning_rate": 0.01, "recurrent_stream": True, "recurrent_aggreegate": False}
    
    wandb.init(config=theconfig)

    config = wandb.config
    num_notes= int(config.num_notes)
    num_avg=int(config.num_avg)
    epsilon=config.epsilon
    learning_rate=config.learning_rate

    ground_truth = torch.tensor([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,0],
        [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
        [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,0,0],
        [1,0,0,0],[1,0,0,0],[1,0,0,0]]).type(torch.DoubleTensor)

    ground_truth_u = torch.unsqueeze(ground_truth, dim=2) #(16,4,1)
    ground_truth_matrix =ground_truth_u.permute((0,2,1)) #(16,1,4)
    
    resynthesized_mary, _ = synthesize("../data/musicfiles/", ground_truth)
    #_, resynthesized_mary, _ = load_samples_and_resynthesized("../data/musicfiles/", num_notes)
    #samples, _, ground_truth = load_samples_and_resynthesized("../data/musicfiles/", 1)
    model = modeleff.EfficientNet.from_name('efficientnet-b0',num_notes=1) 
        
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


    loss = torch.nn.MSELoss()
    log_mel_spec_mary = get_melspec(resynthesized_mary.type(torch.FloatTensor))

    sigmoid = torch.nn.Sigmoid() 
    #softmax = torch.nn.Softmax(dim=2)

    for i in range(500):
        print(i)

        optimizer.zero_grad()
        list_of_output=[]
        for j in range(16):
        
            #thenote = load_resynthesized_note("../data/musicfiles/", j)
            thenote, samples = synthesize("../data/musicfiles/", torch.unsqueeze(ground_truth[j],dim=0))
            log_mel_spec_note = get_melspec(thenote.type(torch.FloatTensor))
            feats = model.extract_features(torch.unsqueeze(log_mel_spec_note,dim=0))[0] #(4,1,16)
            feats=feats.permute((2,1,0)) #(16,1,4)
            output = sigmoid(feats).type(torch.DoubleTensor)
            #output = softmax(output)
            output_reshaped = output.reshape(-1)
            ground_truth_reshaped = ground_truth_matrix.reshape(-1)

            for i in range(len(output_reshaped)):
                #print(ground_truth[j][i],output_reshaped[i])
                wandb.log({"param_" + str(i) + "_predicted" :  output_reshaped[i], "iteration": i})
                wandb.log({"param_" + str(i) + "_gt" :  ground_truth[j][i], "iteration": i})
            #print("end")
            synthesized_audio = sampler(output, samples)
            #synthesized_audio = FunctionWithNumericalGrad.apply(output, samples, torch.tensor([epsilon]), torch.tensor([num_avg]))
            
            
            list_of_output.append(synthesized_audio)
        total_audio = torch.cat(list_of_output,dim=1)
        log_mel_spec_concat = get_melspec(total_audio)

        loss_val = loss(log_mel_spec_concat,log_mel_spec_mary)
        loss_val.backward()

        optimizer.step()

        wandb.log({"loss":loss_val.detach().cpu().numpy(),"iteration": i})

        print("Loss is: ", loss_val)

    if not os.path.isdir("reconstruction"):
        os.mkdir("reconstruction")

    sf.write("reconstruction/resynthesized_mary_recurrent.wav", resynthesized_mary.permute(1,0).detach().numpy(), 44100)
    sf.write("reconstruction/synthesized_audio_recurrent.wav", total_audio.permute(1,0).detach().numpy(), 44100)

train()
