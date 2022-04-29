import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import json
from PIL import Image 
from scipy.io.wavfile import write
import torch
from librosa.filters import mel as librosa_mel_fn
import msynth.models.ef_model as modeleff
from torch.autograd import Variable
from hydra import utils
import torchaudio 
import einops 
import pdb
from omegaconf.dictconfig import DictConfig
from omegaconf import DictConfig, OmegaConf
import argparse
import os
import soundfile as sf
import wandb
import fluidsynth

s = []
fs = fluidsynth.Synth()
#fs.set_reverb(roomsize=1.0, damping=1.0, width=100.0, level=1.0)
#fs.set_chorus(nr=10, level=10.0, speed=1.0, depth=0.5, type=1)
## Your installation of FluidSynth may require a different driver.
## Use something like:
# fs.start(driver="pulseaudio")
sfid = fs.sfload("../data/soundfonts/piano.sf2")
fs.program_select(0, sfid, 0, 0) #pick the instrument


def test(i):
    
    notes=[47,44,42,40]
    for j in range(len(notes)):
        fs.noteon(0, notes[j], 120) #channel, pitch, velocity
        s = fs.get_samples(int(14553.0))
        audio_d = torch.tensor([s.astype(np.float32, order='C') / 32768.0]) 
        fs.noteoff(0, notes[j])
        pdb.set_trace()
        sf.write("../data/musicfiles/note_" + str(j) + "_iter_" + str(i) + ".wav", audio_d.permute(1,0).cpu().detach().numpy(), 44100)

def pyfluidsynth_sampler(controls): #(16,1,4)
   
    notes=[47,44,42,40]
    fs.noteon(0, notes[0], 120) #channel, pitch, velocity
    s = fs.get_samples(int(14553.0))
    audio_d = torch.tensor([s.astype(np.float32, order='C') / 32768.0]) 
    fs.noteoff(0, notes[0])

    fs.noteon(0, notes[1], 120) #channel, pitch, velocity
    s = fs.get_samples(int(14553.0))
    audio_b=torch.tensor([s.astype(np.float32, order='C') / 32768.0])
    fs.noteoff(0, notes[1])

    fs.noteon(0, notes[2], 120) #channel, pitch, velocity
    s = fs.get_samples(int(14553.0))
    audio_a = torch.tensor([s.astype(np.float32, order='C') / 32768.0])  
    fs.noteoff(0, notes[2])

    fs.noteon(0, notes[3], 120) #channel, pitch, velocity
    s = fs.get_samples(int(14553.0))
    audio_g = torch.tensor([s.astype(np.float32, order='C') / 32768.0]) 
    fs.noteoff(0, notes[3])

    samplestwo = torch.cat([torch.unsqueeze(audio_d,dim=0),torch.unsqueeze(audio_b,dim=0),
        torch.unsqueeze(audio_a,dim=0),torch.unsqueeze(audio_g,dim=0)],dim=0) #(4,1,29106)
    samplestwo = samplestwo.permute((1,0,2)).type(torch.cuda.DoubleTensor).repeat(controls.shape[0],1,1)#(16,4,29106)

    samplestwo.requires_grad = False

    synthesized_audio = sampler(controls,samplestwo)
        
    return synthesized_audio, samplestwo


def sampler(controls, samples): #(b,numnotes,1,4)
    
    
    list_of_notes=[] 
    for i in range(controls.shape[1]):
        sampled = torch.bmm(controls[:,i,:,:],
                samples)
        list_of_notes.append(sampled[:,0,:])
    synthesized_audio = torch.cat(list_of_notes,dim=1)
    #synthesized_audio = einops.rearrange(sampled, 'b d s -> d (b s)')#(1,465696)
    return synthesized_audio

def get_melspec(audio, device):
    mel_basis = librosa.filters.mel(44100,n_fft=441*4,n_mels=128)
    mel_basis = torch.from_numpy(mel_basis).float().to(device)
    stft = torch.stft(audio,n_fft=441*4,hop_length=441*2)
    real_part, imag_part = stft.unbind(-1)
    magnitude_concat = torch.sqrt(real_part ** 2 + imag_part ** 2)
    mel_output = torch.matmul(mel_basis, magnitude_concat.float())
    log_mel_spec_concat = torch.log10(torch.clamp(mel_output, min=1e-5))#(1,128,529)
    return log_mel_spec_concat

def synthesize(path_to_musicfiles, controls, pyfluidsynth):
    """This function returns the synthesized audio from the given matrix. 
    It also returns the samples used to do that. This is useful because later we will want a samples 
    matrix that has the same first dimension as the controls matrix, as thats what gets given to 
    the sampler. So synthesize is now used to generate the ground truth, but the samples generated are 
    reapplied at inference time when the model has predicted its own controls"""
    list_of_notes = ["47.wav","44.wav","42.wav","40.wav"]


    if not pyfluidsynth > 0: 
        audio_d,sr = torchaudio.load(path_to_musicfiles + list_of_notes[0])
        audio_b,sr = torchaudio.load(path_to_musicfiles + list_of_notes[1])
        audio_a,sr = torchaudio.load(path_to_musicfiles + list_of_notes[2])
        audio_g,sr = torchaudio.load(path_to_musicfiles + list_of_notes[3])
        samples = torch.cat([torch.unsqueeze(audio_d,dim=0),torch.unsqueeze(audio_b,dim=0),
        torch.unsqueeze(audio_a,dim=0),torch.unsqueeze(audio_g,dim=0)],dim=0) #(4,1,29106
        samples = samples.permute((1,0,2)).type(torch.cuda.DoubleTensor).repeat(controls.shape[0],1,1)#(16,4,29106)
        samples.requires_grad = False
        resynthesized = sampler(torch.unsqueeze(controls,dim=1),samples) 
        return resynthesized, samples
   
    else:
        resynthesized, samples = pyfluidsynth_sampler(torch.unsqueeze(controls,dim=1)) 
        return resynthesized, samples


class FunctionWithNumericalGradSTFT(torch.autograd.Function):
    
    @staticmethod
    def forward(cxt, synth_params, samples, epsilon, num_avg,pyfluidsynth):
        
        # do forward process
        cxt.save_for_backward(synth_params,samples, epsilon, num_avg,pyfluidsynth)
        device = synth_params.device
        if pyfluidsynth.to(device) > torch.tensor([0.0]).to(device):
            synth_signal = sampler(synth_params, samples)
            #synth_signal, samplestwo = pyfluidsynth_sampler(synth_params)
            
        else:
            
            synth_signal = sampler(synth_params, samples)

        log_mel_spec_concat = get_melspec(synth_signal, device)


        return log_mel_spec_concat

    @staticmethod
    def backward(cxt, grad_output): 
        
        synth_params = cxt.saved_tensors[0]
        device = synth_params.device
        samples = cxt.saved_tensors[1].to(device)
        epsilon = cxt.saved_tensors[2].to(device)
        num_avg = cxt.saved_tensors[3][0]
        pyfluidsynth = cxt.saved_tensors[4][0].to(device)

        list_of_grads = []
        delta_k= np.random.randint(0,high=2,size=synth_params.shape)
        delta_k[delta_k==0]=-1
        delta_k = torch.tensor(delta_k).to(device)
        
        # synth +- forward and backward passes
        updated_plus = torch.min(synth_params + epsilon*delta_k, torch.tensor([1.0]).to(device))
        updated_minus = torch.max(synth_params - epsilon*delta_k, torch.tensor([0.0]).to(device))


        if pyfluidsynth > torch.tensor([0.0]).to(device):
            J_plus, samplestwo = pyfluidsynth_sampler(updated_plus)
            J_minus, samplestwo = pyfluidsynth_sampler(updated_minus) 
            J_plus = get_melspec(J_plus, device)
            J_minus = get_melspec(J_minus, device)
        else:
            J_plus = sampler(updated_plus, samples)
            J_minus = sampler(updated_minus, samples) 
            J_plus = get_melspec(J_plus, device)
            J_minus = get_melspec(J_minus, device)
        grad_synth = J_plus - J_minus 

        #grad_synth = grad_synth.reshape(grad_synth.shape[0],-1)
        delta_k = delta_k.reshape(grad_synth.shape[0],-1) 
        

        # loop for each element of synth_params
        #pdb.set_trace()
        params_sublist = []
        for sub_p_idx in range(delta_k.shape[1]):
            grad_s = grad_synth / (2 * epsilon * torch.unsqueeze(torch.unsqueeze(delta_k[:,sub_p_idx],dim=1),dim=1))
            params_sublist.append(torch.unsqueeze(torch.sum(grad_output * grad_s, dim=[1,2]),dim=0)) #params_sublist.append(torch.unsqueeze(torch.sum(grad_output * grad_s, dim=1),dim=0))

        grad_synth = torch.cat(params_sublist,dim=0)
        grad_synth  = torch.unsqueeze(torch.unsqueeze(grad_synth.permute((1,0)),dim=1),dim=1).reshape(synth_params.shape)

        return grad_synth, None, None, None, None


class FunctionWithNumericalGrad(torch.autograd.Function):
    
    @staticmethod
    def forward(cxt, synth_params, samples, epsilon, num_avg,pyfluidsynth):
        cxt.save_for_backward(synth_params,samples, epsilon, num_avg,pyfluidsynth)
        # do forward process
        if pyfluidsynth.cuda() > torch.tensor([0.0]).cuda():
            synth_signal = pyfluidsynth_sampler(synth_params, samples)
        else:
            synth_signal = sampler(synth_params, samples)

        return synth_signal

    @staticmethod
    def backward(cxt, grad_output): 
        synth_params = cxt.saved_tensors[0]
        samples = cxt.saved_tensors[1].cuda()
        epsilon = cxt.saved_tensors[2].cuda()
        num_avg = cxt.saved_tensors[3][0]
        pyfluidsynth = cxt.saved_tensors[4][0].cuda()

        # generate random vector of +-1s, #delta_k = rademacher(synth_params.shape).numpy()
        list_of_grads = []
        for i in range(num_avg):
            delta_k= np.random.randint(0,high=2,size=synth_params.shape)
            delta_k[delta_k==0]=-1
            delta_k = torch.tensor(delta_k).cuda()
            
            # synth +- forward and backward passes
            updated_plus = torch.min(synth_params + epsilon*delta_k, torch.tensor([1.0]).cuda())
            updated_minus = torch.max(synth_params - epsilon*delta_k, torch.tensor([0.0]).cuda())
            if pyfluidsynth > torch.tensor([0.0]).cuda():
                J_plus = pyfluidsynth_sampler(updated_plus)
                J_minus = pyfluidsynth_sampler(updated_minus) 
            else:
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

        grad_synth = torch.mean(torch.cat(list_of_grads),axis=0).cuda()
        

        return grad_synth, None, None, None, None


class FunctionWithNumericalGradSTFTLoss(torch.autograd.Function):

    
    
    @staticmethod
    def forward(cxt, synth_params, samples, epsilon, num_avg,pyfluidsynth, groundtruth):
        
        # do forward process
        loss = torch.nn.MSELoss()
        cxt.save_for_backward(synth_params,samples, epsilon, num_avg,pyfluidsynth, groundtruth)
        device = synth_params.device
        if pyfluidsynth.to(device) > torch.tensor([0.0]).to(device):
            synth_signal = sampler(synth_params, samples)
            #synth_signal, samplestwo = pyfluidsynth_sampler(synth_params)
            
        else:
            
            synth_signal = sampler(synth_params, samples)

        log_mel_spec_concat = get_melspec(synth_signal, device)
        lossval = loss(log_mel_spec_concat, groundtruth)


        return lossval

    @staticmethod
    def backward(cxt, grad_output): 

        loss = torch.nn.MSELoss()
        
        synth_params = cxt.saved_tensors[0]
        device = synth_params.device
        samples = cxt.saved_tensors[1].to(device)
        epsilon = cxt.saved_tensors[2].to(device)
        num_avg = cxt.saved_tensors[3][0]
        pyfluidsynth = cxt.saved_tensors[4][0].to(device)
        groundtruth=cxt.saved_tensors[5].to(device)

        list_of_grads = []
        delta_k= np.random.randint(0,high=2,size=synth_params.shape)
        delta_k[delta_k==0]=-1
        delta_k = torch.tensor(delta_k).to(device)
        
        # synth +- forward and backward passes
        updated_plus = torch.min(synth_params + epsilon*delta_k, torch.tensor([1.0]).to(device))
        updated_minus = torch.max(synth_params - epsilon*delta_k, torch.tensor([0.0]).to(device))


        if pyfluidsynth > torch.tensor([0.0]).to(device):
            J_plus, samplestwo = pyfluidsynth_sampler(updated_plus)
            J_minus, samplestwo = pyfluidsynth_sampler(updated_minus) 
            J_plus = get_melspec(J_plus, device)
            J_minus = get_melspec(J_minus, device)
            lossvalplus = loss(J_plus, groundtruth)
            lossvalminus = loss(J_minus, groundtruth)
        else:
            J_plus = sampler(updated_plus, samples)
            J_minus = sampler(updated_minus, samples) 
            J_plus = get_melspec(J_plus, device)
            J_minus = get_melspec(J_minus, device)
            lossvalplus = loss(J_plus, groundtruth)
            lossvalminus = loss(J_minus, groundtruth)

        grad_synth = lossvalplus-lossvalminus

        #grad_synth = grad_synth.reshape(grad_synth.shape[0],-1)
        delta_k = delta_k.reshape(synth_params.shape[0],-1) 
        

        # loop for each element of synth_params
        #pdb.set_trace()
        params_sublist = []
        for sub_p_idx in range(delta_k.shape[1]):
            grad_s = grad_synth / (2 * epsilon * torch.unsqueeze(torch.unsqueeze(delta_k[:,sub_p_idx],dim=1),dim=1))
            params_sublist.append(torch.unsqueeze(torch.sum(grad_s, dim=[1,2]),dim=0)) #params_sublist.append(torch.unsqueeze(torch.sum(grad_output * grad_s, dim=1),dim=0))

        grad_synth = torch.cat(params_sublist,dim=0)
        grad_synth  = torch.unsqueeze(torch.unsqueeze(grad_synth.permute((1,0)),dim=1),dim=1).reshape(synth_params.shape)

        return grad_synth, None, None, None, None, None



def trainloss(cfg):

    num_notes= int(cfg.num_notes)
    num_avg=int(cfg.num_avg)
    pyfluidsynth=cfg.pyfluidsynth
    epsilon=cfg.epsilon
    recurrent= cfg.recurrent
    apply_gumbelsoftmax=cfg.apply_gumbelsoftmax
    learning_rate=cfg.learning_rate
    freq_version=cfg.freq_version
    reshape_version=cfg.reshape_version
    avgpool_version=cfg.avgpool_version
    apply_softmax =cfg.apply_softmax
    apply_sigmoid=cfg.apply_sigmoid
    real_gradient=cfg.real_gradient
    num_iterations=cfg.num_iterations



    if cfg.log_wandb:
        wandb.init(config=cfg)

    cfg = wandb.config
    

    if cfg.device == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device)


    ground_truth = torch.tensor([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,0],
        [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
        [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,0,0],
        [1,0,0,0],[1,0,0,0],[1,0,0,0]]).type(torch.cuda.DoubleTensor)
    
    if not os.path.isdir("reconstruction"):
        os.mkdir("reconstruction")
        
    
    if recurrent:
        model = modeleff.EfficientNet.from_name('efficientnet-b0',num_notes=1, freqversion=freq_version, avgpoolversion=avgpool_version, reshapeversion=reshape_version).to(device)
    else:
        model = modeleff.EfficientNet.from_name('efficientnet-b0',num_notes=num_notes,freqversion=freq_version, avgpoolversion=avgpool_version, reshapeversion=reshape_version).to(device)

    model = torch.nn.DataParallel(model)

    sigmoid = torch.nn.Sigmoid().to(device)
    softmax = torch.nn.Softmax(dim=2).to(device)
    batchsize=1
        
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss = torch.nn.MSELoss()
    current_loss = torch.tensor([10000.0]).to(device)

    for k in range(num_iterations): 
        print(k)
        
        optimizer.zero_grad()
        
        if recurrent: 
            list_of_audio=[]
            list_of_audio_gt=[]   
            list_of_controls=[] 

            for j in range(num_notes):
                thenote, samples = synthesize("../data/musicfiles/", torch.unsqueeze(torch.unsqueeze(ground_truth[j],dim=0),dim=0).repeat(batchsize,1,1),pyfluidsynth)
                list_of_audio_gt.append(thenote)
                log_mel_spec_note = get_melspec(thenote.type(torch.FloatTensor).to(device),device)

                controls = model(torch.unsqueeze(log_mel_spec_note,dim=1))
                if freq_version:
                    controls=controls.permute((0,3,1,2)) 
                else:
                    controls=controls.permute((0,3,2,1))
                if apply_gumbelsoftmax:
                    controls = torch.nn.functional.gumbel_softmax(10*controls, tau=0.00001, hard=True,dim=2).type(torch.cuda.DoubleTensor)
                if apply_sigmoid:
                    controls = sigmoid(controls).type(torch.cuda.DoubleTensor)
                if apply_softmax:
                    controls = softmax(controls).type(torch.cuda.DoubleTensor)

                if real_gradient:
                    curr_audio = sampler(controls, samples)
                else:
                    #curr_audio = FunctionWithNumericalGrad.apply(controls, samples, torch.tensor([epsilon],requires_grad=False), torch.tensor([num_avg],requires_grad=False),torch.tensor([pyfluidsynth],requires_grad=False))
                    curr_audio=[]
                list_of_audio.append(curr_audio)
                list_of_controls.append(controls)

            resynthesized_mary,samples = synthesize("../data/musicfiles/", torch.unsqueeze(ground_truth[:num_notes],dim=0).repeat(batchsize,1,1),pyfluidsynth)
            log_mel_spec_mary = get_melspec(resynthesized_mary.type(torch.FloatTensor).to(device),device)
            loss_val= FunctionWithNumericalGradSTFTLoss.apply(controls, samples, torch.tensor([epsilon],requires_grad=False), torch.tensor([num_avg],requires_grad=False),torch.tensor([pyfluidsynth],requires_grad=False),log_mel_spec_mary)
            #synthesized_audio = torch.cat(list_of_audio,dim=1)
            
            
            #log_mel_spec_concat = get_melspec(synthesized_audio, device)
            if cfg.log_wandb and k==0: 
                wandb.log({"ground-truth pitch-time":wandb.Table(data=ground_truth.detach().cpu().tolist(), columns=["47","44","42","40"])})
                wandb.log({"ground_truth audio": wandb.Audio(resynthesized_mary[0].detach().cpu().numpy(), caption="ground truth audio", sample_rate=44100)})
                wandb.log({"ground-truth melspec": wandb.Image(log_mel_spec_mary[0].cpu().detach().numpy(), caption="GT Melspec")})
        
        else: 
            resynthesized_mary,samples = synthesize("../data/musicfiles/", torch.unsqueeze(ground_truth[:num_notes],dim=0).repeat(batchsize,1,1),pyfluidsynth)
            log_mel_spec_mary = get_melspec(resynthesized_mary.type(torch.FloatTensor).to(device),device)
            controls = model(torch.unsqueeze(log_mel_spec_mary,dim=1)) #(4,1,16)
            
            if freq_version:
                controls=controls.permute((0,3,1,2))  
            else:
                controls = controls.reshape(controls.shape[0],4,1,num_notes)
                controls=controls.permute((0,3,2,1))
            if apply_gumbelsoftmax:
                    controls = torch.nn.functional.gumbel_softmax(10*controls, tau=0.000001, hard=True,dim=2).type(torch.cuda.DoubleTensor)
            if apply_sigmoid:
                controls = sigmoid(controls).type(torch.cuda.DoubleTensor)
            if apply_softmax:
                controls = softmax(controls).type(torch.cuda.DoubleTensor)

            if real_gradient:
                #pdb.set_trace()
                synthesized_audio = sampler(controls, samples)
            else:
                synthesized_audio = FunctionWithNumericalGrad.apply(controls, samples, torch.tensor([epsilon]), torch.tensor([num_avg]),torch.tensor([pyfluidsynth],requires_grad=False))


            log_mel_spec_concat = get_melspec(synthesized_audio, device)
            loss_val = loss(log_mel_spec_concat,log_mel_spec_mary)
        

        if cfg.log_wandb and k%20==0:
                wandb.log({"predicted pitch-time":wandb.Table(data=controls[0,:,0,:].detach().cpu().tolist(), columns=["47","44","42","40"])})
                wandb.log({"difference pitch-time":wandb.Table(data=(ground_truth-controls[0,:,0,:]).detach().cpu().tolist(), columns=["47","44","42","40"])})
                #wandb.log({"reconstructed audio": wandb.Audio(synthesized_audio[0].detach().cpu().numpy(), caption="reconstructed audio " + str(k), sample_rate=44100)})
                #wandb.log({"predicted melspec": wandb.Image(log_mel_spec_concat[0].cpu().detach().numpy(), caption="Predicted Melspec  " + str(k))})
                #wandb.log({"difference melspec": wandb.Image(log_mel_spec_concat[0].cpu().detach().numpy()-log_mel_spec_mary[0].cpu().detach().numpy(), caption="Predicted Melspec  " + str(k))})
        
           

        loss_val.backward()
        optimizer.step()
        
        if cfg.log_wandb: 
            wandb.log({"reconstruction loss" : loss_val.detach().cpu().numpy(),"iteration": k})

        print("Loss is: ", loss_val)

        """if loss_val < current_loss: 
            print("Saved")
            sf.write("reconstruction/resynthesized_mary.wav", resynthesized_mary.permute(1,0).cpu().detach().numpy(), 44100)
            sf.write("reconstruction/synthesized_audio.wav", synthesized_audio.permute(1,0).cpu().detach().numpy(), 44100)
            current_loss = loss_val"""

def train(cfg: DictConfig):

    os.chdir(utils.get_original_cwd())
  
    if cfg.log_wandb:
        wandb.init(config=OmegaConf.to_container(cfg))
    
    num_notes= int(cfg.num_notes)
    num_avg=int(cfg.num_avg)
    pyfluidsynth=cfg.pyfluidsynth
    epsilon=cfg.epsilon
    recurrent= cfg.recurrent
    apply_gumbelsoftmax=cfg.apply_gumbelsoftmax
    learning_rate=cfg.learning_rate
    recurrent=cfg.recurrent
    freq_version=cfg.freq_version
    reshape_version=cfg.reshape_version
    avgpool_version=cfg.avgpool_version
    apply_softmax =cfg.apply_softmax
    apply_sigmoid=cfg.apply_sigmoid
    real_gradient=cfg.real_gradient
    num_iterations=cfg.num_iterations

    if cfg.device == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device)


    ground_truth = torch.tensor([[0,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,0],
        [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
        [0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0],[1,0,0,0],
        [1,0,0,0],[1,0,0,0],[1,0,0,0]]).type(torch.cuda.DoubleTensor)
    
    if not os.path.isdir("reconstruction"):
        os.mkdir("reconstruction")
        
    
    if recurrent:
        model = modeleff.EfficientNet.from_name('efficientnet-b0',num_notes=1, freqversion=freq_version, avgpoolversion=avgpool_version, reshapeversion=reshape_version).to(device)
    else:
        model = modeleff.EfficientNet.from_name('efficientnet-b0',num_notes=num_notes,freqversion=freq_version, avgpoolversion=avgpool_version, reshapeversion=reshape_version).to(device)

    model = torch.nn.DataParallel(model)

    sigmoid = torch.nn.Sigmoid().to(device)
    softmax = torch.nn.Softmax(dim=2).to(device)
    batchsize=8
        
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss = torch.nn.MSELoss()
    current_loss = torch.tensor([10000.0]).to(device)

    for k in range(num_iterations): 
        print(k)
        
        optimizer.zero_grad()
        
        if recurrent: 
            list_of_audio=[]
            list_of_audio_gt=[]   
            list_of_controls=[]   
            list_of_samples=[]     
            for j in range(num_notes):
                thenote, samples = synthesize("../data/musicfiles/", torch.unsqueeze(torch.unsqueeze(ground_truth[j],dim=0),dim=0).repeat(batchsize,1,1),pyfluidsynth)
                list_of_audio_gt.append(thenote)
                log_mel_spec_note = get_melspec(thenote.type(torch.FloatTensor).to(device),device)

                controls = model(torch.unsqueeze(log_mel_spec_note,dim=1))
                if freq_version:
                    controls=controls.permute((0,3,1,2)) 
                else:
                    controls=controls.permute((0,3,2,1))
                if apply_gumbelsoftmax:
                    controls = torch.nn.functional.gumbel_softmax(10*controls, tau=0.00001, hard=True,dim=2).type(torch.cuda.DoubleTensor)
                if apply_sigmoid:
                    controls = sigmoid(controls).type(torch.cuda.DoubleTensor)
                if apply_softmax:
                    controls = softmax(controls).type(torch.cuda.DoubleTensor)

                if real_gradient:
                    curr_audio = sampler(controls, samples)
                else:
                    #curr_audio = FunctionWithNumericalGrad.apply(controls, samples, torch.tensor([epsilon],requires_grad=False), torch.tensor([num_avg],requires_grad=False),torch.tensor([pyfluidsynth],requires_grad=False))
                    curr_audio=[]
                list_of_audio.append(curr_audio)
                list_of_samples.append(samples)
                list_of_controls.append(controls)
                
            controls = torch.cat(list_of_controls,dim=1)
            samples = torch.cat(list_of_samples,dim=1) #(16,1,4)
            pdb.set_trace()
            _, samples = synthesize("../data/musicfiles/", torch.unsqueeze(ground_truth,dim=0).repeat(batchsize,1,1),pyfluidsynth)
            log_mel_spec_concat= FunctionWithNumericalGrad.apply(controls, samples, torch.tensor([epsilon],requires_grad=False), torch.tensor([num_avg],requires_grad=False),torch.tensor([pyfluidsynth],requires_grad=False))
            #synthesized_audio = torch.cat(list_of_audio,dim=1)
            synthesized_audio_gt = torch.cat(list_of_audio_gt,dim=1) 
            log_mel_spec_mary = get_melspec(synthesized_audio_gt, device)
            #log_mel_spec_concat = get_melspec(synthesized_audio, device)
            if cfg.log_wandb and k==0: 
                wandb.log({"ground-truth pitch-time":wandb.Table(data=ground_truth.detach().cpu().tolist(), columns=["47","44","42","40"])})
                wandb.log({"ground_truth audio": wandb.Audio(synthesized_audio_gt[0].detach().cpu().numpy(), caption="ground truth audio", sample_rate=44100)})
                wandb.log({"ground-truth melspec": wandb.Image(log_mel_spec_mary[0].cpu().detach().numpy(), caption="GT Melspec")})
        
        else: 
            resynthesized_mary,samples = synthesize("../data/musicfiles/", torch.unsqueeze(ground_truth[:num_notes],dim=0).repeat(batchsize,1,1),pyfluidsynth)
            log_mel_spec_mary = get_melspec(resynthesized_mary.type(torch.FloatTensor).to(device),device)
            controls = model(torch.unsqueeze(log_mel_spec_mary,dim=1)) #(4,1,16)
            
            if freq_version:
                controls=controls.permute((0,3,1,2))  
            else:
                controls = controls.reshape(controls.shape[0],4,1,num_notes)
                controls=controls.permute((0,3,2,1))
            if apply_gumbelsoftmax:
                    controls = torch.nn.functional.gumbel_softmax(10*controls, tau=0.000001, hard=True,dim=2).type(torch.cuda.DoubleTensor)
            if apply_sigmoid:
                controls = sigmoid(controls).type(torch.cuda.DoubleTensor)
            if apply_softmax:
                controls = softmax(controls).type(torch.cuda.DoubleTensor)

            if real_gradient:
                #pdb.set_trace()
                synthesized_audio = sampler(controls, samples)
            else:
                synthesized_audio = FunctionWithNumericalGrad.apply(controls, samples, torch.tensor([epsilon]), torch.tensor([num_avg]),torch.tensor([pyfluidsynth],requires_grad=False))


            log_mel_spec_concat = get_melspec(synthesized_audio, device)
        

        if cfg.log_wandb and k%20==0:
                wandb.log({"predicted pitch-time":wandb.Table(data=controls[0,:,0,:].detach().cpu().tolist(), columns=["47","44","42","40"])})
                wandb.log({"difference pitch-time":wandb.Table(data=(ground_truth-controls[0,:,0,:]).detach().cpu().tolist(), columns=["47","44","42","40"])})
                #wandb.log({"reconstructed audio": wandb.Audio(synthesized_audio[0].detach().cpu().numpy(), caption="reconstructed audio " + str(k), sample_rate=44100)})
                wandb.log({"predicted melspec": wandb.Image(log_mel_spec_concat[0].cpu().detach().numpy(), caption="Predicted Melspec  " + str(k))})
                wandb.log({"difference melspec": wandb.Image(log_mel_spec_concat[0].cpu().detach().numpy()-log_mel_spec_mary[0].cpu().detach().numpy(), caption="Predicted Melspec  " + str(k))})
        
        loss_val = loss(log_mel_spec_concat,log_mel_spec_mary)
        loss_val.backward()
        optimizer.step()
        
        if cfg.log_wandb: 
            wandb.log({"reconstruction loss" : loss_val.detach().cpu().numpy(),"iteration": k})

        print("Loss is: ", loss_val)

        """if loss_val < current_loss: 
            print("Saved")
            sf.write("reconstruction/resynthesized_mary.wav", resynthesized_mary.permute(1,0).cpu().detach().numpy(), 44100)
            sf.write("reconstruction/synthesized_audio.wav", synthesized_audio.permute(1,0).cpu().detach().numpy(), 44100)
            current_loss = loss_val"""

#test(1)