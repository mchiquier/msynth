
import msynth.utils.daw_synth as daw_synth
import torch
import random
import numpy as np
import copy
import torch.nn.functional as F
import os
import librosa
from msynth.models.fmsynth import FMNet, Synth, FMNetNormalized, SynthNormalized
import pdb

def _seed_worker(worker_id):
    '''
    Ensures that every data loader worker has a separate seed with respect to NumPy and Python
    function calls, not just within the torch framework. This is very important as it sidesteps
    lack of randomness- and augmentation-related bugs.
    '''
    worker_seed = torch.initial_seed() % (2 ** 32)  # This is distinct for every worker.
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def random_tp_matrix(duration_ms, timecell_ms, pitch_bounds):
    '''
    :param duration_ms (int): Total duration of the generated melody.
    :param timecell_ms (int): Duration of one timecell (i.e. element in the horizontal axis) [ms].
    :param pitch_bounds ((int, int) tuple): pitch_min and pitch_max.
    :return tp_matrix (P, T) numpy array: Timecell / frequency binary description of melody.
    '''
    P = 128
    chord_prob = 0.2

    T = duration_ms // timecell_ms
    tp_matrix = np.zeros((P, T), dtype=np.uint8)
    num_notes = np.random.randint(2, 9)
    existing_starts = set()

    for i in range(num_notes):
        # Add random simultaneous starts of multiple notes.
        found = False
        while not found:
            if i > 0 and np.random.rand() < chord_prob:
                start = np.random.choice(list(existing_starts))
                if start + 1 < T - 2:
                    found = True
                    end = np.random.randint(start + 1, T - 2)
            else:
                start, end = np.random.randint(1, T - 2, 2)
                found = True

        start, end = min(start, end), max(start, end) + 1
        pitch = np.random.randint(pitch_bounds[0], pitch_bounds[1] + 1)

        for t in range(start, end):
            tp_matrix[pitch, t] = 1

        existing_starts.add(start)

    return tp_matrix


def create_train_val_data_loaders(args, logger):
    '''
    return (train_loader, val_aug_loader, val_noaug_loader, dset_args).
    '''

    # TODO: Figure out noaug val dataset args as well.
    dset_args = dict()
    dset_args['soundfont_path'] = args.soundfont_path
    dset_args['sample_rate'] = args.sample_rate
    dset_args['duration_ms'] = int(args.duration * 1000.0)
    dset_args['timecell_ms'] = args.timecell_ms
    dset_args['pitch_bounds'] = args.pitch_bounds
    dset_args['data_clip_mode'] = args.data_clip_mode
    dset_args['clip_interval_ms'] = int(args.clip_interval * 1000.0)

    if args.data_path == 'random_melody':

        train_dataset = MyRandomMelodyDataset(
            logger, 'train', **dset_args)
        val_aug_dataset = MyRandomMelodyDataset(
            logger, 'val', **dset_args)
        val_noaug_dataset = MyRandomMelodyDataset(
            logger, 'val', **dset_args) \
            if args.do_val_noaug else None
    elif args.data_path == 'random_fm':
        train_dataset = MyRandomFMDataset(
            logger, 'train')
        val_aug_dataset = MyRandomFMDataset(
            logger, 'val')
        val_noaug_dataset = MyRandomFMDataset(
            logger, 'val', **dset_args) \
            if args.do_val_noaug else None
    else:

        train_dataset = MyMIDIFileDataset(
            args.data_path, logger, 'train', **dset_args)
        val_aug_dataset = MyRandomMelodyDataset(
            args.data_path, logger, 'val', **dset_args)
        val_noaug_dataset = MyRandomMelodyDataset(
            args.data_path, logger, 'val', **dset_args) \
            if args.do_val_noaug else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
    val_aug_loader = torch.utils.data.DataLoader(
        val_aug_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)
    val_noaug_loader = torch.utils.data.DataLoader(
        val_noaug_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False) \
        if args.do_val_noaug else None


    return (train_loader, val_aug_loader, val_noaug_loader, dset_args)


def create_test_data_loader(train_args, test_args, train_dset_args, logger):
    '''
    return (test_loader, test_dset_args).
    '''

    test_dset_args = copy.deepcopy(train_dset_args)

    if test_args.data_path == 'random_melody':

        test_dataset = MyRandomMelodyDataset(
            logger, 'test')

    elif test_args.data_path == 'random_fm':

        test_dataset = MyRandomFMDataset(
            logger, 'test', **test_dset_args)
    
    else:

        test_dataset = MyMIDIFileDataset(
            test_args.data_path, logger, 'test', **test_dset_args)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_args.batch_size, num_workers=test_args.num_workers,
        shuffle=True, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)

    return (test_loader, test_dset_args)

class MyRandomFMDataset(torch.utils.data.Dataset):
    '''
    X
    '''

    def __init__(self, logger, phase, sample_rate=44100, duration_ms=3000,
                 timecell_ms=50, pitch_bounds=[69 - 16, 69 + 8]):
        '''
        :param logger (MyLogger).
        :param phase (str): train / val_aug / val_noaug / test.
        :param soundfont_path (str).
        :param sample_rate (int): Sample rate [Hz].
        :param duration_ms (int).
        :param timecell_ms (int): Duration of one timecell (i.e. element in the horizontal axis).
        :param pitch_bounds ((int, int) tuple): pitch_min and pitch_max.
        '''
        self.logger = logger
        self.phase = phase

        # Precisely fixed number to avoid model errors; trim or pad data with zeros whenever needed.
        desired_samples = int(duration_ms / 1000.0 * sample_rate)

        self.sample_rate = sample_rate
        self.duration_ms = duration_ms
        self.timecell_ms = timecell_ms
        self.pitch_bounds = pitch_bounds
        self.desired_samples = desired_samples

        #CLASSIFICATION

        n_freqs=1
        which_mod=0

        self.carrier_hz = librosa.midi_to_hz(np.arange(1,n_freqs+1))
        self.carrier_fq =torch.tensor(self.carrier_hz).to(torch.float32)
        self.carrier_weight =torch.zeros(self.carrier_hz.shape)
        self.carrier_weight[which_mod]=1.0
        self.carrier_weight = F.softmax(self.carrier_weight)

        self.mod_hz = librosa.midi_to_hz(np.arange(1,n_freqs+1))
        self.mod_fq =torch.tensor(self.mod_hz).to(torch.float32)
        self.mod_weight =torch.zeros(self.carrier_hz.shape)
        self.mod_weight[which_mod]=1.0
        self.mod_weight = F.softmax(self.mod_weight)

        self.carrier_env_slope = torch.rand(n_freqs)
        self.carrier_env_offset = torch.rand(n_freqs)
        
        self.mod_env_offset = torch.rand(n_freqs)
        self.mod_env_slope = torch.rand(n_freqs)
        self.phase_offset=  torch.tensor([0.0])

        #REGRESSION
    
        self.carrier_env_slope = torch.rand(n_freqs)
        self.carrier_env_offset = torch.rand(n_freqs)
        
        self.mod_env_offset = torch.rand(n_freqs)
        self.mod_env_slope = torch.rand(n_freqs)
        self.phase_offset=  torch.tensor([0.0])
        self.mod_note = torch.rand(1)
        self.mod_fq = 0.0 #440.0 * (2.0 ** ((128*self.mod_note - 69.0) / 12.0)) 
        
    def __len__(self):
        return 10000

    def get_melspec(self,audio, device):
        mel_basis = librosa.filters.mel(22050,n_fft=441*4,n_mels=128)
        mel_basis = torch.from_numpy(mel_basis).float().to(device)
        stft = torch.stft(audio,n_fft=441*4,hop_length=441*2)
        real_part, imag_part = stft.unbind(-1)
        magnitude_concat = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(mel_basis, magnitude_concat.float())
        log_mel_spec_concat = torch.log10(torch.clamp(mel_output, min=1e-5))#(1,128,529)
        return log_mel_spec_concat

    def __getitem__(self, index):

        self.carrier_note = torch.rand(1)
        
        self.carrier_fq = self.carrier_note*(22050.0/2.0)#torch.tensor([librosa.midi_to_hz(np.array([self.carrier_note*128]))]).to(torch.float32) #self.carrier_note*22000 #
        fmnet = FMNetNormalized(self.carrier_fq)
        synth = SynthNormalized('cpu')
        sr=22050#self.sample_rate
        carrier_stereo_detune=0.005
        mod_stereo_detune=0.01
        dur=0.1
        variables=(dur,self.carrier_weight, self.mod_weight,self.carrier_env_slope, self.carrier_env_offset,self.mod_env_slope,self.mod_env_offset,self.phase_offset)
        
        output = synth.make_output(fmnet, variables, sr, carrier_stereo_detune, mod_stereo_detune)[:,0]
        # Return results."""
        thelist=[]
        #self.carrier_env_slope, self.carrier_env_offset,self.mod_env_slope,self.mod_env_offset,self.phase_offset, 
        for param in [self.carrier_note]:
            param = np.expand_dims(param,axis=0)
            thelist.append(param)
        control_matrix = np.concatenate(thelist,axis=1)
        data_retval = dict()
        data_retval['control_matrix'] = control_matrix.astype(np.float32).T
        # data_retval['event_seq'] = event_seq
        data_retval['daw_waveform'] = (output.astype(np.float32)+1.0)/2.0

        """stft = torch.stft(torch.tensor(output),n_fft=128*2,hop_length=128)
        real_part, imag_part = stft.unbind(-1)
        magnitude_concat = torch.sqrt(real_part ** 2 + imag_part ** 2)
        magnitude_concat = magnitude_concat - torch.min(magnitude_concat)
        magnitude_concat = magnitude_concat/torch.max(magnitude_concat)
        data_retval['mel_spec'] = magnitude_concat.numpy()"""

        melspec = self.get_melspec(torch.tensor(output),'cpu')
        print(melspec.shape)
        melspec = melspec - torch.min(melspec)
        melspec = melspec/torch.max(melspec)
        data_retval['mel_spec'] = melspec #(torch.ones(melspec.shape)*self.carrier_note).numpy() #"""mellspec #(

        return data_retval

