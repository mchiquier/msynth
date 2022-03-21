# BVH, Mar 2022.

# v1.
CUDA_VISIBLE_DEVICES=6,7 python train.py --name v1 --batch_size 96

# v2: increase random melody dset size, num_epochs 50 to 100, optimizer adam to adamw, n_fft 256 to 512.
# => "swapping is possible" errors on many machines?
# => stopped short because of incorrect checkpointing.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --name v2 --resume v2 --batch_size 56

# v3: same but with audio logging and correct resume.
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --name v3 --resume v3 --num_epochs 100 --batch_size 32

# v4: waveform_loss_mode stft_mag to samples.
# NOTE: added * 16.0 AFTER this because of tiny loss values.
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --name v4 --batch_size 32 --waveform_loss_mode samples

# v5: num_epochs 100 to 80, waveform_loss_mode samples to stft_real_imag.
# NOTE: upon resume, different default pitch_bounds (D3-D5 to F3-F5).
# NOTE: v3, v4, v5 all show sound artefacts.
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --name v5 --resume v5 --batch_size 32 --waveform_loss_mode stft_real_imag

# v6: calculate + visualize cycles, adjust pitch_bounds again (C3-C6), add random chords (simple simultaneous starts), num_epochs 80 to 70.
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --name v6 --resume v6 --batch_size 8 --waveform_loss_mode stft_real_imag

# v7: waveform_loss_mode stft_real_imag to samples.
CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py --name v7 --resume v7 --batch_size 8 --waveform_loss_mode samples
