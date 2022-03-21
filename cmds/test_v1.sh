# BVH, Mar 2022.

# v6.

python eval/test.py --resume v6 --name v6-st01503 --data_path /proj/vondrick2/datasets/slakh2100_flac_redux/validation/Track01503/MIDI/S01.mid --data_clip_mode sequential --gpu_id 1 --batch_size 8

python eval/test.py --resume v6 --name v6-st01504 --data_path /proj/vondrick2/datasets/slakh2100_flac_redux/validation/Track01504/MIDI/S00.mid --data_clip_mode sequential --gpu_id 2 --batch_size 8

python eval/test.py --resume v6 --name v6-st01505 --data_path /proj/vondrick2/datasets/slakh2100_flac_redux/validation/Track01505/MIDI/S02.mid --data_clip_mode sequential --gpu_id 3 --batch_size 8
