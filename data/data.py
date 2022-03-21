'''
Data loading and processing logic.
Created by Basile Van Hoorick, Feb 2022.
'''

from __init__ import *

# Internal imports.
import daw_synth
import procedural
import real
import utils


def _seed_worker(worker_id):
    '''
    Ensures that every data loader worker has a separate seed with respect to NumPy and Python
    function calls, not just within the torch framework. This is very important as it sidesteps
    lack of randomness- and augmentation-related bugs.
    '''
    worker_seed = torch.initial_seed() % (2 ** 32)  # This is distinct for every worker.
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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

        train_dataset = procedural.MyRandomMelodyDataset(
            logger, 'train', **dset_args)
        val_aug_dataset = procedural.MyRandomMelodyDataset(
            logger, 'val', **dset_args)
        val_noaug_dataset = procedural.MyRandomMelodyDataset(
            logger, 'val', **dset_args) \
            if args.do_val_noaug else None

    else:

        train_dataset = real.MyMIDIFileDataset(
            args.data_path, logger, 'train', **dset_args)
        val_aug_dataset = real.MyMIDIFileDataset(
            args.data_path, logger, 'val', **dset_args)
        val_noaug_dataset = real.MyMIDIFileDataset(
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

    # Load options that can differ between train and evaluation time.
    test_dset_args['data_clip_mode'] = test_args.data_clip_mode
    test_dset_args['clip_interval_ms'] = int(test_args.clip_interval * 1000.0)

    if test_args.data_path == 'random_melody':

        test_dataset = procedural.MyRandomMelodyDataset(
            logger, 'test', **test_dset_args)

    else:

        test_dataset = real.MyMIDIFileDataset(
            test_args.data_path, logger, 'test', **test_dset_args)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_args.batch_size, num_workers=test_args.num_workers,
        shuffle=False, worker_init_fn=_seed_worker, drop_last=True, pin_memory=False)

    return (test_loader, test_dset_args)
