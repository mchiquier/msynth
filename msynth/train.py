'''
Training + validation oversight and recipe configuration.
Created by Basile Van Hoorick, Feb 2022.
'''

from __init__ import *

# Library imports.
import torch_optimizer

# Internal imports.
import argscode as args


import dataloader.data as data
import loss_single as loss
import models.perceiverio as model
import msynth.utils.logvis as logvis
import train_pipeline_single as pipeline
import utils as utils
import pdb

def _get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def _train_one_epoch(args, train_pipeline, phase, epoch, optimizer,
                     lr_scheduler, data_loader, device, logger):
    assert phase in ['train', 'val', 'val_aug', 'val_noaug']

    log_str = f'Epoch (1-based): {epoch + 1} / {args.num_epochs}'
    logger.info()
    logger.info('=' * len(log_str))
    logger.info(log_str)
    if phase == 'train':
        logger.info(f'===> Train ({phase})')
        logger.report_scalar(phase + '/learn_rate', _get_learning_rate(optimizer), step=epoch)
    else:
        logger.info(f'===> Validation ({phase})')

    train_pipeline[1].set_phase(phase)

    steps_per_epoch = len(data_loader)
    total_step_base = steps_per_epoch * epoch  # This has already happened so far.
    start_time = time.time()
    num_exceptions = 0

    for cur_step, data_retval in enumerate(tqdm.tqdm(data_loader)):

        if cur_step == 0:
            logger.info(f'Enter first data loader iteration took {time.time() - start_time:.3f}s')

        total_step = cur_step + total_step_base  # For continuity in wandb.
        
        if args.is_debug:
            
            loss_retval = train_pipeline[1].process_entire_batch(
                data_retval, model_retval, loss_retval, cur_step, total_step)
            total_loss = loss_retval['total']

        else:
            try:
                # First, address every example independently.
                # This part has zero interaction between any pair of GPUs.
                (model_retval, loss_retval) = train_pipeline[0](data_retval, cur_step, total_step)
                if cur_step%500==0:
                    #.set_trace()
                    logger.save_audio(data_retval['daw_waveform'][0].cpu().detach().numpy(), sample_rate=22050, step=total_step, file_name="audio" + str(data_retval['control_matrix'][0,0,:].detach().cpu()), online_name="audio" + str(data_retval['control_matrix'][0,0,:].detach().cpu()))
                    logger.save_image(data_retval['mel_spec'][0].cpu().detach().numpy(), step=total_step, file_name="melspec" + str(data_retval['control_matrix'][0,0,:].detach().cpu()) + ".png", online_name="melspec" + str(data_retval['control_matrix'][0,0,:].detach().cpu()))

                # Second, process accumulated information, for example contrastive loss functionality.
                # This part typically happens on the first GPU, so it should be kept minimal in memory.
                loss_retval = train_pipeline[1].process_entire_batch(
                    data_retval, model_retval, loss_retval, cur_step, total_step)
                total_loss = loss_retval['total'] #loss_retval['total']
                print(total_loss)

            except Exception as e:

                num_exceptions += 1
                if num_exceptions >= 7:
                    raise e
                else:
                    logger.exception(e)
                    continue

        # Perform backpropagation to update model parameters.
        if phase == 'train':

            optimizer.zero_grad()
            total_loss.backward()
            logger.report_scalar(phase + '/loss', total_loss.detach().cpu(), step=total_step)
            logger.report_scalar(phase + '/gt_fc', data_retval['control_matrix'][0,0,:].detach().cpu(), step=total_step)
            logger.report_scalar(phase + '/pred_fc', model_retval['analysis_output'][0,0,:].detach().cpu(), step=total_step)

            # Apply gradient clipping if desired.
            if args.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(train_pipeline[0].parameters(), args.gradient_clip)

            optimizer.step()

        # Print and visualize stuff.
        """logger.handle_train_step(epoch, phase, cur_step, total_step, steps_per_epoch,
                                 data_retval, model_retval, loss_retval, args)"""

        # DEBUG:
        if cur_step >= 200 and args.is_debug:
            logger.warning('Cutting epoch short for debugging...')
            break

    #if phase == 'train':
    #    lr_scheduler.step()


def _train_all_epochs(args, train_pipeline, optimizer, lr_scheduler, start_epoch, train_loader,
                      val_aug_loader, val_noaug_loader, device, logger, checkpoint_fn):

    logger.info('Start training loop...')
    start_time = time.time()
    for epoch in range(start_epoch, args.num_epochs):

        # Training.
        _train_one_epoch(
            args, train_pipeline, 'train', epoch, optimizer,
            lr_scheduler, train_loader, device, logger)

        # Save model weights.
        #checkpoint_fn(epoch)

        if epoch % args.val_every == 0:

           """ # Validation with data augmentation.
            _train_one_epoch(
                args, train_pipeline, 'val_aug', epoch, optimizer,
                lr_scheduler, val_aug_loader, device, logger)

            # Validation without data augmentation.
            if args.do_val_noaug:
                _train_one_epoch(
                    args, train_pipeline, 'val_noaug', epoch, optimizer,
                    lr_scheduler, val_noaug_loader, device, logger)"""

        logger.epoch_finished(epoch)

        # TODO: Optionally, keep track of best weights.

    total_time = time.time() - start_time
    logger.info(f'Total time: {total_time / 3600.0:.3f} hours')


def main(args, logger):

    logger.info()
    logger.info('torch version: ' + str(torch.__version__))
    logger.info('torchvision version: ' + str(torchvision.__version__))
    logger.save_args(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    logger.info('Checkpoint path: ' + args.checkpoint_path)
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Instantiate datasets.
    logger.info('Initializing data loaders...')
    start_time = time.time()
    (train_loader, val_aug_loader, val_noaug_loader, dset_args) = \
        data.create_train_val_data_loaders(args, logger)
    logger.info(f'Took {time.time() - start_time:.3f}s')

    logger.info('Initializing model...')
    start_time = time.time()

    # Instantiate networks.
    """model_args = dict(
        num_samples=int(args.duration * args.sample_rate),
        num_controls=args.num_controls,
        num_timecells= args.num_timecells,
        samples_per_frame=args.samples_per_frame,
        output_pos_enc=args.output_pos_enc,
    )"""
    model_args = dict(
        num_input_freq=442,
        num_input_timebin=101,
        num_controls=args.num_controls,
        num_timecells= args.num_timecells,
        samples_per_frame=args.samples_per_frame,
        output_pos_enc=args.output_pos_enc,
    )
    #analysis_net = model.MyAnalysisModel(logger, **model_args)
    analysis_net = model.MySpecAnalysisModel(logger, **model_args)
    # Bundle networks into a list.
    networks = [analysis_net]
    for i in range(len(networks)):
        networks[i] = networks[i].to(device)
    networks_nodp = [net for net in networks]

    

    # Instantiate optimizer & learning rate scheduler.
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(networks[0].parameters(), lr=args.learn_rate)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(networks[0].parameters(), lr=args.learn_rate)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(networks[0].parameters(), lr=args.learn_rate)
    elif args.optimizer == 'lamb':
        optimizer = torch_optimizer.Lamb(networks[0].parameters(), lr=args.learn_rate)\

    # Instantiate encompassing pipeline for more efficient parallelization.
    train_pipeline = pipeline.MyTrainPipeline(args, logger, networks, optimizer, device)
    train_pipeline = train_pipeline.to(device)
    train_pipeline_nodp = train_pipeline
    if args.device == 'cuda':
        train_pipeline = torch.nn.DataParallel(train_pipeline)
    milestones = [(args.num_epochs * 2) // 5,
                  (args.num_epochs * 3) // 5,
                  (args.num_epochs * 4) // 5]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=args.lr_decay)

    # Load weights from checkpoint if specified.
    if args.resume:
        logger.info('Loading weights from: ' + args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        networks_nodp[0].load_state_dict(checkpoint['analysis_net'])
        networks_nodp[1].load_state_dict(checkpoint['synthesis_net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    logger.info(f'Took {time.time() - start_time:.3f}s')

    # Define logic for how to store checkpoints.
    def save_model_checkpoint(epoch):
        if args.checkpoint_path:
            logger.info(f'Saving model checkpoint to {args.checkpoint_path}...')
            checkpoint = {
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'train_args': args,
                'dset_args': dset_args,
                'model_args': model_args,
            }
            checkpoint['analysis_net'] = networks_nodp[0].state_dict()
            checkpoint['synthesis_net'] = networks_nodp[1].state_dict()
            if epoch % args.checkpoint_interval == 0:
                torch.save(checkpoint,
                           os.path.join(args.checkpoint_path, 'model_{}.pth'.format(epoch)))
            torch.save(checkpoint,
                       os.path.join(args.checkpoint_path, 'checkpoint.pth'))
            logger.info()

    logger.init_wandb("msynth", args, networks)

    # Print train arguments.
    logger.info('Final train command args: ' + str(args))
    logger.info('Final train dataset args: ' + str(dset_args))

    # Start training loop.
    _train_all_epochs(
        args, (train_pipeline, train_pipeline_nodp), optimizer, lr_scheduler, start_epoch,
        train_loader, val_aug_loader, val_noaug_loader, device, logger, save_model_checkpoint)




# DEBUG / WARNING: This is slow, but we can detect NaNs this way:
# torch.autograd.set_detect_anomaly(True)

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

# https://github.com/pytorch/pytorch/issues/11201
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

args = args.train_args()

logger = logvis.MyLogger(args, context='train')

if args.is_debug:

    main(args, logger)

else:

    try:
        main(args, logger)

    except Exception as e:

        logger.exception(e)

        logger.warning('Shutting down due to exception...')
