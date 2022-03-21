'''
Evaluation tools.
'''

from __init__ import *

# Internal imports.
import args
import data
import daw_synth
import logvis
import loss
import model
import pipeline
import utils


def load_networks(checkpoint_path, device, logger, epoch=-1):
    '''
    :param checkpoint_path (str): Path to model checkpoint folder or file.
    :param epoch (int): If >= 0, desired checkpoint epoch to load.
    :return (networks, train_args, dset_args, model_args, epoch).
        networks (list of modules).
        train_args (dict).
        train_dset_args (dict).
        model_args (dict).
        epoch (int).
    '''
    print_fn = logger.info if logger is not None else print
    assert os.path.exists(checkpoint_path)
    if os.path.isdir(checkpoint_path):
        model_fn = f'model_{epoch}.pth' if epoch >= 0 else 'checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_path, model_fn)

    print_fn('Loading weights from: ' + checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Display all arguments to help verify correctness.
    train_args = checkpoint['train_args']
    train_dset_args = checkpoint['dset_args']
    print_fn('Train command args: ' + str(train_args))
    print_fn('Train dataset args: ' + str(train_dset_args))

    # Get network instance parameters.
    model_args = checkpoint['model_args']
    print_fn('My model args: ' + str(model_args))

    # Instantiate networks.
    analysis_net = model.MyAnalysisModel(logger, **model_args)
    synthesis_net = model.MySynthesisModel(logger, **model_args)
    analysis_net = analysis_net.to(device)
    synthesis_net = synthesis_net.to(device)
    analysis_net.load_state_dict(checkpoint['analysis_net'])
    synthesis_net.load_state_dict(checkpoint['synthesis_net'])
    networks = [analysis_net, synthesis_net]

    epoch = checkpoint['epoch']
    print_fn('=> Loaded epoch (1-based): ' + str(epoch + 1))

    return (networks, train_args, train_dset_args, model_args, epoch)


def perform_inference(data_retval, networks, device, logger, all_args):
    '''
    Generates test time predictions.
    :param all_args (dict): train, test, train_dset, test_dest, model.
    '''
    print_fn = logger.info if logger is not None else print
    soundfont_path = all_args['train'].soundfont_path
    sample_rate = all_args['train'].sample_rate
    timecell_ms = all_args['train'].timecell_ms

    # Prepare model.
    analysis_net = networks[0]
    synthesis_net = networks[1]

    # Prepare data.
    control_matrix = data_retval['control_matrix']
    daw_waveform = data_retval['daw_waveform']
    control_matrix = control_matrix.to(device)
    daw_waveform = daw_waveform.to(device)
    (B, P, T) = control_matrix.shape
    (B, S) = daw_waveform.shape

    # Run model to perform one-step processing.
    (analysis_output, _) = analysis_net(daw_waveform)
    (synthesis_output, _) = synthesis_net(control_matrix)
    assert analysis_output.shape == control_matrix.shape
    assert synthesis_output.shape == daw_waveform.shape

    # Optional cycle supervision: perform analysis from predicted waveform, and also perform
    # synthesis from predicted control matrix with both neural proxy and DAW.

    # NOTE: this is different from train pipeline since we don't require differentiability!
    control_matrix_output = (analysis_output > 0.0).type(torch.float32)

    # Run model again to perform two-step processing.
    (anasyn_cycle, _) = synthesis_net(control_matrix_output)
    (synana_cycle, _) = analysis_net(synthesis_output)
    assert anasyn_cycle.shape == daw_waveform.shape
    assert synana_cycle.shape == control_matrix.shape

    # Convert data from torch to numpy for futher processing.
    control_matrix = control_matrix.detach().cpu().numpy()
    daw_waveform = daw_waveform.detach().cpu().numpy()
    analysis_output = analysis_output.detach().cpu().numpy()
    synthesis_output = synthesis_output.detach().cpu().numpy()
    control_matrix_output = control_matrix_output.detach().cpu().numpy()
    anasyn_cycle = anasyn_cycle.detach().cpu().numpy()
    synana_cycle = synana_cycle.detach().cpu().numpy()

    all_anadaw_cycle = []
    
    # Use the DAW on every test example.
    for i in range(B):

        # Binarize analysis output (= logits, not probits) into control matrix for DAW.
        tp_matrix_output = (analysis_output[i] > 0.0).astype(np.float32)
        
        # Synthesize for cycle DAW reconstruction.
        event_seq = daw_synth.tp_matrix_to_event_seq(
            tp_matrix_output, timecell_ms)
        anadaw_cycle = daw_synth.event_seq_to_waveform(
            soundfont_path, event_seq, sample_rate, True, S)
        
        # Update lists to keep.
        all_anadaw_cycle.append(anadaw_cycle)
        
        # Evaluate whatever.
        # TODO

    anadaw_cycle = np.stack(all_anadaw_cycle)
    assert anadaw_cycle.shape == daw_waveform.shape
    
    # Apply sigmoid to control matrices.
    analysis_output = 1.0 / (1.0 + np.exp(-analysis_output))
    synana_cycle = 1.0 / (1.0 + np.exp(-synana_cycle))

    # Organize and return relevant info.
    inference_retval = dict()
    inference_retval['control_matrix'] = control_matrix  # (B, P, T).
    inference_retval['daw_waveform'] = daw_waveform  # (B, S).
    inference_retval['analysis_output'] = analysis_output  # (B, P, T).
    inference_retval['synthesis_output'] = synthesis_output  # (B, S).
    inference_retval['anasyn_cycle'] = anasyn_cycle  # (B, S).
    inference_retval['synana_cycle'] = synana_cycle  # (B, P, T).
    inference_retval['anadaw_cycle'] = anadaw_cycle  # (B, S).

    return inference_retval
