'''
Logging and visualization logic.
'''

from __init__ import *

# Internal imports.
import utils.logvisgen as logvisgen
import utils.visualization as visualization


class MyLogger(logvisgen.Logger):
    '''
    Adapts the generic logger to this specific project.
    '''

    def __init__(self, args, context):
        if 'batch_size' in args:
            if args.is_debug:
                self.step_interval = 80 // args.batch_size
            else:
                self.step_interval = 3200 // args.batch_size
        else:
            if args.is_debug:
                self.step_interval = 20
            else:
                self.step_interval = 200
        self.num_exemplars = 2  # To increase simultaneous examples in wandb during train / val.
        super().__init__(args.log_path, context)

    def handle_train_step(self, epoch, phase, cur_step, total_step, steps_per_epoch,
                          data_retval, model_retval, loss_retval, train_args):

        if cur_step % self.step_interval == 0:

            exemplar_idx = (cur_step // self.step_interval) % self.num_exemplars
            batch_size = data_retval['daw_waveform'].shape[0]
            total_loss = loss_retval['total']
            loss_analysis = loss_retval['analysis']
            loss_synthesis = loss_retval['synthesis']

            # Prepare data.
            waveform_target = data_retval['daw_waveform'][0].detach().cpu().numpy()
            control_matrix_target = data_retval['control_matrix'][0].detach().cpu().numpy()
            analysis_output = model_retval['analysis_output'][0].detach().cpu().numpy()
            synthesis_output = model_retval['synthesis_output'][0].detach().cpu().numpy()
            anasyn_cycle = model_retval['anasyn_cycle'][0].detach().cpu().numpy()
            synana_cycle = model_retval['synana_cycle'][0].detach().cpu().numpy()
            anadaw_cycle = model_retval['anadaw_cycle'][0].detach().cpu().numpy()

            # Print metrics in console.
            self.info(f'[Step {cur_step} / {steps_per_epoch}]  '
                      f'total_loss: {total_loss:.3f}  '
                      f'analysis: {loss_analysis:.3f}  '
                      f'synthesis: {loss_synthesis:.3f}')

            # Save metadata in JSON format, for inspection / debugging / reproducibility.
            metadata = dict()
            metadata['batch_size'] = batch_size
            if 'file_idx' in data_retval:
                metadata['file_idx'] = data_retval['file_idx'][i].item()
            if 'src_fp' in data_retval:
                metadata['src_fp'] = data_retval['src_fp'][i]
            if 'within_idx' in data_retval:
                metadata['within_idx'] = data_retval['within_idx'][i].item()
            if 'clip_start' in data_retval:
                metadata['clip_start'] = data_retval['clip_start'][i].item()
            if 'clip_end' in data_retval:
                metadata['clip_end'] = data_retval['clip_end'][i].item()

            # Apply sigmoid to control matrices.
            analysis_output = 1.0 / (1.0 + np.exp(-analysis_output))
            synana_cycle = 1.0 / (1.0 + np.exp(-synana_cycle))

            # Save input / ground truth + prediction + cycle waveforms,
            # both as audio files and image plots.
            # NOTE: Mainly interested in neural analysis + DAW synthesis for now.
            waveforms_cat = np.concatenate([waveform_target, anadaw_cycle], axis=-1)
            self.save_audio(waveforms_cat, sample_rate=train_args.sample_rate, step=epoch,
                            file_name=f'wf_target_anadaw_e{epoch}_p{phase}_s{cur_step}.wav',
                            online_name=f'wf_target_anadaw_p{phase}_x{exemplar_idx}')
            waveforms_cat = np.concatenate([synthesis_output, anasyn_cycle], axis=-1)
            self.save_audio(waveforms_cat, sample_rate=train_args.sample_rate, step=epoch,
                            file_name=f'wf_syn_anasyn_e{epoch}_p{phase}_s{cur_step}.wav',
                            online_name=None)

            vis_waveforms = visualization.gen_waveform_plot_image(
                [waveform_target * 4.0, synthesis_output * 4.0,
                 anasyn_cycle * 4.0, anadaw_cycle * 4.0],
                ['Input / Target', 'Output', 'Cycle Proxy', 'Cycle DAW'],
                train_args.sample_rate, zoom=1000)
            self.save_image(vis_waveforms, step=epoch,
                            file_name=f'wf_zoom_e{epoch}_p{phase}_s{cur_step}.png',
                            online_name=f'wf_zoom_p{phase}_x{exemplar_idx}')

            # For torch STFT parameters, see my_synthesis_loss() in loss.py.
            # hop_length = nperseg - noverlap.
            nperseg = train_args.n_fft_freq
            noverlap = (train_args.n_fft_freq * 3) // 4
            f, t, Zxx_target = scipy.signal.stft(
                waveform_target, train_args.sample_rate, nperseg=nperseg, noverlap=noverlap)
            vis_stft_target = visualization.get_stft_plot_image(
                f, t, Zxx_target, 'Input / Target STFT Magnitude')
            f, t, Zxx_output = scipy.signal.stft(
                synthesis_output, train_args.sample_rate, nperseg=nperseg, noverlap=noverlap)
            vis_stft_output = visualization.get_stft_plot_image(
                f, t, Zxx_output, 'Output STFT Magnitude')
            f, t, Zxx_cycle_proxy = scipy.signal.stft(
                anasyn_cycle, train_args.sample_rate, nperseg=nperseg, noverlap=noverlap)
            vis_stft_cycle_proxy = visualization.get_stft_plot_image(
                f, t, Zxx_cycle_proxy, 'Cycle Proxy STFT Magnitude')
            f, t, Zxx_cycle_daw = scipy.signal.stft(
                anadaw_cycle, train_args.sample_rate, nperseg=nperseg, noverlap=noverlap)
            vis_stft_cycle_daw = visualization.get_stft_plot_image(
                f, t, Zxx_cycle_daw, 'Cycle DAW STFT Magnitude')

            # Save input / ground truth and prediction spectrograms.
            vis_matrix_target = visualization.get_tp_matrix_plot_image(
                control_matrix_target, 'Input / Target Control Matrix',
                train_args.timecell_ms, train_args.pitch_bounds)
            vis_matrix_output = visualization.get_tp_matrix_plot_image(
                analysis_output, 'Output Control Matrix',
                train_args.timecell_ms, train_args.pitch_bounds)
            vis_matrix_cycle = visualization.get_tp_matrix_plot_image(
                synana_cycle, 'Cycle Control Matrix',
                train_args.timecell_ms, train_args.pitch_bounds)
            vis_empty = np.ones_like(vis_matrix_cycle)

            gallery = rearrange([vis_stft_target, vis_stft_output,
                                 vis_stft_cycle_proxy, vis_stft_cycle_daw,
                                 vis_matrix_target, vis_matrix_output,
                                 vis_matrix_cycle, vis_empty],
                                '(G1 G2) H W C -> G1 G2 H W C', G1=2, G2=4)
            gallery = np.clip(gallery, 0.0, 1.0)
            self.save_gallery(gallery, step=epoch,
                              file_name=f'cm_stft_e{epoch}_p{phase}_s{cur_step}.png',
                              online_name=f'cm_stft_p{phase}_x{exemplar_idx}')

            # Save embedding statistics for debugging.
            self.report_histogram('waveform_target',
                                  waveform_target.flatten(), step=epoch)
            self.report_histogram('stft_mag_target',
                                  np.abs(Zxx_target).flatten(), step=epoch)
            self.report_histogram('control_matrix_target',
                                  control_matrix_target.flatten(), step=epoch)
            self.report_histogram('synthesis_output_wf',
                                  synthesis_output.flatten(), step=epoch)
            self.report_histogram('synthesis_output_stft_mag',
                                  np.abs(Zxx_output).flatten(), step=epoch)
            self.report_histogram('analysis_output',
                                  analysis_output.flatten(), step=epoch)

            self.save_text(metadata, file_name=f'metadata_e{epoch}_p{phase}_s{cur_step}.txt')

    def epoch_finished(self, epoch):
        self.commit_scalars(step=epoch)

    def handle_test_step(self, cur_step, num_steps, data_retval, inference_retval, all_args):
        '''
        :param all_args (dict): train, test, train_dset, test_dest, model.
        '''

        batch_size = inference_retval['daw_waveform'].shape[0]
        sample_rate = all_args['train'].sample_rate
        timecell_ms = all_args['train'].timecell_ms
        n_fft_freq = all_args['train'].n_fft_freq
        pitch_bounds = all_args['train'].pitch_bounds

        for i in range(batch_size):

            # Prepare data.
            cur_substep = i + cur_step * batch_size
            control_matrix_target = inference_retval['control_matrix'][i]  # (P, T).
            waveform_target = inference_retval['daw_waveform'][i]  # (S).
            analysis_output = inference_retval['analysis_output'][i]  # (P, T).
            synthesis_output = inference_retval['synthesis_output'][i]  # (S).
            anasyn_cycle = inference_retval['anasyn_cycle'][i]  # (S).
            synana_cycle = inference_retval['synana_cycle'][i]  # (P, T).
            anadaw_cycle = inference_retval['anadaw_cycle'][i]  # (S).
            control_matrix_output = (analysis_output > 0.5).astype(np.float32)

            # Print metrics in console.
            self.info(f'[Step {cur_substep} / {num_steps * batch_size}]')

            # Save metadata in JSON format, for inspection / debugging / reproducibility.
            metadata = dict()
            metadata['batch_size'] = batch_size
            if 'file_idx' in data_retval:
                metadata['file_idx'] = data_retval['file_idx'][i].item()
            if 'src_fp' in data_retval:
                metadata['src_fp'] = data_retval['src_fp'][i]
            if 'within_idx' in data_retval:
                metadata['within_idx'] = data_retval['within_idx'][i].item()
            if 'clip_start' in data_retval:
                metadata['clip_start'] = data_retval['clip_start'][i].item()
            if 'clip_end' in data_retval:
                metadata['clip_end'] = data_retval['clip_end'][i].item()

            # Save input / ground truth + prediction + cycle waveforms,
            # both as audio files and image plots.
            # NOTE: Mainly interested in neural analysis + DAW synthesis for now.
            waveforms_cat = np.concatenate([waveform_target, anadaw_cycle], axis=-1)
            self.save_audio(waveforms_cat, sample_rate=sample_rate, step=cur_substep,
                            file_name=f'wf_target_anadaw_cat_s{cur_substep}.wav',
                            online_name=f'wf_target_anadaw_cat')
            waveforms_stereo = np.stack([waveform_target, anadaw_cycle], axis=-1)
            self.save_audio(waveforms_stereo, sample_rate=sample_rate, step=cur_substep,
                            file_name=f'wf_target_anadaw_stereo_s{cur_substep}.wav',
                            online_name=f'wf_target_anadaw_stereo')
            waveforms_cat = np.concatenate([synthesis_output, anasyn_cycle], axis=-1)
            self.save_audio(waveforms_cat, sample_rate=sample_rate, step=cur_substep,
                            file_name=f'wf_syn_anasyn_s{cur_substep}.wav',
                            online_name=None)

            vis_waveforms = visualization.gen_waveform_plot_image(
                [waveform_target * 4.0, synthesis_output * 4.0,
                    anasyn_cycle * 4.0, anadaw_cycle * 4.0],
                ['Input / Target', 'Output', 'Cycle Proxy', 'Cycle DAW'],
                sample_rate, zoom=1000)
            self.save_image(vis_waveforms, step=cur_substep,
                            file_name=f'wf_zoom_s{cur_substep}.png',
                            online_name=f'wf_zoom')

            # For torch STFT parameters, see my_synthesis_loss() in loss.py.
            # hop_length = nperseg - noverlap.
            nperseg = n_fft_freq
            noverlap = (n_fft_freq * 3) // 4
            f, t, Zxx_target = scipy.signal.stft(
                waveform_target, sample_rate, nperseg=nperseg, noverlap=noverlap)
            vis_stft_target = visualization.get_stft_plot_image(
                f, t, Zxx_target, 'Input / Target STFT Magnitude')
            f, t, Zxx_output = scipy.signal.stft(
                synthesis_output, sample_rate, nperseg=nperseg, noverlap=noverlap)
            vis_stft_output = visualization.get_stft_plot_image(
                f, t, Zxx_output, 'Output STFT Magnitude')
            f, t, Zxx_cycle_proxy = scipy.signal.stft(
                anasyn_cycle, sample_rate, nperseg=nperseg, noverlap=noverlap)
            vis_stft_cycle_proxy = visualization.get_stft_plot_image(
                f, t, Zxx_cycle_proxy, 'Cycle Proxy STFT Magnitude')
            f, t, Zxx_cycle_daw = scipy.signal.stft(
                anadaw_cycle, sample_rate, nperseg=nperseg, noverlap=noverlap)
            vis_stft_cycle_daw = visualization.get_stft_plot_image(
                f, t, Zxx_cycle_daw, 'Cycle DAW STFT Magnitude')

            # Save input / ground truth and prediction spectrograms.
            vis_matrix_target = visualization.get_tp_matrix_plot_image(
                control_matrix_target, 'Input / Target Control Matrix',
                timecell_ms, pitch_bounds)
            vis_matrix_output = visualization.get_tp_matrix_plot_image(
                analysis_output, 'Output Control Matrix',
                timecell_ms, pitch_bounds)
            vis_matrix_output_binary = visualization.get_tp_matrix_plot_image(
                control_matrix_output, 'Output Control Matrix (Quantized)',
                timecell_ms, pitch_bounds)
            vis_matrix_cycle = visualization.get_tp_matrix_plot_image(
                synana_cycle, 'Cycle Control Matrix',
                timecell_ms, pitch_bounds)
            vis_empty = np.ones_like(vis_matrix_target)

            # Save gallery with almost all information.
            gallery = rearrange([vis_stft_target, vis_stft_output,
                                 vis_stft_cycle_proxy, vis_stft_cycle_daw,
                                 vis_matrix_target, vis_matrix_output,
                                 vis_matrix_cycle, vis_empty],
                                '(G1 G2) H W C -> G1 G2 H W C', G1=2, G2=4)
            gallery = np.clip(gallery, 0.0, 1.0)
            self.save_gallery(gallery, step=cur_substep,
                              file_name=f'cm_stft_all_s{cur_substep}.png',
                              online_name=f'cm_stft_all')

            # Save gallery with only neural analysis + DAW synthesis.
            vis_waveforms = visualization.gen_waveform_plot_image(
                [waveform_target * 4.0, anadaw_cycle * 4.0], ['Input / Target', 'Cycle DAW'],
                sample_rate, zoom=500)
            gallery = rearrange([vis_stft_target, vis_waveforms, vis_stft_cycle_daw,
                                 vis_matrix_target, vis_matrix_output, vis_matrix_output_binary],
                                '(G1 G2) H W C -> G1 G2 H W C', G1=2, G2=3)
            gallery = np.clip(gallery, 0.0, 1.0)
            self.save_gallery(gallery, step=cur_substep,
                              file_name=f'wf_cm_stft_anadaw_s{cur_substep}.png',
                              online_name=f'wf_cm_stft_anadaw')

            # Save embedding statistics for debugging.
            self.report_histogram('waveform_target',
                                  waveform_target.flatten(), step=cur_substep)
            self.report_histogram('stft_mag_target',
                                  np.abs(Zxx_target).flatten(), step=cur_substep)
            self.report_histogram('control_matrix_target',
                                  control_matrix_target.flatten(), step=cur_substep)
            self.report_histogram('synthesis_output_wf',
                                  synthesis_output.flatten(), step=cur_substep)
            self.report_histogram('synthesis_output_stft_mag',
                                  np.abs(Zxx_output).flatten(), step=cur_substep)
            self.report_histogram('analysis_output',
                                  analysis_output.flatten(), step=cur_substep)

            self.save_text(metadata, file_name=f'metadata_s{cur_substep}.txt')