'''
Model constituents.
'''

import torch
# Library imports.
# https://huggingface.co/docs/transformers/quicktour
import transformers
from transformers.models.perceiver import modeling_perceiver
from einops import rearrange, repeat
# Internal imports.
import utils
import numpy as np


class MyPerceiverBackbone(torch.nn.Module):

    def __init__(self, logger, input_shape, output_shape, samples_per_frame,
                 output_pos_enc):
        '''
        X
        '''
        super().__init__()
        self.logger = logger
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.samples_per_frame = samples_per_frame
        self.output_pos_enc = output_pos_enc

        # This is heavily inspired by huggingface modeling_perceiver.py

        self.config = transformers.PerceiverConfig(
            num_latents=256,
            d_latents=1024,
            d_model=512,
            num_blocks=1,
            num_self_attends_per_block=12,
            num_self_attention_heads=8,
            num_cross_attention_heads=1,
            samples_per_patch=samples_per_frame,
        )

        # Instantiate preprocessor (before encoder).
        if len(input_shape) == 1:
            # Assume input is waveform.
            self.input_preprocessor = modeling_perceiver.PerceiverAudioPreprocessor(
                self.config,
                prep_type='patches',
                samples_per_patch=samples_per_frame,
                out_channels=60,
                position_encoding_type='fourier',
                fourier_position_encoding_kwargs=dict(
                    max_resolution=(input_shape[0],),
                    num_bands=129,
                    concat_pos=True,
                    sine_only=False,
                ),
            )

        elif len(input_shape) == 2:
            # Assume input is control matrix.
            self.input_preprocessor = modeling_perceiver.PerceiverImagePreprocessor(
                self.config,
                prep_type='conv1x1',
                spatial_downsample=1,
                position_encoding_type='fourier',
                in_channels=1,
                out_channels=384*2,
                fourier_position_encoding_kwargs=dict(
                    max_resolution=input_shape,
                    num_bands=384,
                    concat_pos=True,
                    sine_only=False,
                ),
            )

        else:
            raise ValueError(input_shape)

        # Instantiate postprocessor (after decoder).
        if len(output_shape) == 1:
            # Assume output is waveform.
            # NOTE: num_channels in decoder needs to be at least num_bands * 2.
            output_num_channels = 512*4
            output_index_dims = (output_shape[0] // samples_per_frame,)
            # query_num_channels = 512
            num_bands = 192*4
            self.output_postprocessor = modeling_perceiver.PerceiverAudioPostprocessor(
                self.config,
                in_channels=output_num_channels,
            )

        elif len(output_shape) == 2:
            # Assume output is control matrix.
            # NOTE: num_channels in decoder needs to be at least num_bands * 4.
            output_num_channels = 128
            output_index_dims = output_shape
            # query_num_channels = 256
            num_bands = 48
            self.output_postprocessor = modeling_perceiver.PerceiverProjectionPostprocessor(
                in_channels=output_num_channels,
                out_channels=1,
            )

        else:
            raise ValueError(output_shape)

        if output_pos_enc == 'fourier':
            # This is why huggingface's implementation is awkward; doesn't automatically infer this.
            # NOTE: There is only one cross attention (PerceiverLayer) step, so it is not that bad.
            query_num_channels = (num_bands * 2 + 1) * len(output_shape)
            position_encoding_kwargs = dict(
                max_resolution=output_index_dims,
                num_bands=num_bands,
                concat_pos=True,
                sine_only=False,
            )

        elif output_pos_enc == 'trainable':
            query_num_channels = 64 #256
            position_encoding_kwargs = dict(
                index_dims=output_index_dims,
                num_channels=query_num_channels,
            )

        else:
            raise ValueError(output_pos_enc)

        # Instantiate decoder.
        self.decoder = modeling_perceiver.PerceiverBasicDecoder(
            self.config,
            output_num_channels=output_num_channels,
            output_index_dims=output_index_dims,
            num_channels=query_num_channels,
            position_encoding_type=output_pos_enc,
            concat_preprocessed_input=False,
            fourier_position_encoding_kwargs=position_encoding_kwargs,
            trainable_position_encoding_kwargs=position_encoding_kwargs,
        )

        self.perceiver = transformers.PerceiverModel(
            self.config,
            input_preprocessor=self.input_preprocessor,
            decoder=self.decoder,
            output_postprocessor=self.output_postprocessor,
        )

        # self.perceiver.post_init()

        pass

    def forward(self, input):
        '''
        :param input (B, S) or (B, C, T) tensor.
        :return (output, last_hidden_state)
            output (B, S) or (B, C, T) tensor.
            last_hidden_state (B, L, D) tensor.
        '''
        if len(self.input_shape) == 1:
            # Assume input is waveform.
            pass
        elif len(self.input_shape) == 2:
            # Assume input is control matrix.
            input = repeat(input, 'B C T -> B Z C T', Z=1)

        # NOTE: It is crucial to specify the output shape as a flat array of flat indices here.
        # This is just how huggingface perceiver works.
        if len(self.output_shape) == 1:
            # Assume output is waveform.
            subsampling = torch.arange(self.output_shape[0] // self.samples_per_frame)
        elif len(self.output_shape) == 2:
            # Assume output is control matrix.
            subsampling = torch.arange(np.prod(self.output_shape))

        perceiver_output = self.perceiver(inputs=input, subsampled_output_points=subsampling)
        output = perceiver_output.logits
        last_hidden_state = perceiver_output.last_hidden_state

        if len(self.output_shape) == 1:
            # Assume output is waveform.
            pass
        elif len(self.output_shape) == 2:
            # Assume output is control matrix.
            #1 (5 1) 1 ... 1 5 (1 1)
            pass
            #output = rearrange(output, 'B (C T) Z -> B C (T Z)',C=self.output_shape[0], T=self.output_shape[1], Z=1)

        return (output, last_hidden_state)