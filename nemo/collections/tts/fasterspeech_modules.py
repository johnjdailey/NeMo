# Copyright 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

import nemo
from nemo.backends.pytorch import nm as nemo_nm
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.backends.pytorch.nm import LossNM
from nemo.collections.asr.parts import AudioDataset
from nemo.collections.asr.parts import WaveformFeaturizer
from nemo.collections.tts.parts import fastspeech
from nemo.core.neural_types import AudioSignal
from nemo.core.neural_types import ChannelType
from nemo.core.neural_types import EmbeddedTextType
from nemo.core.neural_types import LengthsType
from nemo.core.neural_types import MaskType
from nemo.core.neural_types import NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['FasterSpeechDataLayer', 'FasterSpeech', 'FasterSpeechDurLoss']


class FasterSpeechDataLayer(DataLayerNM):
    """Data Layer for Faster Speech model.

    Basically, replicated behavior from AudioToText Data Layer, zipped with ground truth durations for additional loss.

    Args:
        manifest_filepath (str): Dataset parameter.
            Path to JSON containing data.
        durs_dir (str): Path to durations arrays directory.
        labels (list): Dataset parameter.
            List of characters that can be output by the ASR model.
            For Jasper, this is the 28 character set {a-z '}. The CTC blank
            symbol is automatically added later for models using ctc.
        batch_size (int): batch size
        sample_rate (int): Target sampling rate for data. Audio files will be
            resampled to sample_rate if it is not already.
            Defaults to 16000.
        int_values (bool): Bool indicating whether the audio file is saved as
            int data or float data.
            Defaults to False.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        normalize_transcripts (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim_silence (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        load_audio (bool): Dataset parameter.
            Controls whether the dataloader loads the audio signal and
            transcript or just the transcript.
            Defaults to True.
        drop_last (bool): See PyTorch DataLoader.
            Defaults to False.
        shuffle (bool): See PyTorch DataLoader.
            Defaults to True.
        num_workers (int): See PyTorch DataLoader.
            Defaults to 0.
        perturb_config (dict): Currently disabled.

    """

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports."""
        return dict(
            audio=NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            audio_len=NeuralType(tuple('B'), LengthsType()),
            text=NeuralType(('B', 'T'), EmbeddedTextType()),
            text_mask=NeuralType(('B', 'T'), MaskType()),
            dur=NeuralType(('B', 'T'), LengthsType()),
        )

    def __init__(
            self,
            manifest_filepath,
            durs_dir,
            labels,
            batch_size,
            sample_rate=16000,
            int_values=False,
            bos_id=None,
            eos_id=None,
            pad_id=None,
            min_duration=0.1,
            max_duration=None,
            normalize_transcripts=True,
            trim_silence=False,
            load_audio=True,
            drop_last=False,
            shuffle=True,
            num_workers=0,
    ):
        super().__init__()

        # Set up dataset.
        self._featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=None)
        dataset_params = {
            'manifest_filepath': manifest_filepath,
            'labels': labels,
            'featurizer': self._featurizer,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'normalize': normalize_transcripts,
            'trim': trim_silence,
            'bos_id': bos_id,
            'eos_id': eos_id,
            'load_audio': load_audio,
        }
        audio_dataset = AudioDataset(**dataset_params)
        self._dataset = fastspeech.FastSpeechDataset(audio_dataset, durs_dir)
        self._pad_id = pad_id or 0
        self._space_id = labels.index(' ')
        self._sample_rate = sample_rate

        sampler = None
        if self._placement == nemo.core.DeviceType.AllGpu:
            sampler = torch.utils.data.distributed.DistributedSampler(self._dataset)

        self._dataloader = torch.utils.data.DataLoader(
            dataset=self._dataset,
            batch_size=batch_size,
            collate_fn=self._collate,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
        )

    def _collate(self, batch):
        def merge(tensors, value=0.0, dtype=torch.float):
            max_len = max(tensor.shape[0] for tensor in tensors)
            new_tensors = []
            for tensor in tensors:
                pad = (2 * len(tensor.shape)) * [0]
                pad[-1] = max_len - tensor.shape[0]
                new_tensors.append(F.pad(tensor, pad=pad, value=value))
            return torch.stack(new_tensors).to(dtype=dtype)

        def make_mask(lengths):
            return merge([torch.ones(length + 2) for length in lengths], value=0, dtype=torch.bool)

        batch = {key: [example[key] for example in batch] for key in batch[0]}

        audio = merge(batch['audio'])
        audio_len = torch.tensor(batch['audio_len'], dtype=torch.long)
        text = merge(
            [F.pad(text, pad=[1, 1], value=self._space_id) for text in batch['text']],
            value=self._pad_id,
            dtype=torch.long,
        )
        text_mask = make_mask(batch.pop('text_len'))
        dur = merge(batch['dur_true'], dtype=torch.long)

        assert text.shape == text_mask.shape, f'{text.shape} vs {text_mask.shape}'
        assert text.shape == dur.shape, f'{text.shape} vs {dur.shape}'

        return audio, audio_len, text, text_mask, dur

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def dataset(self) -> Optional[torch.utils.data.Dataset]:
        return None

    @property
    def data_iterator(self) -> Optional[torch.utils.data.DataLoader]:
        return self._dataloader


class FasterSpeech(nemo_nm.TrainableNM):
    """FasterSpeech Model."""

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definitions of module input ports."""
        return dict(
            text=NeuralType(('B', 'T'), EmbeddedTextType()),
            text_mask=NeuralType(('B', 'T'), MaskType()),
            dur=NeuralType(('B', 'T'), LengthsType(), optional=True),
        )

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports."""
        return dict(pred=NeuralType(('B', 'T', 'D'), ChannelType()), len=NeuralType(('B',), LengthsType()))

    def __init__(
            self, n_vocab, d_emb, pad_id, jasper_kwargs, d_out,
    ):
        super().__init__()

        # TODO: Have to come up with better implementation after testing phase.
        jasper_params = jasper_kwargs['jasper']
        d_enc_out = jasper_params[-1]["filters"]

        # Embedding for input text.
        self.emb = nn.Embedding(n_vocab, d_emb, padding_idx=pad_id).to(self._device)

        # To use with module(...., force_pt=True).
        self.jasper = nemo.collections.asr.JasperEncoder(feat_in=d_emb, **jasper_kwargs).to(self._device)

        # self.out = nn.Linear(d_enc_out, d_out).to(self._device)
        self.out = nn.Conv1d(d_enc_out, d_out, kernel_size=1, bias=True).to(self._device)

    def forward(self, text, text_mask, dur=None):
        if dur is not None:
            raise ValueError("Durations expansion is not implemented yet.")

        text_emb = self.emb(text).transpose(-1, -2)
        text_len = text_mask.sum(-1)

        pred, pred_len = self.jasper(text_emb, text_len, force_pt=True)
        assert text.shape[-1] == pred.shape[-1]

        pred = self.out(pred).transpose(-1, -2)

        return pred, pred_len


class FasterSpeechDurLoss(LossNM):
    """Neural Module Wrapper for Faster Speech Loss."""

    @property
    @add_port_docs
    def input_ports(self):
        """Returns definitions of module input ports."""
        return dict(
            dur_true=NeuralType(('B', 'T'), LengthsType()),
            dur_pred=NeuralType(('B', 'T', 'D'), ChannelType()),
            text_mask=NeuralType(('B', 'T'), MaskType()),
        )

    @property
    @add_port_docs
    def output_ports(self):
        """Returns definitions of module output ports."""
        return dict(loss=NeuralType(None))

    def __init__(self, reduction='true_mean'):
        super().__init__()

        self._reduction = reduction

    def _loss_function(self, dur_true, dur_pred, text_mask):
        """Do the actual math in FasterSpeech loss calculation."""

        if dur_pred.shape[-1] != 1:
            raise ValueError('Wrong dur_pred shape.')
        dur_pred = dur_pred.squeeze(-1)

        loss = F.mse_loss(dur_pred, (dur_true + 1).float().log(), reduction='none')
        loss *= text_mask.float()

        if self._reduction == 'true_mean':
            loss = loss.sum(-1) / text_mask.sum(-1)
            loss = loss.mean()
        else:
            raise ValueError("Wrong reduction method.")

        return loss
