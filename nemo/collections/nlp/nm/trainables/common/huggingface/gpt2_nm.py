# =============================================================================
# Copyright 2020 NVIDIA. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and
# The HuggingFace Inc. team.
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
# =============================================================================

from typing import List, Optional

from transformers import GPT2Config, GPT2Model

from nemo.backends.pytorch.nm import TrainableNM
from nemo.core.neural_modules import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, NeuralType
from nemo.utils.decorators import add_port_docs

__all__ = ['GPT2']


class GPT2(TrainableNM):
    """
    GPT2 wraps around the Huggingface implementation of GPT2 from their
    transformers repository for easy use within NeMo.

    Args:
        # pretrained_model_name (str): If using a pretrained model, this should
        #     be the model's name. Otherwise, should be left as None.
        # config_filename (str): path to model configuration file. Optional.
        # vocab_size (int): Size of the vocabulary file, if not using a
        #     pretrained model.
        # hidden_size (int): Size of the encoder and pooler layers.
        # num_hidden_layers (int): Number of hidden layers in the encoder.
        # num_attention_heads (int): Number of attention heads for each layer.
        # intermediate_size (int): Size of intermediate layers in the encoder.
        # hidden_act (str): Activation function for encoder and pooler layers;
        #     "gelu", "relu", and "swish" are supported.
        # max_position_embeddings (int): The maximum number of tokens in a
        # sequence.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        input_ids (torch.LongTensor of shape (batch_size, sequence_length)) –
            Indices of input sequence tokens in the vocabulary.
            If past is used, optionally only the last input_ids have to be input (see past).
        past (List[torch.FloatTensor] of length config.n_layers) – Contains pre-computed hidden-states
            (key and values in the attention blocks) as computed by the model (see past output below).
            Can be used to speed up sequential decoding. If past is used, the user can optionally input
            only the last input_ids (those that don’t have their past given to this model) of shape (batch_size, 1)
            instead of all input_ids of shape (batch_size, sequence_length).
        attention_mask (torch.FloatTensor of shape (batch_size, sequence_length), optional, defaults to None) –
            Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]:
            1 for tokens that are NOT MASKED, 0 for MASKED tokens.
        token_type_ids (torch.LongTensor of shape (batch_size, input_ids_length), optional, defaults to None) –
            input_ids_length = sequence_length if `past is None else 1 Segment token indices to indicate first
            and second portions of the inputs. Indices are selected in [0, 1]: 0 corresponds to a sentence A token,
            1 corresponds to a sentence B token If past is used, optionally only the last token_type_ids have to be input (see past).
        position_ids (torch.LongTensor of shape (batch_size, sequence_length), optional, defaults to None) –
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the 
            range [0, config.max_position_embeddings - 1].
        head_mask (torch.FloatTensor of shape (num_heads,) or (num_layers, num_heads), optional, defaults to None)
             – Mask to nullify selected heads of the self-attention modules. Mask values selected in [0, 1]: 
             1 indicates the head is not masked, 0 indicates the head is masked.
        input_embeds (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size), optional, defaults to None)
            Optionally, instead of passing input_ids you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert input_ids indices into associated vectors
            than the model’s internal embedding lookup matrix. If past is used, optionally only the last input_embeds
            have to be input (see past).    
        use_cache (bool) – If use_cache is True, past key value states are returned and can be used to speed up
            decoding (see past). Defaults to True.
        """
        return {
            "input_ids": NeuralType(('B', 'T'), ChannelType()),
            "token_type_ids": NeuralType(('B', 'T'), ChannelType()),
            "attention_mask": NeuralType(('B', 'T'), ChannelType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        hidden_states: output embedding 
        """
        return {"hidden_states": NeuralType(('B', 'T', 'D'), ChannelType())}

    def __init__(
        self,
        pretrained_model_name=None,
        config_filename=None,
        vocab_size=None,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    ):

        super().__init__()

        # Check that only one of pretrained_model_name, config_filename, and
        # vocab_size was passed in
        total = 0
        if pretrained_model_name is not None:
            total += 1
        if config_filename is not None:
            total += 1
        if vocab_size is not None:
            total += 1

        if total != 1:
            raise ValueError(
                "Only one of pretrained_model_name, vocab_size, "
                + "or config_filename should be passed into the "
                + "BERT constructor."
            )

        # TK: The following code checks the same once again.
        if vocab_size is not None:
            config = BertConfig(
                vocab_size_or_config_json_file=vocab_size,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                hidden_act=hidden_act,
                max_position_embeddings=max_position_embeddings,
            )
            model = BertModel(config)
        elif pretrained_model_name is not None:
            model = BertModel.from_pretrained(pretrained_model_name)
        elif config_filename is not None:
            config = BertConfig.from_json_file(config_filename)
            model = BertModel(config)
        else:
            raise ValueError(
                "Either pretrained_model_name or vocab_size must" + " be passed into the BERT constructor"
            )

        model.to(self._device)

        self.add_module("bert", model)
        self.config = model.config
        self._hidden_size = model.config.hidden_size

    @property
    def hidden_size(self):
        """
            Property returning hidden size.

            Returns:
                Hidden size.
        """
        return self._hidden_size

    @staticmethod
    def list_pretrained_models() -> Optional[List[PretrainedModelInfo]]:
        pretrained_models = []
        for key, value in BERT_PRETRAINED_MODEL_ARCHIVE_MAP.items():
            model_info = PretrainedModelInfo(
                pretrained_model_name=key,
                description="weights by HuggingFace",
                parameters=BERT_PRETRAINED_CONFIG_ARCHIVE_MAP[key],
                location=value,
            )
            pretrained_models.append(model_info)
        return pretrained_models

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
