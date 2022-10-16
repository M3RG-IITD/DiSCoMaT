# !/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE_TaBERT file in the root directory of this source tree.

import logging
from enum import Enum


class TransformerVersion(Enum):
    PYTORCH_PRETRAINED_BERT = 0
    TRANSFORMERS = 1


TRANSFORMER_VERSION = None

try:
    from pytorch_pretrained_bert.modeling import (
        BertForMaskedLM, BertForPreTraining, BertModel,
        BertConfig,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead, BertLayerNorm, gelu
    )
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    hf_flag = 'old'
    TRANSFORMER_VERSION = TransformerVersion.PYTORCH_PRETRAINED_BERT
    logging.warning('You are using the old version of `pytorch_pretrained_bert`')
    
except Exception as e:
    print(f"Using transformers as {e}")
    from transformers.models.bert import BertTokenizer    # noqa
    from transformers import (    # noqa
        BertForMaskedLM, BertForPreTraining, BertModel, BertConfig
    )
    from transformers.models.bert.modeling_bert import (
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead
    )
    from transformers.activations import gelu
    from torch.nn import LayerNorm as BertLayerNorm

    hf_flag = 'new'
    TRANSFORMER_VERSION = TransformerVersion.TRANSFORMERS
