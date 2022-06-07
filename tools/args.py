#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataclasses import dataclass
from dataclasses import field
from transformers import TrainingArguments
from typing import Optional


@dataclass
class ModelArguments:
    """ model/config/tokenizer 相关参数 """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    random_segment: bool = field(
        default=False,
        metadata={"help": "Whether to reinitialize the token type embeddings (only for BERT)."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DynamicDataTrainingArguments:
    """ 控制prompt相关参数 """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "Number of training instances per class"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )

    label_to_word: str = field(
        default=None,
        metadata={"help": "Label to word mapping"}
    )

    processor: str = field(
        default=None,
        metadata={"help": "processor name"}
    )

    max_seq_length: int = field(
        default=None,
        metadata={"help": "full length (512)"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    multipart_type: str = field(
        default=None,
        metadata={"help": "tokenize multipart input"}
    )


@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )

    num_labels: int = field(
        default=None,
        metadata={"help": "task labels num"}
    )

    output_mode: str = field(
        default=None,
        metadata={"help": "task mode"}
    )

    data_dir: str = field(
        default=None,
        metadata={"help": "data dir, include train, dev, test"}
    )
