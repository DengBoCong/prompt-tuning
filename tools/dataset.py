#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import dataclasses
import pandas as pd
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers.data.processors import DataProcessor
from transformers.data.processors import InputExample
from typing import Any, Dict, List, Optional, Union, Callable


@dataclass(frozen=True)
class InputFeatures:
    """see transformer InputFeatures"""
    input_ids: List[int] = None
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None
    mask_pos: Optional[List[int]] = None  # Position of the mask token
    label_word_list: Optional[List[int]] = None  # Label word mapping (dynamic)

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class PromptDataset(Dataset):
    """Dataset for Prompt"""

    def __init__(self,
                 data_dir: str,
                 label_to_word: Dict[str, Any],
                 tokenizer: Any,
                 processor: DataProcessor,
                 template: Any,
                 max_seq_length: int,
                 tokenize_multipart_input: Callable,
                 mode: str = "train",
                 num_sample: int = 16,
                 special_tokens: List[str] = None,
                 **kwargs) -> None:
        """
        :param data_dir: 数据集路径目录
        :param label_to_word: Label-Word mapping
        :param tokenizer: 编码器
        :param processor: 数据处理器
        :param template: template, str/list
        :param max_seq_length: max seq len
        :param mode: 当前执行模式
        :param num_sample: 采样template的数量
        :param special_tokens: label中特殊token
        """
        assert mode in ["train", "dev", "test"]

        self.data_dir = data_dir
        self.processor = processor
        self.label_to_word = label_to_word
        self.tokenizer = tokenizer
        self.template = template
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.num_sample = num_sample
        self.special_tokens = special_tokens
        # 这里留个参数选项给InputFeature输入处理方法
        self.kwargs = kwargs
        self.tokenize_multipart_input = tokenize_multipart_input

        if self.special_tokens is None:
            self.special_tokens = ["<", "[", ".", ","]

        self.label_list = self.processor.get_labels()
        self.num_labels = len(self.label_list)
        self.label_word_list = []

        for key in self.label_to_word:
            # For RoBERTa/BART/T5, tokenization also considers space, so we use space+word as label words.
            if self.label_to_word[key][0] not in self.special_tokens:
                # Make sure space+word is in the vocabulary
                assert len(self.tokenizer.tokenize(f" {self.label_to_word[key]}")) == 1
                self.label_to_word[key] = self.tokenizer._convert_token_to_id(
                    self.tokenizer.tokenize(f" {self.label_to_word[key]}")[0])
            else:
                self.label_to_word[key] = self.tokenizer._convert_token_to_id(self.label_to_word[key])

            if len(self.label_list) > 1:
                self.label_word_list = [self.label_to_word[label] for label in self.label_list]
            else:
                self.label_word_list = [self.label_to_word[label] for label in ["0", "1"]]

        if self.mode == "train":
            # We do not do multiple sampling when it's the training mode
            self.num_sample = 1
        else:
            self.num_sample = self.num_sample

        # 在inference阶段需要被多次采样
        if isinstance(self.template, list):
            self.num_sample *= len(self.template)

        # The support examples are sourced from the training set.
        self.support_examples = self.processor.get_train_examples(self.data_dir)

        if mode == "dev":
            self.query_examples = self.processor.get_dev_examples(self.data_dir)
        elif mode == "test":
            self.query_examples = self.processor.get_test_examples(self.data_dir)
        else:
            self.query_examples = self.support_examples

        self.size = len(self.query_examples) * self.num_sample
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        for sample_idx in range(self.num_sample):
            for query_idx in range(len(self.query_examples)):
                context_indices = [support_idx for support_idx in support_indices
                                   if support_idx != query_idx or mode != "train"]
                self.example_idx.append((query_idx, context_indices, sample_idx))

        # If it is not training, we pre-process the data; otherwise, we process the data online.
        if mode != "train":
            self.features, count = [], 0
            for query_idx, context_indices, sample_idx in self.example_idx:
                example = self.query_examples[query_idx]
                if isinstance(self.template, list):
                    template = self.template[sample_idx % len(self.template)]  # Use template in order
                else:
                    template = self.template

                self.features.append(self.convert_fn(example=example, template=template))

                count += 1
        else:
            self.features = None

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.features is None:
            query_idx, context_indices, sample_idx = self.example_idx[i]
            example = self.query_examples[query_idx]
            if isinstance(self.template, list):
                template = self.template[sample_idx % len(self.template)]  # Use template in order
            else:
                template = self.template

            features = self.convert_fn(example=example, template=template)
        else:
            features = self.features[i]

        return features

    def get_labels(self) -> List[str]:
        return self.label_list

    def convert_fn(self, example: InputExample, template: str) -> InputFeatures:
        """
        :param example: input example
        :param template: template
        """
        label_map = {label: i for i, label in enumerate(self.label_list)}  # Mapping the label names to label ids
        if len(self.label_list) == 1:
            label_map = {"0": 0, "1": 1}

        if example.label is None:
            example_label = None
        elif len(self.label_list) == 1:
            example_label = float(example.label)
        else:
            example_label = label_map[example.label]

        assert not pd.isna(example.text_a) and example.text_a is not None

        inputs = self.tokenize_multipart_input(
            input_text_list=[example.text_a] if example.text_b is None else [example.text_a, example.text_b],
            max_length=self.max_seq_length,
            tokenizer=self.tokenizer,
            template=template,
            label_word_list=self.label_word_list,
            **self.kwargs
        )

        return InputFeatures(**inputs, label=example_label)
