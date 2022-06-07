#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import torch
from core.prompt_bert import BertForPromptTuning
from tools.args import DynamicDataTrainingArguments
from tools.args import DynamicTrainingArguments
from tools.args import ModelArguments
from tools.dataset import PromptDataset
from tools.glue_data_processor import processors_mapping
from tools.tools import multipart_map
from tools.tools import resize_token_type_embeddings
from tools.trainer import Trainer
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import HfArgumentParser
from transformers import set_seed


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=training_args.num_labels,
        cache_dir=model_args.cache_dir
    )

    if config.model_type == "bert":
        model_fn = BertForPromptTuning
    else:
        raise NotImplementedError(f"`{config.model_type}` not impl")

    special_tokens = []

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )

    # 这里可以融合多个processors_mapping
    processor = processors_mapping[data_args.processor]

    tokenize_multipart_input = multipart_map[data_args.multipart_type]

    train_dataset = PromptDataset(model_args.data_dir, data_args.label_to_word, tokenizer,
                                  processor, data_args.template, data_args.max_seq_length,
                                  tokenize_multipart_input, "train", data_args.num_sample)

    if training_args.do_eval:
        eval_dataset = PromptDataset(model_args.data_dir, data_args.label_to_word, tokenizer,
                                     processor, data_args.template, data_args.max_seq_length,
                                     tokenize_multipart_input, "dev", data_args.num_sample)

    if training_args.do_predict:
        test_dataset = PromptDataset(model_args.data_dir, data_args.label_to_word, tokenizer,
                                     processor, data_args.template, data_args.max_seq_length,
                                     tokenize_multipart_input, "test", data_args.num_sample)

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir
    )

    # For BERT, increase the size of the segment (token type) embeddings
    if config.model_type == "bert":
        model.resize_token_embeddings(len(tokenizer))
        resize_token_type_embeddings(model, new_num_types=10, random_segment=model_args.random_segment)

    model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics_fn(data_args.task_name)
    )

    trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)


if __name__ == "__main__":
    main()
