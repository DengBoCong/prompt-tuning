#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
from tools.data_loader import loader_map
from tools.glue_data_processor import label_of_mapping
from core.gen_template import template_generator_map
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100, help="Random seeds")
    parser.add_argument("--task_name", type=str, default="", help="Task names")
    parser.add_argument("--k", type=int, default=16, help="Training examples for each class.")
    parser.add_argument("--data_dir", type=str, default="data/original", help="Path to original data")
    parser.add_argument("--output_dir", type=str, default="data", help="Output path")
    parser.add_argument("--dev_rate", type=int, default=1, help="dev:train scale")
    parser.add_argument("--data_loader", type=str, default="glue", choices=["glue"], help="Data loader")
    parser.add_argument("--template_generator", type=str, default="lm_bff",
                        choices=["lm_bff"], help="Template generator")
    parser.add_argument("--generator_config_path", type=str, default="data/config/lm_bff.json", help="Data loader")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, str(args.dev_rate))

    # random seed
    np.random.seed(args.seed)
    if args.data_loader == "glue":
        train_data, dev_data = loader_map[args.data_loader]().generate_k_shot(
            k=args.k, data_dir=args.data_dir, task_name=args.task_name, dev_rate=args.dev_rate
        )
        datasets = loader_map[args.data_loader].gen_samples(task_name=args.task_name, sources=train_data)
    else:
        raise ValueError(f"DataLoader `{args.data_loader}` not found")

    with open(args.generator_config_path, "r", encoding="utf-8") as file:
        generator_config = json.load(file)

    if args.template_generator == "lm_bff":
        model = T5ForConditionalGeneration.from_pretrained(generator_config["model_dir"])
        tokenizer = T5Tokenizer.from_pretrained(generator_config["model_dir"])
        tokenizer.sep_token = generator_config["end_token"]

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[index for index in range(torch.cuda.device_count())])
        model.eval()

        template_generator = template_generator_map[args.template_generator]()
        res_templates = template_generator.search_template(
            model, tokenizer, datasets, generator_config["beam"], label_of_mapping[args.task_name],
            generator_config["inspired_templates"], generator_config["target_number"],
            generator_config["batch_size"], generator_config["gen_max_len"],
            truncates=generator_config["truncates"], end_token=generator_config["end_token"],
            forbid_tokens=generator_config["forbid_tokens"],
            forbid_continuous_token=generator_config["forbid_continuous_token"],
            replace_token_map_list=generator_config["replace_token_map_list"]
        )

        with open(args.output_dir, "w", encoding="utf-8") as save_file:
            for text, score, _ in res_templates:
                save_file.write(f"{score}\t{text}\n")
    else:
        raise ValueError(f"TemplateGenerator `{args.template_generator}` not found")

    print(train_data)


if __name__ == '__main__':
    main()
