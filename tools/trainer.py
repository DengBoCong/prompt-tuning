#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from transformers.optimization import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer import Trainer as TransformerTrainer


class Trainer(TransformerTrainer):
    def create_optimizer(self):
        """注意，这里写的是类bert结构的逻辑，其他不一样的直接覆写方法即可"""
        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if "encoder.layer" in n:
                        layer_num = int(n[n.find("encoder.layer") + 14:].split(".")[0])
                        if layer_num >= self.args.fix_layers:
                            params[n] = p
                    elif "embeddings" not in n:
                        params[n] = p
                else:
                    params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [{
                "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay
            }, {
                "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }]

            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon
            )

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps
            )

        return self.lr_scheduler
