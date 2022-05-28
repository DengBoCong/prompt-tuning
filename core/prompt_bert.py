#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers import PretrainedConfig
from typing import Optional


class BertForPromptTuning(BertPreTrainedModel):
    def __init__(self, config: PretrainedConfig, **kwarg):
        super(BertForPromptTuning, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

        # Exit early and only return mask logits.
        # For label search.
        self.return_full_softmax = kwarg.get("return_full_softmax", None)

        self.label_word_list = kwarg.get("label_word_list", None)

        # if labels be passed and num_labels == 1
        self.lower_bounds = kwarg.get("label_word_list", None)
        self.upper_bounds = kwarg.get("label_word_list", None)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                ab_pos_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None):
        if ab_pos_mask is not None:
            ab_pos_mask = ab_pos_mask.squeeze()

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = bert_outputs[0]
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), ab_pos_mask]
        prediction_mask_scores = self.cls(sequence_mask_output)

        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = None
        if self.label_word_list:
            logits = []
            for label_id in range(len(self.label_word_list)):
                logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
            logits = torch.cat(logits, -1)

        # Regression task
        if self.config.num_labels == 1:
            log_softmax = nn.LogSoftmax(dim=-1)(logits)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack(
                    [1 - (labels.view(-1) - self.lower_bounds) / (self.upper_bounds - self.lower_bounds),
                     (labels.view(-1) - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)], -1
                )
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        outputs = (logits,)
        if self.num_labels == 1:
            outputs = (
                torch.exp(logits[..., 1].unsqueeze(-1)) * (self.upper_bounds - self.lower_bounds) + self.lower_bounds,)

        return ((loss,) + outputs) if loss is not None else outputs
