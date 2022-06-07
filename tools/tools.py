#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from typing import List, Any, Dict


def resize_token_type_embeddings(model: Any, new_num_types: int, random_segment: bool):
    """ Resize the segment (token type) embeddings for BERT """
    if hasattr(model, "bert"):
        old_token_type_embeddings = model.bert.embeddings.token_type_embeddings
    else:
        raise NotImplementedError
    new_token_type_embeddings = nn.Embedding(new_num_types, old_token_type_embeddings.weight.size(1))
    if not random_segment:
        new_token_type_embeddings.weight.data[:old_token_type_embeddings.weight.size(0)] = old_token_type_embeddings.weight.data

    model.config.type_vocab_size = new_num_types
    if hasattr(model, "bert"):
        model.bert.embeddings.token_type_embeddings = new_token_type_embeddings
    else:
        raise NotImplementedError


def tokenize_multipart_input_for_gen_en(input_text_list: List[str],
                                        max_length: int,
                                        tokenizer: Any,
                                        template: str,
                                        label_word_list: List[str],
                                        **kwargs) -> Dict[str, Any]:
    """Concatenate all sentences and prompts based on the provided template.
        Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
        *xx* represent variables:
            *cls*: cls_token
            *mask*: mask_token
            *sep*: sep_token
            *sep+*: sep_token, also means +1 for segment id
            *sent_i*: sentence i (input_text_list[i])
            *sent-_i*: same as above, but delete the last token
            *sentl_i*: same as above, but use lower case for the first word
            *sentl-_i*: same as above, but use lower case for the first word and delete the last token
            *+sent_i*: same as above, but add a space before the sentence
            *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
            *label_i*: label_word_list[i]

        Use "_" to replace space.
        PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.

        kwargs: first_sent_limit、other_sent_limit、truncate_head
    :param input_text_list: 输入文本list
    :param max_length: 最大长度
    :param tokenizer: 编码器
    :param template: template
    :param label_word_list: label word list
    """
    assert template is not None

    input_ids = []
    attention_mask = []
    token_type_ids = []
    mask_pos = None

    special_token_mapping = {
        "cls": tokenizer.cls_token_id, "mask": tokenizer.mask_token_id,
        "sep": tokenizer.sep_token_id, "sep+": tokenizer.sep_token_id,
    }
    template_list = template.split("*")  # Get variable list in the template
    segment_id = 0  # Current segment id. Segment id +1 if encountering sep+.

    for part_id, part in enumerate(template_list):
        new_tokens, segment_plus_1_flag = [], False
        if part in special_token_mapping:
            if part == "cls" and "T5" in type(tokenizer).__name__:
                # T5 does not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
            if part == "sep+":
                segment_plus_1_flag = True
        elif part[:6] == "label_":
            # Note that label_word_list already has extra space, so do not add more space ahead of it.
            label_id = int(part.split("_")[1])
            label_word = label_word_list[label_id]
            new_tokens.append(label_word)
        elif part[:5] == "sent_":
            sent_id = int(part.split("_")[1])
            new_tokens += tokenizer.encode(input_text_list[sent_id], add_special_tokens=False)
        elif part[:6] == "+sent_":
            # Add space
            sent_id = int(part.split("_")[1])
            new_tokens += tokenizer.encode(" " + input_text_list[sent_id], add_special_tokens=False)
        elif part[:6] == "sent-_":
            # Delete the last token
            sent_id = int(part.split("_")[1])
            new_tokens += tokenizer.encode(input_text_list[sent_id][:-1], add_special_tokens=False)
        elif part[:6] == "sentl_":
            # Lower case the first token
            sent_id = int(part.split("_")[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += tokenizer.encode(text, add_special_tokens=False)
        elif part[:7] == "+sentl_":
            # Lower case the first token and add space
            sent_id = int(part.split("_")[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += tokenizer.encode(" " + text, add_special_tokens=False)
        elif part[:7] == "sentl-_":
            # Lower case the first token and discard the last token
            sent_id = int(part.split("_")[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += tokenizer.encode(text[:-1], add_special_tokens=False)
        elif part[:6] == "sentu_":
            # Upper case the first token
            sent_id = int(part.split("_")[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += tokenizer.encode(text, add_special_tokens=False)
        elif part[:7] == "+sentu_":
            # Upper case the first token and add space
            sent_id = int(part.split("_")[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += tokenizer.encode(" " + text, add_special_tokens=False)
        else:
            # Just natural language prompt
            part = part.replace("_", " ")
            # handle special case when T5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer._convert_token_to_id(part))
            else:
                new_tokens += tokenizer.encode(part, add_special_tokens=False)

        if part[:4] == "sent" or part[1:5] == "sent":
            # If this part is the sentence, limit the sentence length
            sent_id = int(part.split("_")[1])
            if sent_id == 0:
                if kwargs.get("first_sent_limit", None) is not None:
                    new_tokens = new_tokens[:kwargs["first_sent_limit"]]
            else:
                if kwargs.get("other_sent_limit", None) is not None:
                    new_tokens = new_tokens[:kwargs["other_sent_limit"]]

        input_ids += new_tokens
        attention_mask += [1 for i in range(len(new_tokens))]
        token_type_ids += [segment_id for i in range(len(new_tokens))]

        if segment_plus_1_flag:
            segment_id += 1

    # Padding
    if kwargs.get("first_sent_limit", None) is not None and len(input_ids) > max_length:
        print(f"Input exceeds max_length limit: {tokenizer.decode(input_ids)}")

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(0)
        token_type_ids.append(0)

    # Truncate
    if len(input_ids) > max_length:
        if kwargs.get("truncate_head", None):
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    mask_pos = [input_ids.index(tokenizer.mask_token_id)]
    # Make sure that the masked position is inside the max_length
    assert mask_pos[0] < max_length

    result = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "BERT" in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result["token_type_ids"] = token_type_ids

    result["mask_pos"] = mask_pos

    return result


# 调用工具融合
multipart_map = {
    "glue": tokenize_multipart_input_for_gen_en
}
