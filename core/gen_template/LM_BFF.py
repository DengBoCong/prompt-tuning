#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from core.gen_template import TemplateGenerator
from transformers import AutoModel
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List, Dict, Any, Callable


class LMBFFTemplateGenerator(TemplateGenerator):
    def __init__(self) -> None:
        super(LMBFFTemplateGenerator, self).__init__()

    def search_template(self,
                        model: Any,
                        tokenizer: Any,
                        dataset: List[Dict[str, Any]],
                        beam: int,
                        label_mapping: Dict[Any, Any],
                        inspired_templates: List[str],
                        target_number: int,
                        batch_size: int = 32,
                        gen_max_len: int = 20,
                        label: Any = None,
                        truncates: List[str] = None,
                        first_mask_token: str = "<extra_id_0>",
                        end_token: str = "</s>",
                        template_encoder: Callable[[str, List[str], Any, Any, Dict[Any, Any]], List[int]] = None,
                        forbid_tokens: List[int] = None,
                        forbid_continuous_token: List[int] = None,
                        replace_token_map: Dict[str, str] = None,
                        *args, **kwargs):
        """
        :param model: 用来生成prompt的模型
        :param tokenizer: 编码器
        :param dataset: 载入的数据，一般形式: {"label": line[-1], "text": [line[8], line[9]]}
        :param beam: beam search size
        :param label_mapping: 标签描述映射，形如将数字标签映射成文字标签，一般形式: {0: 'terrible', 1: 'great'}
        :param inspired_templates: 启发输入模板
        :param target_number: 生成目标词的范围
        :param batch_size: T5推理的batch size
        :param gen_max_len: 生成内容的最大长度
        :param label: 指定某个label的文本才进行template生成
        :param truncates: 截断，配合inspired_templates定制化，数量需要一致
        :param first_mask_token: 首个用于生成位置mask的token
        :param end_token: 结束token
        :param template_encoder: 和template匹配的文本编码方法
        :param forbid_tokens: 跳过一些特定的token，如"..."
        :param forbid_continuous_token: 跳过一些不可连续生成的token，如标点符号
        :param replace_token_map: 用于替换生成的文本中部分的token
        """
        if isinstance(model, str):
            model = AutoModel.from_pretrained(model)
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        res_templates = []

        assert len(truncates) == len(inspired_templates)
        for inspired_template, truncate in zip(inspired_templates, truncates):
            generate_text = self.generate(dataset, inspired_template, model, tokenizer, target_number, label_mapping,
                                          beam, batch_size, gen_max_len, label, truncate, first_mask_token, end_token,
                                          template_encoder, forbid_tokens, forbid_continuous_token)[:beam // 2]

            if replace_token_map:
                for text, score, text_id in generate_text:
                    for ori_token, repl_token in replace_token_map.items():
                        text.replace(ori_token, repl_token)
                    res_templates.append((text, score, text_id))
            else:
                res_templates = generate_text

    def generate(self,
                 dataset: List[Dict[str, Any]],
                 inspired_template: str,
                 model: Any,
                 tokenizer: Any,
                 target_number: int,
                 label_mapping: Dict[Any, Any],
                 beam: int,
                 batch_size: int = 32,
                 gen_max_len: int = 20,
                 label: Any = None,
                 truncate: str = None,
                 first_mask_token: str = "<extra_id_0>",
                 end_token: str = "</s>",
                 template_encoder: Callable[[str, List[str], Any, Any, Dict[Any, Any]], List[int]] = None,
                 forbid_tokens: List[int] = None,
                 forbid_continuous_token: List[int] = None):
        """
        :param dataset: 载入的数据，一般形式: {"label": line[-1], "text": [line[8], line[9]]}
        :param inspired_template: 启发输入模板
        :param model: 用来生成prompt的模型
        :param tokenizer: 编码器
        :param target_number: 生成目标词的范围
        :param label_mapping: 标签描述映射，形如将数字标签映射成文字标签，一般形式: {0: 'terrible', 1: 'great'}
        :param beam: beam search size
        :param batch_size: T5推理的batch size
        :param gen_max_len: 生成内容的最大长度
        :param label: 指定某个label的文本才进行template生成
        :param truncate: 截断
        :param first_mask_token: 首个用于生成位置mask的token
        :param end_token: 结束token
        :param template_encoder: 和template匹配的文本编码方法
        :param forbid_tokens: 跳过一些特定的token，如"..."
        :param forbid_continuous_token: 跳过一些不可连续生成的token，如标点符号
        """
        if template_encoder is None:
            template_encoder = self.encode_text_by_template

        input_tensors, max_length = [], 0

        for item in dataset:
            if label is None or item["label"] == label:
                input_text = template_encoder(inspired_template, item["text"], item["label"], tokenizer, label_mapping)
                if truncate is not None:
                    if truncate == "head":
                        input_text = input_text[-256:]
                    elif truncate == "tail":
                        input_text = input_text[:256]
                    else:
                        raise NotImplementedError
                input_ids = torch.tensor(input_text).long()
                max_length = max(max_length, input_ids.size(-1))
                input_tensors.append(input_ids)

        # Concatenate inputs as a batch
        input_ids = torch.zeros((len(input_tensors), max_length)).long()
        attention_mask = torch.zeros((len(input_tensors), max_length)).long()
        for i in range(len(input_tensors)):
            input_ids[i, :input_tensors[i].size(-1)] = input_tensors[i]
            attention_mask[i, :input_tensors[i].size(-1)] = 1

        assert len(input_tensors) > 0

        start_mask = tokenizer._convert_token_to_id(first_mask_token)
        ori_decoder_input_ids = torch.zeros((input_ids.size(0), gen_max_len)).long()
        ori_decoder_input_ids[..., 0] = model.config.decoder_start_token_id

        current_output = [{"decoder_input_ids": ori_decoder_input_ids, "ll": 0, "output_id": 1, "output": []}]
        for i in tqdm(range(gen_max_len - 2)):
            new_current_output = []
            for item in current_output:
                if item["output_id"] > target_number:
                    # Enough contents
                    new_current_output.append(item)
                    continue
                decoder_input_ids = item["decoder_input_ids"]

                # Forward
                turn = input_ids.size(0) // batch_size
                if input_ids.size(0) % batch_size != 0:
                    turn += 1
                aggr_output = []
                for t in range(turn):
                    start = t * batch_size
                    end = min((t + 1) * batch_size, input_ids.size(0))

                    with torch.no_grad():
                        aggr_output.append(model(input_ids[start:end], attention_mask=attention_mask[start:end],
                                                 decoder_input_ids=decoder_input_ids[start:end])[0])
                aggr_output = torch.cat(aggr_output, 0)

                # Gather results across all input sentences, and sort generated tokens by log likelihood
                aggr_output = aggr_output.mean(0)
                log_denominator = torch.logsumexp(aggr_output[i], -1).item()
                ids = list(range(model.config.vocab_size))
                ids.sort(key=lambda x: aggr_output[i][x].item(), reverse=True)
                ids = ids[:beam + 3]

                for word_id in ids:
                    output_id = item["output_id"]

                    check = True
                    # random stop and finish one part
                    if word_id == start_mask - output_id or word_id == tokenizer._convert_token_to_id(end_token):
                        output_id += 1

                    output_text = item["output"] + [word_id]
                    ll = item["ll"] + aggr_output[i][word_id] - log_denominator
                    new_decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.size())
                    new_decoder_input_ids[:] = decoder_input_ids
                    new_decoder_input_ids[..., i + 1] = word_id

                    forbid_tokens = [3, 19794, 22354] if forbid_tokens is None else forbid_tokens
                    if word_id in forbid_tokens:
                        check = False

                    # Forbid continuous
                    forbid_continuous_token = [5] if forbid_continuous_token is None else forbid_continuous_token
                    if len(output_text) > 1 and output_text[-2] == output_text[-1] and \
                            output_text[-1] in forbid_continuous_token:
                        check = False

                    if check:
                        # Add new results to beam search pool
                        new_item = {"decoder_input_ids": new_decoder_input_ids,
                                    "ll": ll, "output_id": output_id, "output": output_text}
                        new_current_output.append(new_item)

            if len(new_current_output) == 0:
                break

            new_current_output.sort(key=lambda x: x["ll"], reverse=True)
            new_current_output = new_current_output[:beam]
            current_output = new_current_output

        result = []
        for item in current_output:
            generate_text = ""
            for token in item["output"]:
                generate_text += tokenizer._convert_id_to_token(token)

            result.append((generate_text, item["ll"].item(), item["output"]))

        return result

    @staticmethod
    def encode_text_by_template(inspired_template: str,
                                input_text_tuple: List[str],
                                label: Any, tokenizer: Any,
                                label_mapping: Dict[Any, Any]) -> List[int]:
        """ 给英文T5用的编码规则，其他特别要求的模型自行定义
        :param inspired_template: 启发输入模板
        :param input_text_tuple: 数据文本
        :param label: 标签
        :param tokenizer: 编码器
        :param label_mapping: 标签描述映射，形如将数字标签映射成文字标签，一般形式: {0: 'terrible', 1: 'great'}
        :return:
        """

        def enc(token: str):
            return tokenizer.encode(token, add_special_tokens=False)

        special_token_mapping = {"cls": tokenizer.cls_token_id, "mask": tokenizer.mask_token_id,
                                 "sep": tokenizer.sep_token_id, "sep+": tokenizer.sep_token_id}
        for index in range(10):
            special_token_mapping[f"<extra_id_{index}>"] = tokenizer._convert_token_to_id(f"<extra_id_{index}>")
        inspired_template_list = inspired_template.split('*')
        input_ids = []
        for part in inspired_template_list:
            new_tokens = []
            if part in special_token_mapping:
                if part == "cls" and "T5" in type(tokenizer).__name__:
                    # T5 does not have cls token
                    continue
                new_tokens.append(special_token_mapping[part])
            elif part[:5] == "label":
                new_tokens += enc(" " + label_mapping[label])
            elif part[:5] == "sent_":
                sent_id = int(part.split("_")[1])
                new_tokens += enc(input_text_tuple[sent_id])
            elif part[:6] == "+sent_":
                sent_id = int(part.split("_")[1])
                new_tokens += enc(" " + input_text_tuple[sent_id])  # add space
            elif part[:6] == "sent-_":
                # Delete the last token
                sent_id = int(part.split("_")[1])
                new_tokens += enc(input_text_tuple[sent_id][:-1])
            elif part[:7] == "+sentl_":
                # Lower case the first token
                sent_id = int(part.split("_")[1])
                text = input_text_tuple[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(" " + text)
            elif part[:7] == "+sentu_":
                # Upper case the first token
                sent_id = int(part.split("_")[1])
                text = input_text_tuple[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(' ' + text)
            elif part[:6] == "sentl_":
                # Lower case the first token
                sent_id = int(part.split("_")[1])
                text = input_text_tuple[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text)
            elif part[:6] == "sentu_":
                # Lower case the first token
                sent_id = int(part.split("_")[1])
                text = input_text_tuple[sent_id]
                text = text[:1].upper() + text[1:]
                new_tokens += enc(text)
            elif part[:7] == "sentl-_":
                # Lower case the first token
                sent_id = int(part.split("_")[1])
                text = input_text_tuple[sent_id]
                text = text[:1].lower() + text[1:]
                new_tokens += enc(text[:-1])
            else:
                part = part.replace("_", " ")  # there cannot be space in command, so use "_" to replace space
                # handle special case when t5 tokenizer might add an extra space
                if len(part) == 1:
                    new_tokens.append(tokenizer._convert_token_to_id(part))
                else:
                    new_tokens += enc(part)

            input_ids += new_tokens
        return input_ids
