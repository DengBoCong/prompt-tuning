#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class GLUEDataLoader(object):
    """Support GLUE Dataset"""

    def __init__(self, seed: int, **kwargs) -> None:
        """
        :param seed: 随机种子
        """
        super(GLUEDataLoader, self).__init__()
        self.seed = seed
        np.random.seed(seed)

    def generate_k_shot(self, k: int, data_dir: str, task_name: str, dev_rate: int = 1) -> Tuple[List[str], List[str]]:
        """
        :param k: 类内sampling num
        :param data_dir: 数据集路径
        :param task_name: 任务数据集名
        :param dev_rate: dev:train 比例
        """
        dataset = self.load_std_dataset(data_dir, task_name)

        if task_name in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
            # GLUE style
            train_header, train_lines = self.split_header(task_name, dataset["train"])
            np.random.shuffle(train_lines)
        else:
            # Other datasets, default DataFrame
            train_lines = dataset["train"].values.tolist()
            np.random.shuffle(train_lines)

        # Get label list for balanced sampling
        label_list = {}
        for line in train_lines:
            label = self.get_label(task_name, line)
            if label not in label_list:
                label_list[label] = [line]
            else:
                label_list[label].append(line)

        train_data, dev_data = [], []
        for label in label_list:
            train_data.extend(label_list[label][:k])
        for label in label_list:
            dev_data.extend(label_list[label][k:k * (dev_rate + 1)])

        return train_data, dev_data

    @staticmethod
    def split_header(task_name: str, lines: List[str]) -> Tuple[List[str], List[str]]:
        """ 返回文件头
        :param task_name: 任务数据集名
        :param lines: 已读取数据文件的所有行
        """
        if task_name in ["CoLA"]:
            return [], lines
        elif task_name in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI"]:
            return lines[0:1], lines[1:]
        else:
            raise ValueError("Unknown GLUE task.")

    @staticmethod
    def load_std_dataset(data_dir: str, task_name: str, splits: List[str] = None) -> Dict[str, List[str]]:
        """ 加载预设标准数据集
        :param data_dir: 数据集路径
        :param task_name: 任务数据集名
        :param splits: 数据集分类
        """
        dataset = {}
        if task_name in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
            # GLUE style (tsv)
            if not splits:
                if task_name == "MNLI":
                    splits = ["train", "dev_matched", "dev_mismatched"]
                else:
                    splits = ["train", "dev"]
            for split in splits:
                filename = os.path.join(data_dir, f"{split}.tsv")
                with open(filename, "r") as file:
                    lines = file.readlines()
                dataset[split] = lines
        else:
            # Other datasets (csv)
            splits = splits if splits else ["train", "test"]
            for split in splits:
                filename = os.path.join(data_dir, f"{split}.csv")
                dataset[split] = pd.read_csv(filename, header=None)

        return dataset

    @staticmethod
    def get_label(task_name: str, line: str) -> Optional[str, int]:
        """
        :param task_name: 任务数据集名
        :param line: 已读取数据文件的行
        """
        if task_name in ["MNLI", "MRPC", "QNLI", "QQP", "RTE", "SNLI", "SST-2", "STS-B", "WNLI", "CoLA"]:
            # GLUE style
            line = line.strip().split("\t")
            if task_name == "CoLA":
                return line[1]
            elif task_name == "MNLI":
                return line[-1]
            elif task_name == "MRPC":
                return line[0]
            elif task_name == "QNLI":
                return line[-1]
            elif task_name == "QQP":
                return line[-1]
            elif task_name == "RTE":
                return line[-1]
            elif task_name == "SNLI":
                return line[-1]
            elif task_name == "SST-2":
                return line[-1]
            elif task_name == "STS-B":
                return 0 if float(line[-1]) < 2.5 else 1
            elif task_name == "WNLI":
                return line[-1]
            else:
                raise NotImplementedError
        else:
            return line[0]




