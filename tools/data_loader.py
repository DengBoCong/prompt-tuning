#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any


class DataLoader(abc.ABC):
    """Dataset Loader"""

    @abc.abstractmethod
    def generate_k_shot(self, **kwargs):
        raise NotImplementedError


class GLUEDataLoader(DataLoader):
    """Support GLUE Dataset"""

    def __init__(self, **kwargs) -> None:
        super(GLUEDataLoader, self).__init__()

    def generate_k_shot(self,
                        k: int,
                        data_dir: str,
                        task_name: str,
                        dev_rate: int = 1,
                        **kwargs) -> Tuple[List[str], List[str]]:
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
            line = line.strip().strip("\n")
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
    def get_label(task_name: str, line: str) -> Any:
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

    @staticmethod
    def gen_samples(task_name: str, sources: Any) -> List[Dict[str, Any]]:
        """
        :param task_name: 任务数据集名
        :param sources: 已读取数据文件的行 或 文件地址
        """
        if isinstance(sources, str):
            samples = []
            with open(sources, "r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip().strip("\n")
                    samples.append(line)
        else:
            samples = sources

        if task_name != "CoLA" and isinstance(sources, str):
            samples = samples[1:]

        dataset = []
        for sample in samples:
            sample = sample.strip().split("\t")
            if task_name == "CoLA":
                dataset.append({"label": sample[1], "text": [sample[-1]]})
            elif task_name == "MNLI":
                dataset.append({"label": sample[-1], "text": [sample[8], sample[9]]})
            elif task_name == "MRPC":
                dataset.append({"label": sample[0], "text": [sample[-2], sample[-1]]})
            elif task_name == "QNLI":
                dataset.append({"label": sample[-1], "text": [sample[1], sample[2]]})
            elif task_name == "QQP":
                dataset.append({"label": sample[-1], "text": [sample[3], sample[4]]})
            elif task_name == "RTE":
                dataset.append({"label": sample[-1], "text": [sample[1], sample[2]]})
            elif task_name == "SNLI":
                dataset.append({"label": sample[-1], "text": [sample[7], sample[8]]})
            elif task_name == "SST-2":
                dataset.append({"label": sample[-1], "text": [sample[0]]})
            elif task_name == "STS-B":
                dataset.append({"label": "0" if float(sample[-1]) < 2.5 else "1", "text": [sample[-3], sample[-2]]})
            elif task_name == "WNLI":
                dataset.append({"label": sample[-1], "text": [sample[1], sample[2]]})
            else:
                dataset.append({"label": sample[0], "text": [sample[1]]})

        return dataset


loader_map = {
    "glue": GLUEDataLoader
}
