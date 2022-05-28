#! -*- coding: utf-8 -*-
# Author: DengBoCong <bocongdeng@gmail.com>
#
# License: MIT License

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from core.gen_template.LM_BFF import LMBFFTemplateGenerator


__all__ = ["LMBFFTemplateGenerator"]


class TemplateGenerator(abc.ABC):
    @abc.abstractmethod
    def search_template(self, *args, **kwargs):
        raise NotImplementedError
