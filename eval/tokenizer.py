"""
Tokenizer to support Chinese
"""
import jieba
from abc import abstractmethod
from modelscope import AutoTokenizer


def singleton(cls):
    instances = {}

    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper


class Tokenizer(object):

    @abstractmethod
    def tokenize(self, text):
        pass


@singleton
class JiebaTokenizer(Tokenizer):

    def __init__(self, cut_all=False):
        jieba.initialize()
        self.cut_all = cut_all

    def tokenize(self, text):
        return list(jieba.cut(text, cut_all=self.cut_all))


@singleton
class MsTokenizer(Tokenizer):

    def __init__(self, name="TongyiFinance/Tongyi-Finance-14B"):
        self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)

    def tokenize(self, text):
        return self.tokenizer(text).get("input_ids", [])
