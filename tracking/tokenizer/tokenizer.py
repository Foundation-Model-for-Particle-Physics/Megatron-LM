"""Megatron tokenizers."""

from numpy import ndarray
import pandas as pd
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer

from .module_id import ModuleIDTokenizer

class BertModuleIDTokenizer(MegatronTokenizer):
    """BERT-style tokenizer for tracking data."""

    def __init__(self, vocab_file: str, min_hits_per_track: int = 5, with_padding: bool = False, max_track_length: int = -1):
        super().__init__(vocab_file)
        self.tokenizer = ModuleIDTokenizer(vocab_file, min_hits_per_track, with_padding, max_track_length)

        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']
        self.mask_id = self.tokenizer.vocab['[MASK]']

    def add_token(self, token):
        if token not in self.vocab:
            self.inv_vocab[self.vocab_size] = token
            # self.vocab_size comes from len(vocab)
            # and it will increase as we add elements
            self.vocab[token] = self.vocab_size

    def add_additional_special_tokens(self, tokens_list):
        setattr(self, "additional_special_tokens", tokens_list)
        for value in tokens_list:
            self.add_token(value)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def tokenize(self, hits: pd.DataFrame):
        return self.tokenizer.tokenize(hits)[0]

    def decode(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def bos(self):
        """ Id of the beginning of sentence token in the vocabulary."""
        return self.tokenizer._bos_token_id

    @property
    def eos(self):
        """ Id of the end of sentence token in the vocabulary."""
        return self.tokenizer._eos_token_id

    @property
    def bos_token(self):
        """ Beginning of sentence token id """
        return self.tokenizer._bos_token

    @property
    def eos_token(self):
        """ End of sentence token id """
        return self.tokenizer._eos_token

    @property
    def additional_special_tokens(self):
        """ All the additional special tokens you may want to use (list of strings)."""
        return self._additional_special_tokens

    @property
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers)."""
        return [self.vocab.get(token) for token in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value


class GPT2ModuleIDTokenizer(MegatronTokenizer):
    """GPT2-style tokenizer for tracking data."""

    def __init__(self, vocab_file: str, min_hits_per_track: int = 5, with_padding: bool = False, max_track_length: int = -1):
        super().__init__(vocab_file)
        self.tokenizer = ModuleIDTokenizer(
            vocab_file, min_hits_per_track,
            with_padding, max_track_length, with_eod=True)

        self.eod_id = self.tokenizer.vocab["[EOD]"]

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def tokenize(self, hits: pd.DataFrame):
        return self.tokenizer.tokenize(hits)[0]

    def detokenize(self, ids: ndarray) -> str:
        return self.tokenizer.decode(ids)

    @property
    def eod(self):
        return self.eod_id
