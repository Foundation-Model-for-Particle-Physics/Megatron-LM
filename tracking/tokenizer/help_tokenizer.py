from __future__ import annotations

import collections
from pathlib import Path


def load_vocab(vocab_file: str | Path):
    """Load vocab from file."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file) as reader:
        for line in reader:
            token = line.strip()
            if not token:
                continue
            vocab[token] = index
            index += 1
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens] using the vocab."""
    return [vocab[item] for item in items]


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)
