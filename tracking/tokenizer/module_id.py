from __future__ import annotations

from pathlib import Path

import pandas as pd

from .help_tokenizer import convert_ids_to_tokens, load_vocab


class MoudleIDTokenizer(object):
    """Tokenizer for tracking data for BERT-style models."""

    def __init__(
        self,
        vocab_file: str | Path,
        min_hits_per_track: int = 5,
        with_padding: bool = False,
        max_track_length: int = -1,
    ):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        self.pad_id = self.vocab["[PAD]"]  # padding token
        self.mask_id = self.vocab["[MASK]"]  # masking token
        self.unknown_id = self.vocab["[UNK]"]  # unknown token
        self.spacepoint_bos = -9900  # to indicate the beginning of a sequence of spacepoints
        self.spacepoint_eos = -9910
        self.spacepoint_pad_number = -9999

        self._bos_token = "[BOS]"  # begin of sequence token
        self._bos_token_id = self.vocab.get(self._bos_token)

        self._eos_token = "[EOS]"  # end of sequence token
        self._eos_token_id = self.vocab.get(self._eos_token)

        self._additional_special_tokens = []
        self.min_hits_per_track = min_hits_per_track
        self.with_padding = with_padding
        self.max_track_length = max_track_length
        if with_padding:
            assert self.max_track_length > 0, "Padding is enabled, but max_track_length is not set."

    @property
    def vocab_size(self):
        return len(self.vocab)

    def tokenize(self, hits: pd.DataFrame):
        """Tokenize tracks in one event."""
        assert isinstance(hits, pd.DataFrame)
        needed_columns = ["hit_id", "umid", "particle_id", "nhits", "p_pt"]
        for col in needed_columns:
            assert col in hits.columns, f"hits must contain column: {col}"

        # remove tracks with less than min_hits_per_track hits
        if self.min_hits_per_track > 0:
            hits = hits[hits.nhits >= self.min_hits_per_track]

        # remove noise hits
        hits = hits[hits.particle_id != 0]

        # sort particles by their pT by descending order
        vlid_groups = hits.groupby("particle_id")
        sorted_pids = vlid_groups.p_pt.mean().sort_values(ascending=False).index

        lengths = []
        tracks = []
        all_track_hit_ids = []

        # loop over all tracks in the event
        for vlid in sorted_pids:
            track_info = vlid_groups.get_group(vlid)
            track = [
                self.vocab["[BOS]"],
                *track_info.umid.map(lambda x: self.vocab.get(str(x))).to_list(),
                self.vocab["[EOS]"],
            ]
            track_hit_ids = [self.spacepoint_bos, *track_info.hit_id.to_list(), self.spacepoint_eos]

            if self.with_padding:
                track_len = len(track)
                assert (
                    track_len <= self.max_track_length
                ), f"Track length {track_len} is greater than the specified max track length."
                padding_length = self.max_track_length - len(track)
                track = track + [self.pad_id] * padding_length
                track_hit_ids = track_hit_ids + [self.spacepoint_pad_number] * padding_length

            lengths.append(len(track))
            tracks.append(track)
            all_track_hit_ids.append(track_hit_ids)
        # flatten the list
        track = [item for sublist in tracks for item in sublist]
        track_hit_ids = [item for sublist in all_track_hit_ids for item in sublist]
        return track, lengths, track_hit_ids

    def decode(self, ids):
        tokens = convert_ids_to_tokens(ids)
        return " ".join(tokens)

    def decode_token_ids(self, token_ids):
        tokens = convert_ids_to_tokens(token_ids)
        exclude_list = ["[PAD]", "[CLS]", "[SEP]"]
        tokens = [token for token in tokens if token not in exclude_list]
        return " ".join(tokens)

