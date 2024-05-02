"""Preprocessing the Tracking ML data for pretraining BERT or GPT in Megatron-LM.

It reads the parquet files with ActsReader, tokenizes the data using TrackDataTokenzier, and builds
the indexed dataset with MMapIndexedDatasetBuilder.
"""

import concurrent.futures
import logging
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import yaml
from tqdm import tqdm

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
from tracking.datamodules.odd_reader import ActsReader
from tracking.tokenizer.module_id import ModuleIDTokenizer

logging.basicConfig(
    filename="create_data.log",
    encoding="utf-8",
    level=logging.DEBUG,
    format="[%(asctime)s %(levelname)s %(name)s]: %(message)s",
    datefmt="%Y/%m/%d %I:%M:%S %p",
)
logger = logging.getLogger(__name__)


@dataclass
class TrackingDataConfig:
    """Configuration for preprocessing the tracking data."""

    data_path: str
    output_prefix: str
    vocab_file: str
    min_hits_per_track: int
    with_padding: bool
    max_track_length: int
    num_workers: int = 1
    max_evts: int = -1


def process_one_event(evt_idx: int, reader, tokenizer):
    """Process one event.

    One event is a document. Use the Megatron-LM style.
    """
    hits = reader.read_event(evt_idx)[0]
    hits = hits.rename(columns={"measurement_id": "hit_id"})
    umid = hits.umid.map(reader.umid_dict_inv)
    hits["umid"] = umid
    tracks, lengths, track_ids = tokenizer.tokenize(hits)
    num_tracks = len(lengths)
    return tracks, lengths, track_ids, num_tracks


def preprocess_tracking_data(config: TrackingDataConfig):
    """Preprocess the tracking data for pretraining BERT.

    Args:
        data_path (str): Path to the tracking data.
        output_path (str): Path to save the preprocessed data.
    """
    # Load the tracking data
    reader = ActsReader(inputdir=config.data_path)
    tokenizer = ModuleIDTokenizer(
        config.vocab_file,
        config.min_hits_per_track,
        config.with_padding,
        config.max_track_length,
    )

    bin_path = f"{config.output_prefix}.bin"
    builder = IndexedDatasetBuilder(bin_path, dtype=np.int32, multimodal=False)

    end_evt = reader.nevts if config.max_evts < 0 else config.max_evts

    # process all events with multiprocessing
    track_id_file = open(f"{config.output_prefix}.track_ids.bin", "wb")
    num_tracks_per_evt = []
    if config.num_workers > 1:
        process_one_event_fn = partial(process_one_event, reader=reader, tokenizer=tokenizer)

        with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            futures = [executor.submit(process_one_event_fn, evt_idx) for evt_idx in range(end_evt)]

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=end_evt, desc="Processing events"
            ):
                tracks, lengths, track_ids, num_tracks = future.result()
                track_tensor = torch.tensor(tracks, dtype=torch.int16)
                builder.add_document(track_tensor, lengths)

                track_ids_np = np.array(track_ids, dtype=np.int64)
                track_id_file.write(track_ids_np.tobytes(order="C"))

                num_tracks_per_evt.append(num_tracks)
    else:
        for evt_idx in tqdm(range(end_evt), desc="Processing events", total=end_evt):
            tracks, lengths, track_ids, num_tracks = process_one_event(evt_idx)
            tracks = torch.tensor(tracks, dtype=torch.int16)
            builder.add_document(tracks, lengths)

            track_ids_np = np.array(track_ids, dtype=np.int64)
            track_id_file.write(track_ids_np.tobytes(order="C"))

            num_tracks_per_evt.append(num_tracks)

    builder.finalize(f"{config.output_prefix}.idx")
    track_id_file.close()
    logger.info(f"Preprocessed data saved to {config.output_prefix}.bin")

    with open(f"{config.output_prefix}.num_tracks.bin", "wb") as f:
        f.write(np.array(num_tracks_per_evt, dtype=np.int16).tobytes(order="C"))


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config")
    args = arg_parser.parse_args()
    with open(args.config) as f:
        config = TrackingDataConfig(**yaml.safe_load(f))

    print(config)
    preprocess_tracking_data(config)
