"""Read csv files obtained from ACTS with the Open Data Detector.

The files are CSV files organized as follows:
acts/event000001000-hits.csv
acts/event000001000-measurements.csv
acts/event000001000-meas2hits.csv
acts/event000001000-spacepoints.csv
acts/event000001000-particles_final.csv
acts/event000001000-cells.csv
"""

from __future__ import annotations

import itertools
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def make_true_edges(hits: pd.DataFrame):
    """Make true edges from the hits dataframe."""
    hit_list = (
        hits.groupby(["particle_id", "geometry_id"], sort=False)["index"]
        .agg(list)
        .groupby(level=0)
        .agg(list)
    )

    e = []
    for row in hit_list.to_numpy():
        for i, j in zip(row[0:-1], row[1:]):
            e.extend(list(itertools.product(i, j)))

    return np.array(e).T

logger = logging.getLogger(__name__)


class ActsReader(object):
    def __init__(self, inputdir: str | Path,
                 outputdir: str | None = None,
                 outname_prefix: str = "",
                 overwrite: bool = False,
                 spname: str = "spacepoint"):
        """Initialize the reader."""
        if not inputdir:
            logger.warning("No input directory specified, using current directory.")
            inputdir = Path.cwd()

        self.inputdir = Path(inputdir) if isinstance(inputdir, str) else inputdir
        if not self.inputdir.exists() or not self.inputdir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {inputdir}")

        logger.info(f"input directory: {self.inputdir}")

        self.outputdir = Path(outputdir) if outputdir else self.inputdir / "processed_data"
        self.outputdir.mkdir(parents=True, exist_ok=True)

        self.outname_prefix = outname_prefix + "_" if outname_prefix else ""

        self.overwrite = overwrite
        self.spname = spname

        # count how many events in the directory
        all_evts = list(Path.rglob(Path(self.inputdir), f"*event*-{spname}.csv"))
        self.is_parquet = len(all_evts) == 0

        if self.is_parquet:
            pattern = "([0-9]*).parquet"
            self.all_evtids = sorted([
                int(re.search(pattern, Path(x).name).group(1).strip())
                for x in Path.glob(Path(self.inputdir) / "particles", "*.parquet")
            ])
            self.nevts = len(self.all_evtids)
        else:
            pattern = f"event([0-9]*)-{spname}.csv"
            self.all_evtids = sorted([
                int(re.search(pattern, Path(x).name).group(1).strip()) for x in all_evts
            ])
            self.nevts = len(self.all_evtids)

        print(f"total {self.nevts} events in directory: {self.inputdir}")
        self.all_event_filenames = all_evts

        if self.nevts == 0:
            raise ValueError(f"No events found in {self.inputdir}")

        # load detector info
        detector = pd.read_csv(Path(self.inputdir).parent.parent / "detector.csv")
        self.detector = detector
        self.build_detector_vocabulary(detector)

    def build_detector_vocabulary(self, detector: pd.DataFrame):
        """Build the detector vocabulary for the reader."""
        assert "geometry_id" in detector.columns, "geometry_id not in detector.csv"

        detector_umid = detector.geometry_id.unique()
        umid_dict = {}
        index = 1
        for i in detector_umid:
            umid_dict[i] = index
            index += 1
        self.umid_dict = umid_dict
        self.num_modules = len(detector_umid)
        # Inverting the umid_dict
        self.umid_dict_inv = {v: k for k, v in umid_dict.items()}

    def read_csv_event(
        self, evt_idx: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Read one event from the input directory through the event index.

        Return:
            hits: pd.DataFrame, hits information
            particles: pd.DataFrame, particles information
            true_edges: np.ndarray, true edges
        """
        if (evt_idx is None or evt_idx < 0) and self.nevts > 0:
            evtid = self.all_evtids[0]
            print(f"read event {evtid}.")
        else:
            evtid = self.all_evtids[evt_idx]

        # construct file names for each csv file for this event
        prefix = self.all_event_filenames[evt_idx][: -len(self.spname) - 5]
        hit_fname = f"{prefix}-hits.csv"
        measurements_fname = f"{prefix}-measurements.csv"
        measurements2hits_fname = f"{prefix}-measurement-simhit-map.csv"
        sp_fname = f"{prefix}-{self.spname}.csv"
        p_name = f"{prefix}-particles_final.csv"

        # read hit files
        hits = pd.read_csv(hit_fname)
        hits = hits[hits.columns[:-1]]
        hits = hits.reset_index().rename(columns={"index": "hit_id"})

        # read measurements, maps to hits, and spacepoints
        measurements = pd.read_csv(measurements_fname)
        meas2hits = pd.read_csv(measurements2hits_fname)
        sp = pd.read_csv(sp_fname)

        # add geometry_id to space points
        vlid_groups = sp.groupby(["geometry_id"])
        try:
            sp = pd.concat([
                vlid_groups.get_group(x).assign(umid=self.umid_dict[x]) for x in vlid_groups.groups
            ])
        except KeyError:
            logger.exception(f"No geometry_id in spacepoints in file {prefix}")
            return None, None, None

        logger.info(sp.columns)

        # read particles and add more variables for performance evaluation
        particles = pd.read_csv(p_name)
        pt = np.sqrt(particles.px**2 + particles.py**2)
        momentum = np.sqrt(pt**2 + particles.pz**2)
        theta = np.arccos(particles.pz / momentum)
        eta = -np.log(np.tan(0.5 * theta))
        radius = np.sqrt(particles.vx**2 + particles.vy**2)
        particles = particles.assign(p_pt=pt, p_radius=radius, p_eta=eta)

        # read cluster information
        cell_fname = f"{prefix}-cells.csv"
        cells = pd.read_csv(cell_fname)
        if cells.shape[0] > 0:
            # calculate cluster shape information
            direction_count_u = cells.groupby(["hit_id"]).channel0.agg(["min", "max"])
            direction_count_v = cells.groupby(["hit_id"]).channel1.agg(["min", "max"])
            nb_u = direction_count_u["max"] - direction_count_u["min"] + 1
            nb_v = direction_count_v["max"] - direction_count_v["min"] + 1
            hit_cells = cells.groupby(["hit_id"]).value.count().to_numpy()
            hit_value = cells.groupby(["hit_id"]).value.sum().to_numpy()
            # as I don't access to the rotation matrix and the pixel pitches,
            # I can't calculate cluster's local/global position
            sp = sp.assign(len_u=nb_u, len_v=nb_v, cell_count=hit_cells, cell_val=hit_value)

        sp_hits = sp.merge(meas2hits, on="measurement_id", how="left").merge(
            hits[["hit_id", "particle_id"]], on="hit_id", how="left"
        )
        sp_hits = sp_hits.merge(
            particles[["particle_id", "vx", "vy", "vz", "p_pt", "p_eta"]],
            on="particle_id",
            how="left",
        )
        num_hits = sp_hits.groupby(["particle_id"]).hit_id.count()
        sp_hits = sp_hits.merge(num_hits.to_frame(name="nhits"), on="particle_id", how="left")

        r = np.sqrt(sp_hits.x**2 + sp_hits.y**2)
        phi = np.arctan2(sp_hits.y, sp_hits.x)
        sp_hits = sp_hits.assign(r=r, phi=phi)

        sp_hits = sp_hits.assign(
            R=np.sqrt(
                (sp_hits.x - sp_hits.vx) ** 2
                + (sp_hits.y - sp_hits.vy) ** 2
                + (sp_hits.z - sp_hits.vz) ** 2
            )
        )
        sp_hits = sp_hits.sort_values("R").reset_index(drop=True).reset_index(drop=False)

        true_edges = make_true_edges(sp_hits)
        self.particles = particles
        self.clusters = measurements
        self.spacepoints = sp_hits
        self.true_edges = true_edges
        self.evtid = evtid

        return sp_hits, particles, true_edges

    def read_parquet_event(
        self, evt_idx: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Read one event from the input directory through the event index.

        Return:
            hits: pd.DataFrame, hits information
            particles: pd.DataFrame, particles information
            true_edges: np.ndarray, true edges
        """
        if (evt_idx is None or evt_idx < 0) and self.nevts > 0:
            evtid = self.all_evtids[0]
            print(f"read event {evtid}.")
        else:
            evtid = self.all_evtids[evt_idx]

        # construct file names for each csv file for this event
        sp_fname = self.inputdir / "spacepoints" / f"{evt_idx}.parquet"
        p_fname = self.inputdir / "particles" / f"{evt_idx}.parquet"
        edge_fname = self.inputdir / "true_edges" / f"{evt_idx}.parquet"

        # read space points
        sp_hits = pq.read_table(sp_fname).to_pandas()
        particles = pq.read_table(p_fname).to_pandas()
        true_edges = pq.read_table(edge_fname).to_pandas()

        self.spacepoints = sp_hits
        self.particles = particles
        self.true_edges = true_edges
        return sp_hits, particles, true_edges

    def read_event(
        self, evt_idx: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """Read one event from the input directory through the event index.

        Return:
            hits: pd.DataFrame, hits information
            particles: pd.DataFrame, particles information
            true_edges: np.ndarray, true edges
        """
        if self.is_parquet:
            return self.read_parquet_event(evt_idx)

        return self.read_csv_event(evt_idx)
