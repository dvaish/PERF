from pathlib import Path
from typing import Tuple, List, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from tidalsim.util.pickle import load
from tidalsim.modeling.schemas import ClusteringSchema, EstimatedPerfSchema, GoldenPerfSchema


@dataclass
class PerfMetrics:
    ipc: float


# Pick which
def choose_for_rtl_sim(clustering_df: DataFrame[ClusteringSchema]) -> DataFrame[ClusteringSchema]:
    pass


# Build the performance metrics struct from a csv file dumped from RTL simulation
def parse_perf_file(
    perf_file: Path,
    detailed_warmup_insts: int,
) -> PerfMetrics:
    perf_data = pd.read_csv(perf_file)
    perf_data["insts_retired_after_interval"] = np.cumsum(perf_data["instret"])
    perf_data["insts_retired_before_interval"] = (
        perf_data["insts_retired_after_interval"] - perf_data["instret"]
    )
    # Find the first row where more than [detailed_warmup_insts] have elapsed, and only begin tracking IPC from that row onwards
    start_point = (perf_data["insts_retired_before_interval"] >= detailed_warmup_insts).idxmax()
    perf_data_visible = perf_data[start_point:]
    ipc = np.sum(perf_data_visible["instret"]) / np.sum(perf_data_visible["cycles"])
    return PerfMetrics(ipc=ipc)


# Given a dataframe with intervals and their clusters, pick the
def get_checkpoint_insts(clustering_df: DataFrame[ClusteringSchema]) -> List[int]:
    # For now, only the first interval in each cluster that has 'chosen_for_rtl_sim' == True is actually run in RTL simulation
    # TODO: fix this
    print(clustering_df.loc[clustering_df["chosen_for_rtl_sim"]].to_string())
    simulated_points: DataFrame[ClusteringSchema] = (
        clustering_df.loc[clustering_df["chosen_for_rtl_sim"]]
        .groupby("cluster_id", as_index=False)
        .nth(0)
        .sort_values("cluster_id")
    )
    return simulated_points["inst_start"].to_list()


# Returns a list of performance metrics to be used for extrapolation for each cluster_id
def get_perf_infos(
    clustering_df: DataFrame[ClusteringSchema], cluster_dir: Path, detailed_warmup_insts: int
) -> List[PerfMetrics]:
    perf_infos: List[PerfMetrics] = []
    for index, row in simulated_points.iterrows():
        perf_file = cluster_dir / "checkpoints" / f"0x80000000.{row['inst_start']}" / "perf.csv"
        perf_info = parse_perf_file(perf_file, detailed_warmup_insts)
        perf_infos.append(perf_info)
    return perf_infos


def analyze_tidalsim_results(
    run_dir: Path,
    interval_length: int,
    clusters: int,
    elf: bool,
    detailed_warmup_insts: int,
    interpolate_clusters: bool,
) -> Tuple[DataFrame[EstimatedPerfSchema], Optional[DataFrame[GoldenPerfSchema]]]:
    interval_dir = run_dir / f"n_{interval_length}_{'elf' if elf else 'spike'}"
    cluster_dir = interval_dir / f"c_{clusters}"

    clustering_df = load(cluster_dir / "clustering_df.pickle")
    perf_infos = get_perf_infos(clustering_df, cluster_dir)

    ipcs = np.array([perf.ipc for perf in perf_infos])
    estimated_perf_df: DataFrame[EstimatedPerfSchema]
    if not interpolate_clusters:
        # If we don't interpolate, we just use the IPC of the simulated point for that cluster
        estimated_perf_df = clustering_df.assign(
            est_ipc=ipcs[clustering_df["cluster_id"]],
            est_cycles=lambda x: np.round(x["instret"] * np.reciprocal(x["est_ipc"])),
        )
    else:
        # If we do interpolate, we use a weighted (by inverse L2 norm) average of the IPCs of all simulated points
        kmeans_file = cluster_dir / "kmeans_model.pickle"
        kmeans = load(kmeans_file)

        # for all points, compute norms to all centroids and store as separate vecs
        norms: np.ndarray = clustering_df["embedding"].apply(
            lambda s: np.linalg.norm(kmeans.cluster_centers_ - s, axis=1)
        )
        # combine vecs to speed up computation
        norms = np.stack(norms)  # type: ignore
        # invert to weight closer points heigher, and normalize vecs to sum to 1
        norms = 1 / norms
        weight_vecs = norms / norms.sum(axis=1, keepdims=True)
        # multiply weight vecs by ips to get weighted average
        est_ipc = weight_vecs @ ipcs
        # assign to df
        estimated_perf_df = clustering_df.assign(
            est_ipc=est_ipc,
            est_cycles=lambda x: np.round(x["instret"] * np.reciprocal(x["est_ipc"])),
        )

    golden_perf_file = run_dir / "golden" / "perf.csv"
    if golden_perf_file.exists():
        golden_perf_df = parse_golden_perf(golden_perf_file)
        return estimated_perf_df, golden_perf_df
    else:
        return estimated_perf_df, None


def parse_golden_perf(perf_csv: Path) -> DataFrame[GoldenPerfSchema]:
    perf_data = pd.read_csv(perf_csv)
    golden_perf_df: DataFrame[GoldenPerfSchema] = perf_data.assign(
        ipc=lambda x: x["instret"] / x["cycles"],
        inst_count=lambda x: np.cumsum(x["instret"].to_numpy()),
    )  # type: ignore
    return golden_perf_df
