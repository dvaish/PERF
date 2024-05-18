from pathlib import Path
from typing import Tuple, Optional, cast
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from sklearn.metrics import cluster

from tidalsim.util.pickle import load
from tidalsim.modeling.schemas import ClusteringSchema, EstimatedPerfSchema, GoldenPerfSchema


@dataclass
class PerfMetrics:
    cycles: int
    ipc: float


# Build the performance metrics struct from a csv file dumped from RTL simulation
# Assume perf metrics are captured without any offset from the start of the interval
# Detailed warmup is accounted for in this function
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
    return PerfMetrics(ipc=ipc, cycles=np.sum(perf_data["cycles"]))


# Fill each row in the [clustering_df] that was chosen for RTL simulation with performance numbers from RTL sim
# BUT, DO NOT perform any extrapolation yet (leave the estimated perf blank for any row without RTL sim)
def fill_perf_metrics(
    clustering_df: DataFrame[ClusteringSchema],
    cluster_dir: Path,
    detailed_warmup_insts: int,
) -> DataFrame[EstimatedPerfSchema]:
    # Augment the dataframe with zeroed out estimated perf columns
    perf_df = cast(
        DataFrame[EstimatedPerfSchema],
        clustering_df.assign(
            est_cycles_cold=np.zeros(len(clustering_df.index)),
            est_ipc_cold=np.zeros(len(clustering_df.index)),
            est_cycles_warm=np.zeros(len(clustering_df.index)),
            est_ipc_warm=np.zeros(len(clustering_df.index)),
        ),
    )
    simulated_rows = perf_df.loc[perf_df["chosen_for_rtl_sim"] == True]
    for index, row in simulated_rows.iterrows():
        checkpoint_dir = cluster_dir / "checkpoints" / f"0x80000000.{row.inst_start}"
        cold_perf_file = checkpoint_dir / "perf_cold.csv"
        warm_perf_file = checkpoint_dir / "perf_warm.csv"
        if cold_perf_file.exists():
            cold_perf = parse_perf_file(cold_perf_file, detailed_warmup_insts)
            row.est_cycles_cold = cold_perf.cycles
            row.est_ipc_cold = cold_perf.ipc
        if warm_perf_file.exists():
            warm_perf = parse_perf_file(warm_perf_file, detailed_warmup_insts)
            row.est_cycles_warm = warm_perf.cycles
            row.est_ipc_warm = warm_perf.ipc
        perf_df.iloc[index] = row
    return perf_df


# Pick which intervals we are going to simulate in RTL.
# Randomly sample from each cluster.
# AND pick the closest point to each centroid too.
def pick_intervals_for_rtl_sim(
    clustering_df: DataFrame[ClusteringSchema], points_per_cluster: int
) -> None:
    grouped_intervals = clustering_df.groupby("cluster_id")
    chosen_intervals = grouped_intervals.sample(points_per_cluster, random_state=1)
    clustering_df.loc[chosen_intervals.index, "chosen_for_rtl_sim"] = True
    c = clustering_df.groupby("cluster_id")["dist_to_centroid"].idxmin()
    clustering_df.loc[c, "chosen_for_rtl_sim"] = True


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
