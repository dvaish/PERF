from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Optional, List
import pprint

from pandera.typing import DataFrame
import numpy as np
from joblib import Parallel, delayed

from tidalsim.bb.common import BasicBlocks
from tidalsim.bb.spike import spike_trace_to_bbs, spike_trace_to_embedding_df
from tidalsim.bb.elf import objdump_to_bbs
from tidalsim.modeling.extrapolation import pick_intervals_for_rtl_sim
from tidalsim.util.cli import (
    run_cmd,
    run_cmd_pipe,
    run_rtl_sim_cmd,
    run_cmd_pipe_stdout,
)
from tidalsim.util.spike_ckpt import get_spike_cmd, gen_checkpoints
from tidalsim.util.spike_log import parse_spike_log
from tidalsim.util.pickle import load, dump
from tidalsim.modeling.schemas import EmbeddingSchema, ClusteringSchema
from tidalsim.cache_model.mtr import MTR, mtr_ckpts_from_inst_points
from tidalsim.cache_model.cache import CacheParams, CacheState


@dataclass
class RunArgs:
    binary: Path
    rundir: Path
    chipyard_root: Path
    rtl_simulator: Path
    n_threads: int
    n_harts: int = 1
    isa: str = "rv64gc"


@dataclass
class TidalsimArgs(RunArgs):
    interval_length: int = 10000
    clusters: int = 18
    warmup: bool = False
    points_per_cluster: int = 1
    bb_from_elf: bool = False


@dataclass
class GoldenSimArgs(RunArgs):
    sampling_period: int = 10000
    pass


def create_binary_rundir(rundir: Path, binary: Path) -> Path:
    binary_name = binary.name
    binary_dir = rundir / f"{binary_name}"
    binary_dir.mkdir(exist_ok=True)
    return binary_dir


def goldensim(args: GoldenSimArgs) -> None:
    binary_dir = create_binary_rundir(args.rundir, args.binary)
    golden_sim_dir = binary_dir / "golden"
    golden_sim_dir.mkdir(exist_ok=True)
    golden_perf_file = golden_sim_dir / "perf.csv"
    logging.info(f"Running full RTL simulation of {args.binary} in {golden_sim_dir}")

    if golden_perf_file.exists() and golden_perf_file.is_file():
        logging.info(
            f"Golden performance results already exist in {golden_sim_dir}, not-rerunning RTL"
            " simulation"
        )
    else:
        logging.info("Taking spike checkpoint at instruction 0 to inject into RTL simulation")
        gen_checkpoints(
            args.binary,
            start_pc=0x8000_0000,
            inst_points=[0],
            ckpt_base_dir=golden_sim_dir,
            n_threads=1,
            n_harts=args.n_harts,
            isa=args.isa,
        )
        inst_0_ckpt = golden_sim_dir / "0x80000000.0"
        rtl_sim_cmd = run_rtl_sim_cmd(
            simulator=args.rtl_simulator,
            perf_file=golden_perf_file,
            perf_sample_period=args.sampling_period,
            max_instructions=None,
            chipyard_root=args.chipyard_root,
            binary=(inst_0_ckpt / "mem.elf"),
            loadarch=(inst_0_ckpt / "loadarch"),
            suppress_exit=False,
            checkpoint_dir=None,
        )
        run_cmd(rtl_sim_cmd, cwd=golden_sim_dir)


def tidalsim(args: TidalsimArgs) -> None:
    logging.info(f"""Tidalsim called with:
    binary = {args.binary}
    interval_length = {args.interval_length}
    dest_dir = {args.rundir}""")

    binary_dir = create_binary_rundir(args.rundir, args.binary)

    # Create the spike commit log if it doesn't already exist
    spike_trace_file = (
        (binary_dir / "spike.full_trace") if args.warmup else (binary_dir / "spike.trace")
    )
    # Collect a full commit log from spike if we're doing functional warmup
    full_commit_log = args.warmup
    if spike_trace_file.exists():
        assert spike_trace_file.is_file()
        logging.info(f"Spike trace file already exists in {spike_trace_file}, not rerunning spike")
    else:
        logging.info(f"Spike trace doesn't exist at {spike_trace_file}, running spike")
        spike_cmd = get_spike_cmd(
            args.binary,
            args.n_harts,
            args.isa,
            debug_file=None,
            inst_log=True,
            commit_log=full_commit_log,
            suppress_exit=False,
        )
        run_cmd_pipe(spike_cmd, cwd=args.rundir, stderr=spike_trace_file)

    # Extract basic blocks from either the spike commit log or binary ELF
    bb: BasicBlocks

    if args.bb_from_elf:
        # Construct basic blocks from elf if it doesn't already exist
        elf_bb_file = binary_dir / "elf_basicblocks.pickle"
        if elf_bb_file.exists():
            logging.info(f"ELF-based BB extraction already run, loading results from {elf_bb_file}")
            bb = load(elf_bb_file)
        else:
            logging.info("Running ELF-based BB extraction")
            # Check if objdump file exists, else run objdump
            objdump_file = binary_dir / f"{binary_dir.name}.objdump"
            if objdump_file.exists():
                logging.info(f"Using objdump file found at {objdump_file}")
            else:
                objdump_cmd = f"riscv64-unknown-elf-objdump -d {str(args.binary)}"
                logging.info(f"Running {objdump_cmd} to generate objdump from riscv binary")
                run_cmd_pipe_stdout(objdump_cmd, cwd=args.rundir, stdout=objdump_file)

            with objdump_file.open("r") as f:
                bb = objdump_to_bbs(f)
                dump(bb, elf_bb_file)
            logging.info(f"ELF-based BB extraction results saved to {elf_bb_file}")
    else:
        # Construct basic blocks from spike commit log if it doesn't already exist
        spike_bb_file = binary_dir / "spike_basicblocks.pickle"
        if spike_bb_file.exists():
            logging.info(
                "Spike commit log based BB extraction already run, loading results from"
                f" {spike_bb_file}"
            )
            bb = load(spike_bb_file)
        else:
            logging.info("Running spike commit log based BB extraction")
            with spike_trace_file.open("r") as f:
                spike_trace_log = parse_spike_log(f, full_commit_log)
                bb = spike_trace_to_bbs(spike_trace_log)
                dump(bb, spike_bb_file)
            logging.info(f"Spike commit log based BB extraction results saved to {spike_bb_file}")

    logging.debug(f"Basic blocks: {bb}")

    # Given an interval length, compute the BBV-based interval embedding
    if args.bb_from_elf:
        embedding_dir = binary_dir / f"n_{args.interval_length}_elf"
    else:
        embedding_dir = binary_dir / f"n_{args.interval_length}_spike"

    embedding_dir.mkdir(exist_ok=True)
    embedding_df_file = embedding_dir / "embedding_df.pickle"
    embedding_df: DataFrame[EmbeddingSchema]
    if embedding_df_file.exists():
        logging.info(f"BBV embedding dataframe exists in {embedding_df_file}, loading")
        embedding_df = load(embedding_df_file)
    else:
        logging.info("Computing BBV embedding dataframe")
        with spike_trace_file.open("r") as spike_trace:
            spike_trace_log = parse_spike_log(spike_trace, args.warmup)
            embedding_df = spike_trace_to_embedding_df(spike_trace_log, bb, args.interval_length)
            dump(embedding_df, embedding_df_file)
        logging.info(f"Saving BBV embedding dataframe to {embedding_df_file}")
    logging.info(f"BBV embedding dataframe:\n{embedding_df}")
    logging.info(f"BBV embedding # of features: {embedding_df['embedding'][0].size}")

    # Perform clustering and select centroids
    cluster_dir = embedding_dir / f"c_{args.clusters}"
    cluster_dir.mkdir(exist_ok=True)
    logging.info(f"Storing clustering for clusters = {args.clusters} in: {cluster_dir}")

    # TODO: standardize features and see if that makes a difference for clustering
    from sklearn.cluster import KMeans

    kmeans_file = cluster_dir / "kmeans_model.pickle"
    kmeans: KMeans
    if kmeans_file.exists():
        logging.info(f"Loading k-means model from {kmeans_file}")
        kmeans = load(kmeans_file)
    else:
        logging.info(f"Performing k-means clustering with {args.clusters} clusters")
        matrix = np.vstack(embedding_df["embedding"].to_numpy())  # type: ignore
        kmeans = KMeans(n_clusters=args.clusters, n_init="auto", verbose=100, random_state=100).fit(
            matrix
        )
        logging.info(f"Saving k-means model to {kmeans_file}")
        dump(kmeans, kmeans_file)

    # Augment the dataframe with the cluster label and distances
    clustering_df_file = cluster_dir / "clustering_df.pickle"
    clustering_df: DataFrame[ClusteringSchema]
    if clustering_df_file.exists():
        logging.info(f"Loading clustering DF from {clustering_df_file}")
        clustering_df = load(clustering_df_file)
    else:
        clustering_df = embedding_df.assign(
            cluster_id=kmeans.labels_,
            dist_to_centroid=lambda x: np.linalg.norm(np.vstack(embedding_df["embedding"].to_numpy()) - kmeans.cluster_centers_[x["cluster_id"]], axis=1),  # type: ignore
            chosen_for_rtl_sim=lambda x: [False for _ in range(len(x.index))],
            # chosen_for_rtl_sim=lambda x: x.groupby("cluster_id")["dist_to_centroid"].transform(
            #     lambda dists: dists == np.min(dists)
            # ),
        )
        pick_intervals_for_rtl_sim(
            clustering_df, 0
        )  # for now, only take the closest point from each centroid
        dump(clustering_df, clustering_df_file)
        logging.info(f"Saving clustering DF to {clustering_df_file}")

    logging.info(f"Clustering DF\n{clustering_df}")

    # Create the directories for each interval we want to simulate in RTL simulation
    intervals_to_simulate = clustering_df.loc[clustering_df.chosen_for_rtl_sim == True]
    intervals_start_points = sorted(intervals_to_simulate["inst_start"].tolist())
    checkpoint_dir = cluster_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoints = [checkpoint_dir / f"0x80000000.{i}" for i in intervals_start_points]
    for c in checkpoints:
        c.mkdir(exist_ok=True)
    logging.info(
        "The intervals starting at these dynamic instruction counts will be simulated"
        f" \n{', '.join([str(x) for x in intervals_start_points])}"
    )

    # Construct MTR checkpoints for the L1d cache
    mtr_ckpts: Optional[List[MTR]] = None
    if args.warmup:
        mtr_ckpts_exist = [(c / "mtr.pickle").exists() for c in checkpoints]
        if all(mtr_ckpts_exist):
            logging.info("MTR checkpoints already exist for each interval to simulate")
            mtr_ckpts = [load(c / "mtr.pickle") for c in checkpoints]
        else:
            logging.info(f"Generating MTR checkpoints at inst points {intervals_start_points}")
            with spike_trace_file.open("r") as f:
                spike_trace_log = parse_spike_log(f, full_commit_log)
                mtr_ckpts = mtr_ckpts_from_inst_points(
                    spike_trace_log, block_size=64, inst_points=intervals_start_points
                )
            for mtr_ckpt, ckpt_dir in zip(mtr_ckpts, checkpoints):
                dump(mtr_ckpt, ckpt_dir / "mtr.pickle")
                with (ckpt_dir / "mtr.pretty").open("w") as f:
                    pprint.pprint(mtr_ckpt, stream=f)

    # Capture arch checkpoints from spike
    # Cache this result if all the checkpoints are already available
    checkpoints_exist = [
        (c / "loadarch").exists() and (c / "mem.elf").exists() for c in checkpoints
    ]
    if all(checkpoints_exist):
        logging.info("Checkpoints already exist, not rerunning spike")
    else:
        logging.info("Generating arch checkpoints with spike")
        gen_checkpoints(
            args.binary,
            start_pc=0x8000_0000,
            inst_points=intervals_start_points,
            ckpt_base_dir=checkpoint_dir,
            n_threads=args.n_threads,
            n_harts=args.n_harts,
            isa=args.isa,
        )

    if args.warmup:
        assert mtr_ckpts
        cache_params = CacheParams(phys_addr_bits=32, block_size_bytes=64, n_sets=64, n_ways=4)
        for mtr, ckpt_dir in zip(mtr_ckpts, checkpoints):
            cache_state: CacheState
            with (ckpt_dir / "mem.0x80000000.bin").open("rb") as f:
                cache_state = mtr.as_cache(cache_params, f, dram_base=0x8000_0000)
            cache_state.dump_data_arrays(ckpt_dir, "dcache_data_array")
            cache_state.dump_tag_arrays(ckpt_dir, "dcache_tag_array")

    # Run each checkpoint in RTL sim and extract perf metrics
    perf_filename = "perf_warmup.csv" if args.warmup else "perf_cold.csv"
    perf_files_exist = all([(c / perf_filename).exists() for c in checkpoints])
    if perf_files_exist:
        logging.info(
            "Performance metrics for checkpoints already collected, skipping RTL simulation"
        )
    else:
        logging.info(
            "Running parallel RTL simulations to collect performance metrics for checkpoints"
        )

        def run_checkpoint_rtl_sim(checkpoint_dir: Path) -> None:
            rtl_sim_cmd = run_rtl_sim_cmd(
                simulator=args.rtl_simulator,
                perf_file=(checkpoint_dir / perf_filename),
                perf_sample_period=int(
                    args.interval_length / 10
                ),  # sample perf at least 10 times in an interval to give some granularity for detailed warmup
                max_instructions=args.interval_length,
                chipyard_root=args.chipyard_root,
                binary=(checkpoint_dir / "mem.elf"),
                loadarch=(checkpoint_dir / "loadarch"),
                suppress_exit=True,
                checkpoint_dir=(checkpoint_dir if args.warmup else None),
            )
            run_cmd(rtl_sim_cmd, cwd=checkpoint_dir)

        Parallel(n_jobs=args.n_threads)(
            delayed(run_checkpoint_rtl_sim)(checkpoint) for checkpoint in checkpoints
        )
