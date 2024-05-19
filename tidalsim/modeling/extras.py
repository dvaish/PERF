import argparse
from pathlib import Path
import shutil
import stat
import sys
from joblib import Parallel, delayed
import logging
import pdb
import pprint
from pandera.typing import DataFrame
import numpy as np

from tidalsim.util.cli import (
    run_cmd,
    run_cmd_capture,
    run_cmd_pipe,
    run_cmd_pipe_stdout,
    run_rtl_sim_cmd,
)
from tidalsim.util.spike_ckpt import *
from tidalsim.util.spike_log import parse_spike_log
from tidalsim.bb.spike import spike_trace_to_bbs, spike_trace_to_embedding_df, BasicBlocks
from tidalsim.bb.elf import objdump_to_bbs
from tidalsim.util.pickle import dump, load
from tidalsim.util.random import inst_points_to_inst_steps
from tidalsim.modeling.clustering import *
from tidalsim.modeling.schemas import *
from tidalsim.cache_model.mtr import mtr_ckpts_from_inst_points, MTR


def asdf(args):

    if args.cache_warmup:
        assert mtr_ckpts
        cache_params = CacheParams(phys_addr_bits=32, block_size_bytes=64, n_sets=64, n_ways=4)
        for mtr, ckpt_dir in zip(mtr_ckpts, checkpoints):
            cache_state: CacheState
            with (ckpt_dir / "mem.0x80000000.bin").open("rb") as f:
                cache_state = mtr.as_cache(cache_params, f, dram_base=0x8000_0000)
            cache_state.dump_data_arrays(ckpt_dir, "dcache_data_array")
            cache_state.dump_tag_arrays(ckpt_dir, "dcache_tag_array")

    # Run each checkpoint in RTL sim and extract perf metrics
    perf_filename = "perf_warmup.csv" if args.cache_warmup else "perf_cold.csv"
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
                simulator=simulator,
                perf_file=(checkpoint_dir / perf_filename),
                perf_sample_period=int(args.interval_length / 10),
                max_instructions=args.interval_length,
                chipyard_root=chipyard_root,
                binary=(checkpoint_dir / "mem.elf"),
                loadarch=(checkpoint_dir / "loadarch"),
                suppress_exit=True,
                checkpoint_dir=(checkpoint_dir if args.cache_warmup else None),
            )
            run_cmd(rtl_sim_cmd, cwd=checkpoint_dir)

        Parallel(n_jobs=-1)(
            delayed(run_checkpoint_rtl_sim)(checkpoint) for checkpoint in checkpoints
        )
