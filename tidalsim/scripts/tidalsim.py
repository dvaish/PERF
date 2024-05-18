import argparse
from pathlib import Path
import logging


from tidalsim.modeling.run import TidalsimArgs, tidalsim


def main():
    logging.basicConfig(
        format="%(levelname)s - %(filename)s:%(lineno)d - %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(prog="tidalsim", description="Sampled simulation")
    parser.add_argument("--binary", type=str, required=True, help="RISC-V binary to run")
    parser.add_argument(
        "-n",
        "--interval-length",
        type=int,
        required=True,
        help="Length of a program interval in instructions",
    )
    parser.add_argument("-c", "--clusters", type=int, required=True, help="Number of clusters")
    # parser.add_argument('--n-harts', type=int, default=1, help='Number of harts [default 1]')
    n_harts = 1  # hardcode this for now
    # parser.add_argument('--isa', type=str, help='ISA to pass to spike [default rv64gc]', default='rv64gc')
    isa = "rv64gc"  # hardcode this for now
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Path to the RTL simulator binary with state injection support",
    )
    parser.add_argument(
        "--chipyard-root", type=str, required=True, help="Path to the base of Chipyard"
    )
    parser.add_argument(
        "--rundir", type=str, required=True, help="Directory in which checkpoints are dumped"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "-e", "--elf", action="store_true", help="Run ELF-based basic block extraction"
    )
    parser.add_argument(
        "--golden-sim",
        action="store_true",
        help="Run full RTL simulation of the binary and save performance metrics",
    )
    parser.add_argument(
        "--cache-warmup",
        action="store_true",
        help=(
            "Use functional warmup to initialize the L1d cache at the start of each RTL simulation"
        ),
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    tidalsim_args = TidalsimArgs(
        binary=Path(args.binary).resolve(),
        rundir=Path(args.rundir).resolve(),
        chipyard_root=Path(args.chipyard_root).resolve(),
        rtl_simulator=Path(args.simulator).resolve(),
        n_harts=n_harts,
        isa=isa,
        interval_length=args.interval_length,
        clusters=args.clusters,
        warmup=args.cache_warmup,
        points_per_cluster=1,
        bb_from_elf=args.elf,
    )
    assert tidalsim_args.rtl_simulator.exists() and tidalsim_args.rtl_simulator.is_file()
    assert tidalsim_args.chipyard_root.is_dir()
    assert tidalsim_args.binary.exists()
    assert tidalsim_args.interval_length > 1
    tidalsim_args.rundir.mkdir(parents=True, exist_ok=True)

    tidalsim(tidalsim_args)
