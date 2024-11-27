import argparse
from pathlib import Path
import logging

from joblib import Parallel, delayed

from tidalsim.modeling.run import GoldenSimArgs, goldensim


def main():
    logging.basicConfig(
        format="%(levelname)s - %(filename)s:%(lineno)d - %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(prog="goldensim", description="Full RTL simulation")
    parser.add_argument(
        "--sampling-period",
        type=int,
        required=True,
        help="How often should the IPC be sampled?",
    )
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
        "binary",
        type=str,
        help="RISC-V binary or a directory containing RISC-V binaries to run",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    binary = Path(args.binary).resolve()
    if binary.is_file():
        goldensim_args = GoldenSimArgs(
            binary=Path(args.binary).resolve(),
            rundir=Path(args.rundir).resolve(),
            chipyard_root=Path(args.chipyard_root).resolve(),
            rtl_simulator=Path(args.simulator).resolve(),
            n_threads=1,
            n_harts=n_harts,
            isa=isa,
            sampling_period=args.sampling_period,
        )
        assert goldensim_args.chipyard_root.is_dir()
        assert goldensim_args.binary.exists()
        goldensim_args.rundir.mkdir(parents=True, exist_ok=True)

        goldensim(goldensim_args)
    else:
        assert binary.is_dir()
        goldensim_args = [
            GoldenSimArgs(
                binary=binary,
                rundir=Path(args.rundir).resolve(),
                chipyard_root=Path(args.chipyard_root).resolve(),
                rtl_simulator=Path(args.simulator).resolve(),
                n_threads=1,
                n_harts=n_harts,
                isa=isa,
                sampling_period=args.sampling_period,
            )
            for binary in binary.glob("*")
            if binary.is_file()
        ]
        assert goldensim_args[0].chipyard_root.is_dir()
        goldensim_args[0].rundir.mkdir(parents=True, exist_ok=True)

        Parallel(n_jobs=len(goldensim_args))(delayed(goldensim)(args) for args in goldensim_args)
