import argparse
from pathlib import Path
import logging


from tidalsim.modeling.run import GoldenSimArgs, goldensim


def main():
    logging.basicConfig(
        format="%(levelname)s - %(filename)s:%(lineno)d - %(message)s", level=logging.INFO
    )

    parser = argparse.ArgumentParser(prog="goldensim", description="Full RTL simulation")
    parser.add_argument("--binary", type=str, required=True, help="RISC-V binary to run")
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
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    goldensim_args = GoldenSimArgs(
        binary=Path(args.binary).resolve(),
        rundir=Path(args.rundir).resolve(),
        chipyard_root=Path(args.chipyard_root).resolve(),
        rtl_simulator=Path(args.simulator).resolve(),
        n_harts=n_harts,
        isa=isa,
    )
    assert goldensim_args.chipyard_root.is_dir()
    assert goldensim_args.binary.exists()
    assert goldensim_args.sampling_period > 1
    goldensim_args.rundir.mkdir(parents=True, exist_ok=True)

    goldensim(goldensim_args)
