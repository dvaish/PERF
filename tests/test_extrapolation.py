from pathlib import Path
from pandera.typing import DataFrame

from tidalsim.modeling.extrapolation import parse_perf_file, get_checkpoint_insts, PerfMetrics
from tidalsim.modeling.schemas import ClusteringSchema


class TestExtrapolation:
    def test_parse_perf_file(self, tmp_path: Path) -> None:
        perf_file_csv = """cycles,instret
180,100
140,100
130,100
135,100"""
        perf_file = tmp_path / "perf.csv"
        with perf_file.open("w") as f:
            f.write(perf_file_csv)
        perf = parse_perf_file(perf_file, detailed_warmup_insts=0)
        expected_perf = PerfMetrics(ipc=400 / (180 + 140 + 130 + 135))
        assert perf == expected_perf

        perf = parse_perf_file(perf_file, detailed_warmup_insts=100)
        expected_perf = PerfMetrics(ipc=300 / (140 + 130 + 135))
        assert perf == expected_perf

        perf = parse_perf_file(
            perf_file, detailed_warmup_insts=150
        )  # should round up to the next interval after 150 insts
        expected_perf = PerfMetrics(ipc=200 / (130 + 135))
        assert perf == expected_perf

    def test_get_checkpoint_insts(self) -> None:
        clustering_df = DataFrame[ClusteringSchema]({
            "instret": [100, 100, 100, 100],
            "inst_count": [100, 200, 300, 400],
            "inst_start": [0, 100, 200, 300],
            "embedding": [[0, 1], [0, 1], [1, 2], [3, 4]],
            "cluster_id": [2, 2, 1, 0],
            "dist_to_centroid": [0.0, 0.0, 0.0, 0.0],
            "chosen_for_rtl_sim": [False, True, True, True],
        })
        insts = get_checkpoint_insts(clustering_df)
        assert insts == [300, 200, 100]
