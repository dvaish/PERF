from pathlib import Path
from pandera.typing import DataFrame

from tidalsim.modeling.extrapolation import (
    fill_perf_metrics,
    parse_perf_file,
    PerfMetrics,
    pick_intervals_for_rtl_sim,
)
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
        cycles = 180 + 140 + 130 + 135
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

    def test_fill_perf_metrics(self, tmp_path: Path) -> None:
        clustering_df = DataFrame[ClusteringSchema]({
            "instret": [100, 100, 100, 100],
            "inst_count": [100, 200, 300, 400],
            "inst_start": [0, 100, 200, 300],
            "embedding": [[0, 1], [0, 1], [1, 2], [3, 4]],
            "cluster_id": [2, 2, 1, 0],
            "dist_to_centroid": [0.0, 0.0, 0.0, 0.0],
            "chosen_for_rtl_sim": [False, True, True, True],
        })

        # Place some dummy perf data on disk
        perf_file_cold = """cycles,instret
70,50
100,50
        """
        perf_file_warm = """cycles,instret
60,50
80,50
        """
        cluster_dir = tmp_path / "checkpoints" / "0x80000000.100"
        cluster_dir.mkdir(parents=True)
        (cluster_dir / "perf_cold.csv").write_text(perf_file_cold)
        (cluster_dir / "perf_warmup.csv").write_text(perf_file_warm)

        perf_df = fill_perf_metrics(clustering_df, tmp_path, detailed_warmup_insts=0)
        row_with_data = perf_df.iloc[1]
        assert row_with_data.est_ipc_cold == (100 / 170)
        assert row_with_data.est_ipc_warm == (100 / 140)

    def test_pick_intervals_for_rtl_sim(self) -> None:
        clustering_df = DataFrame[ClusteringSchema]({
            "instret": [100, 100, 100, 100, 100],
            "inst_count": [100, 200, 300, 400, 500],
            "inst_start": [0, 100, 200, 300, 400],
            "embedding": [[0, 1], [0, 1], [], [1, 2], [3, 4]],
            "cluster_id": [2, 2, 2, 1, 0],
            "dist_to_centroid": [1.0, 1.0, 0.0, 0.0, 0.0],
            "chosen_for_rtl_sim": [False, False, False, False, False],
        })
        pick_intervals_for_rtl_sim(clustering_df, 1)
        assert clustering_df["chosen_for_rtl_sim"].to_list() == [True, False, True, True, True]
