import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_exact_itc99_vs_deeptpi.py"


def write_summary(path: Path, gaps: list[float], macro_tc: float, macro_gap: float) -> None:
    path.write_text(
        json.dumps(
            {
                "aggregate": {
                    "macro_filtered_final_tc_pct": macro_tc,
                    "macro_gap_vs_deeptpi_pp": macro_gap,
                },
                "per_circuit": [{"gap_vs_deeptpi_pp": gap} for gap in gaps],
            }
        )
    )


def test_score_penalizes_each_remaining_deficit(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    write_summary(summary, [-3.213, 1.518, 1.242, 1.025, -1.280], 93.8564, -0.1416)
    output = subprocess.check_output([sys.executable, str(SCRIPT), str(summary)], text=True)
    assert float(output) == -4.3991436


def test_check_requires_every_circuit_and_macro_to_win(tmp_path: Path) -> None:
    summary = tmp_path / "summary.json"
    write_summary(summary, [0.1] * 5, 94.1, 0.102)
    result = subprocess.run([sys.executable, str(SCRIPT), str(summary), "--check"], check=False)
    assert result.returncode == 0

    write_summary(summary, [0.1, -0.01, 0.1, 0.1, 0.1], 94.1, 0.102)
    result = subprocess.run([sys.executable, str(SCRIPT), str(summary), "--check"], check=False)
    assert result.returncode == 1
