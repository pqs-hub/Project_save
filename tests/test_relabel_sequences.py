import csv

import json

from scripts.relabel_sequences_with_backend import load_sequences, merge_labels, write_job_log


def test_load_sequences_limits_prefix_steps(tmp_path):
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark_id", "sequence_id", "step"])
        writer.writeheader()
        for sequence_id in ("a", "b"):
            writer.writerow({"benchmark_id": "subckt_0002", "sequence_id": sequence_id, "step": 0})
            for step in range(1, 6):
                writer.writerow({"benchmark_id": "subckt_0002", "sequence_id": sequence_id, "step": step})

    grouped = load_sequences(labels, max_sequences=1, max_steps=3)

    assert list(grouped) == ["subckt_0002::a"]
    assert [int(row["step"]) for row in grouped["subckt_0002::a"]] == [1, 2, 3]


def test_write_job_log_uses_safe_filename(tmp_path):
    write_job_log(tmp_path, "subckt_0002::trajectory/0", {"status": "ok"})

    paths = list((tmp_path / "logs").glob("*.json"))
    assert len(paths) == 1
    assert json.loads(paths[0].read_text()) == {"status": "ok"}


def test_merge_labels_can_drop_partial_sequence(tmp_path):
    fields = ["benchmark_id", "sequence_id", "step", "status"]
    for sequence_id, statuses in {"good": ["ok", "ok"], "bad": ["ok", "error"]}.items():
        path = tmp_path / "sequences" / "subckt_0002" / sequence_id / "labels.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for step, status in enumerate(statuses, 1):
                writer.writerow(
                    {"benchmark_id": "subckt_0002", "sequence_id": sequence_id, "step": step, "status": status}
                )

    merged = merge_labels(tmp_path, drop_partial_sequences=True)
    with merged.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["sequence_id"] for row in rows] == ["good", "good"]
