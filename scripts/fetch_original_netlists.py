"""Fetch the public source netlists used by the DeepTPI benchmark families.

The script downloads only the eight files needed by this project.  ITC'99
sources are the full-scan combinational ``*_C.bench`` netlists.  EPFL sources
are the original BLIF releases because BLIF retains internal net names.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


SOURCES = {
    "b15_C": "https://raw.githubusercontent.com/squillero/itc99-poli/master/i99t/b15/b15_C.bench",
    "b17_C": "https://raw.githubusercontent.com/squillero/itc99-poli/master/i99t/b17/b17_C.bench",
    "b20_C": "https://raw.githubusercontent.com/squillero/itc99-poli/master/i99t/b20/b20_C.bench",
    "b21_C": "https://raw.githubusercontent.com/squillero/itc99-poli/master/i99t/b21/b21_C.bench",
    "b22_C": "https://raw.githubusercontent.com/squillero/itc99-poli/master/i99t/b22/b22_C.bench",
    "i2c_aig": "https://raw.githubusercontent.com/lsils/benchmarks/master/random_control/i2c.blif",
    "max_aig": "https://raw.githubusercontent.com/lsils/benchmarks/master/arithmetic/max.blif",
    "mem_ctrl_aig": "https://raw.githubusercontent.com/lsils/benchmarks/master/random_control/mem_ctrl.blif",
    "i2c_epfl_aiger": "https://raw.githubusercontent.com/lsils/benchmarks/52b26f0e2cf1e88298a8b76c5e68e75013ba3977/random_control/i2c.aig",
    "max_epfl_aiger": "https://raw.githubusercontent.com/lsils/benchmarks/52b26f0e2cf1e88298a8b76c5e68e75013ba3977/arithmetic/max.aig",
    "mem_ctrl_epfl_aiger": "https://raw.githubusercontent.com/lsils/benchmarks/52b26f0e2cf1e88298a8b76c5e68e75013ba3977/random_control/mem_ctrl.aig",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(url: str, retries: int = 3, timeout: int = 120) -> bytes:
    request = Request(url, headers={"User-Agent": "TPI-my.3-netlist-recovery/1"})
    error: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (OSError, URLError) as exc:
            error = exc
            if attempt + 1 < retries:
                time.sleep(1 + attempt)
    raise RuntimeError(f"failed to download {url}: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--benchmarks", default=",".join(SOURCES))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    names = [item.strip() for item in args.benchmarks.split(",") if item.strip()]
    unknown = sorted(set(names) - set(SOURCES))
    if unknown:
        raise SystemExit(f"unknown benchmark(s): {','.join(unknown)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "1", "files": {}}
    for name in names:
        url = SOURCES[name]
        suffix = Path(url).suffix
        path = args.out_dir / f"{name}{suffix}"
        if path.exists() and not args.force:
            data = path.read_bytes()
            status = "existing"
        else:
            data = download(url)
            path.write_bytes(data)
            status = "downloaded"
        manifest["files"][name] = {
            "path": str(path),
            "url": url,
            "bytes": len(data),
            "sha256": sha256(data),
        }
        print(f"{name}\t{status}\tbytes={len(data)}\tsha256={sha256(data)}")

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
