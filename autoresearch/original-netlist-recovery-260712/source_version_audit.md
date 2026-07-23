# i2c / mem_ctrl source-version audit

Audit date: 2026-07-12

## Conclusion

Both DeepTPI circuits belong to the **EPFL Combinational Benchmark Suite (2015)
lineage**, but neither graph is the raw public EPFL AIG.  They are private,
preprocessed BENCH/AIG derivatives made before DeepTPI created
`benchmarks_circuits_graphs.npz` (commit `27a0242`, 2022-04-10).  The repository
publishes the renamed NPZ but not those two intermediate BENCH files or the
preprocessing command line, so there is no public Git commit that can be named
as the exact 136-PI/1028-PI file.

This is a preprocessing revision mismatch, not an EPFL Git branch mismatch.

## DeepTPI target fingerprints

After collapsing the fanout-split BUFF layer conceptually:

| circuit | PI | inferred PO/sinks | AND | NOT | split BUFF |
|---|---:|---:|---:|---:|---:|
| i2c_aig | 136 | 127 | 1067 | 735 | 1462 |
| mem_ctrl_aig | 1028 | 967 | 41802 | 31400 | 53123 |
| max_aig (control) | 512 | 129 | 2826 | 2734 | 3367 |

NPZ SHA-256:
`84b94b42d21d195bf6914712c06a680c3426624a62ecf616f99a5ce367bbd2b6`.

## Versions tried

| source candidate | i2c PI/PO/AND | mem_ctrl PI/PO/AND | result |
|---|---:|---:|---|
| EPFL Git introduction commit `52b26f0` (2018) | 147/142/1342 | 1204/1231/46836 | correct design family; not exact preprocessing |
| EPFL `master` | 147/142/1342 | 1204/1231/46836 | byte-identical to 2018 files |
| EPFL branches `new`, `new_2023`, `new_2024`, `update-2025-results` | 147/142/1342 | 1204/1231/46836 | available AIGs have the same hashes |
| DDD/CVUT `EPFL.7z` mirror | 147/142/1342 | 1204/1231/46836 | byte-identical EPFL mirror |
| IWLS 2005 OpenCores netlists | sequential source; full-scan forms differ | sequential source; full-scan forms differ | wrong netlist lineage/revision |
| OpenABC-D original BENCH | 177/128/1169 | 1187/962/18092 | wrong structural scale |

Public EPFL AIG hashes, now pinned in `sources/manifest.json`:

- i2c: `3f6ab413a3f745c87811fa6565d972146f1f4e6ede78fae71fcb16e469b9b213`
- mem_ctrl: `390d601836717129b903a5ecc8fd96b41d322d26895d666b303c64b796bd0d1d`

## ABC preprocessing experiments

The public EPFL AIGs were also processed with individual `balance`, `rewrite`,
`refactor`, `resub`, `dc2`, the DeepGate documented synthesis sequence, standard
`resyn2`, repeated `dc2`, and combined `resyn2; dc2` recipes.

Representative AND counts:

| recipe | i2c | mem_ctrl | max |
|---|---:|---:|---:|
| raw EPFL | 1342 | 46836 | 2865 |
| `dc2` | 1147 | 42797 | 2831 |
| `dc2; dc2` | 1076 | 41175 | 2821 |
| `resyn2; dc2` | 1076 | 41948 | 2829 |
| DeepTPI target | 1067 | 41802 | 2826 |

The close, correlated movement across all three circuits supports a shared ABC
optimization pipeline, but no tested public recipe reproduces all target counts.
ABC results are also version-sensitive.  Therefore matching only node counts is
not sufficient to claim an exact source.

## Recovery implication

The public EPFL BLIF/AIG is suitable as the functional design ancestor, but it
cannot yet provide a safe `Nxxx -> original net` mapping because DeepTPI removed
the PI names/order and the unpublished preprocessing changed the port set.
Recovery remains fail-closed until either:

1. the authors provide the 2022 intermediate `i2c_aig.bench` and
   `mem_ctrl_aig.bench`; or
2. the exact preprocessing executable/script is recovered and structural/formal
   equivalence establishes the PI/PO correspondence.
