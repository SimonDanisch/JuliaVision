#!/usr/bin/env python
"""Build the queryable DuckDB layer over the GPUMODE/kernelbot-data parquet dump.

The dump is 11 parquet files in two different schema families.  This script does
not copy the bulk data: `submissions` is a VIEW that unions the parquet files in
place (DuckDB pushes projections down, so filtering on score/leaderboard never
reads the `code` column).  Only the two small derived tables are materialised.

    python tools/kernelbot_build.py [--data DIR] [--rebuild]

Layout it creates in DIR:
    kernelbot.duckdb
      leaderboards      table  problem definitions (39 rows)
      submissions       view   every run of every submission, normalised
      kernels           view   passing leaderboard runs only, one row per submission
      best_kernels      table  each user's fastest passing kernel per (problem, gpu)
      code_features     table  technique flags per distinct kernel, no code column
      amd_successful    view   convenience: the pre-filtered AMD exports
      amd_dedup         view
"""

import argparse
import pathlib
import sys

import duckdb

# Files that partition the corpus by leaderboard_id.  successful_/deduplicated_
# are strict subsets of submissions.parquet and are deliberately NOT unioned.
FAMILY_A = [  # export schema: problem_name/user_name/mode/runner/score/passed/code
    "amd_1_1m_competition_submissions",
    "nvidia_nvfp4_submissions",
    "pmpp_v2_submissions",
    "trimul_submissions",
    "helion_b200_nebius_submissions",
    "linalg_submissions",
]
FAMILY_B = ["submissions"]  # AMD legacy schema: run_mode/run_score/run_system_info

SCORE_FLOOR = 1e-7  # seconds; below this the measurement is broken, not a fast kernel

# Technique flags.  Each entry is (column, SQL boolean over `code`).  Kept as one
# list so the CLI can print the vocabulary without duplicating it.
FEATURES = [
    # language / framework
    ("is_triton", "code LIKE '%import triton%' OR code LIKE '%@triton.jit%'"),
    ("is_cuda_cpp", "code LIKE '%__global__%'"),
    ("is_hip", "code LIKE '%hip_runtime%' OR code LIKE '%__builtin_amdgcn%'"),
    ("is_cute", "code LIKE '%cute::%' OR code LIKE '%cutlass%'"),
    ("is_helion", "code LIKE '%import helion%'"),
    ("is_inline_asm", "code LIKE '%asm volatile%' OR code LIKE '%asm(%'"),
    # memory movement
    ("uses_cp_async", "code LIKE '%cp.async%'"),
    # CuTe spells TMA a dozen ways (SM90_TMA_LOAD, tma_atom, make_tma_copy, ...)
    ("uses_tma", r"regexp_matches(code, '(?i)cp\.async\.bulk|tensormap|\btma[_.]|_tma\b|TMA_LOAD|TMA_STORE')"),
    ("uses_ldmatrix", "code LIKE '%ldmatrix%' OR code LIKE '%stmatrix%'"),
    ("uses_vec_load", "code LIKE '%float4%' OR code LIKE '%uint4%' OR code LIKE '%int4(%' OR code LIKE '%__half2%'"),
    ("uses_swizzle", "lower(code) LIKE '%swizzle%'"),
    ("uses_double_buffer", "lower(code) LIKE '%double_buffer%' OR lower(code) LIKE '%doublebuffer%' OR lower(code) LIKE '%pipelin%'"),
    ("uses_shared", "code LIKE '%__shared__%' OR code LIKE '%tl.zeros%' OR code LIKE '%LDS%'"),
    # matrix cores
    ("uses_mma", "code LIKE '%mma.sync%' OR code LIKE '%mma_sync%'"),
    ("uses_wgmma", "code LIKE '%wgmma%'"),
    ("uses_tcgen05", "code LIKE '%tcgen05%'"),
    ("uses_mfma", "code LIKE '%mfma%'"),
    ("uses_wmma", "code LIKE '%wmma%'"),
    ("uses_tl_dot", "code LIKE '%tl.dot%'"),
    # scheduling / decomposition
    ("uses_warp_spec", "lower(code) LIKE '%warp_special%' OR lower(code) LIKE '%producer%' OR lower(code) LIKE '%consumer%'"),
    ("uses_persistent", "lower(code) LIKE '%persistent%'"),
    ("uses_split_k", "lower(code) LIKE '%split_k%' OR lower(code) LIKE '%splitk%' OR lower(code) LIKE '%stream_k%' OR lower(code) LIKE '%streamk%'"),
    ("uses_num_stages", "code LIKE '%num_stages%'"),
    ("uses_shuffle", "code LIKE '%__shfl%' OR code LIKE '%shfl_xor%' OR code LIKE '%__reduce_%'"),
    ("uses_atomic", "lower(code) LIKE '%atomic%'"),
    ("uses_barrier_async", "code LIKE '%mbarrier%' OR code LIKE '%barrier.cluster%' OR code LIKE '%cuda::barrier%'"),
    ("uses_cluster", "code LIKE '%cluster_dim%' OR code LIKE '%clusterDim%' OR code LIKE '%__cluster%'"),
    ("uses_autotune", "code LIKE '%autotune%'"),
]


def normalise_a(name: str, path: pathlib.Path) -> str:
    return f"""
SELECT '{name}' AS source,
       submission_id, leaderboard_id, problem_name,
       CAST(user_id AS VARCHAR) AS user_id, user_name,
       code_id, file_name, submission_time,
       mode, runner AS gpu, status,
       score, passed, code
FROM read_parquet('{path}')
"""


def normalise_b(name: str, path: pathlib.Path) -> str:
    # AMD legacy dump: no problem_name/user_name columns, score lives in run_*,
    # and the GPU string is the raw torch device name.
    return f"""
SELECT '{name}' AS source,
       s.submission_id, s.leaderboard_id, l.name AS problem_name,
       CAST(s.user_id AS VARCHAR) AS user_id, NULL AS user_name,
       s.code_id, s.file_name, s.submission_time,
       s.run_mode AS mode,
       CASE WHEN s.run_system_info.gpu LIKE 'AMD Instinct MI300X%' THEN 'MI300'
            ELSE s.run_system_info.gpu END AS gpu,
       CASE WHEN s.run_passed THEN 'passed' ELSE 'failed' END AS status,
       s.run_score AS score, s.run_passed AS passed,
       decode(s.code) AS code  -- stored as postgres bytea in this export
FROM read_parquet('{path}') s
LEFT JOIN leaderboards l ON l.id = s.leaderboard_id
"""


def build(data: pathlib.Path, rebuild: bool) -> None:
    db = data / "kernelbot.duckdb"
    if rebuild and db.exists():
        db.unlink()
    con = duckdb.connect(str(db))

    present = {f.stem: f for f in data.glob("*.parquet")}
    missing = [n for n in FAMILY_A + FAMILY_B if n not in present]
    if missing:
        print(f"note: not downloaded, skipping: {', '.join(missing)}", file=sys.stderr)

    con.execute("DROP TABLE IF EXISTS leaderboards")
    con.execute(
        f"CREATE TABLE leaderboards AS SELECT * FROM read_parquet('{present['leaderboards']}')"
    )

    parts = [normalise_a(n, present[n]) for n in FAMILY_A if n in present]
    parts += [normalise_b(n, present[n]) for n in FAMILY_B if n in present]
    con.execute("CREATE OR REPLACE VIEW submissions AS " + "\nUNION ALL\n".join(parts))

    # One row per submission: its official leaderboard run, best score if the
    # same submission was scored more than once.  This is the table to query.
    con.execute("""
CREATE OR REPLACE VIEW kernels AS
SELECT source, leaderboard_id, problem_name, gpu,
       submission_id, user_id, any_value(user_name) AS user_name,
       code_id, any_value(file_name) AS file_name,
       min(submission_time) AS submission_time,
       min(score) AS score, any_value(code) AS code
FROM submissions
WHERE mode = 'leaderboard' AND passed AND score IS NOT NULL AND score > 0
GROUP BY source, leaderboard_id, problem_name, gpu, submission_id, user_id, code_id
""")

    for name, file in (("amd_successful", "successful_submissions"),
                       ("amd_dedup", "deduplicated_successful_submissions")):
        if file in present:
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{present[file]}')"
            )

    # Materialised: each user's single fastest kernel per (problem, gpu).  This
    # is what "learn from the winners" actually reads, and it carries the code.
    # A GPU kernel cannot run in under 100 ns; one eigh entry scores 4.6e-25 s.
    # Such rows are kept but flagged and ranked last, never silently dropped.
    print("materialising best_kernels ...", flush=True)
    con.execute("DROP TABLE IF EXISTS best_kernels")
    con.execute(f"""
CREATE TABLE best_kernels AS
WITH ranked AS (
  SELECT *, score >= {SCORE_FLOOR} AS plausible,
         row_number() OVER (PARTITION BY leaderboard_id, gpu, user_id
                            ORDER BY (score < {SCORE_FLOOR}), score) AS user_rn
  FROM kernels
)
SELECT problem_name, leaderboard_id, gpu,
       row_number() OVER (PARTITION BY leaderboard_id, gpu ORDER BY NOT plausible, score) AS rank,
       score, plausible, user_name, user_id, submission_id, code_id, file_name, submission_time,
       length(code) AS code_bytes, code
FROM ranked WHERE user_rn = 1
ORDER BY leaderboard_id, gpu, NOT plausible, score
""")
    con.execute("CREATE INDEX bk_problem ON best_kernels(problem_name)")
    con.execute("CREATE INDEX bk_rank ON best_kernels(leaderboard_id, gpu, rank)")

    print("materialising code_features ...", flush=True)
    flags = ",\n       ".join(f"({expr}) AS {col}" for col, expr in FEATURES)
    con.execute("DROP TABLE IF EXISTS code_features")
    con.execute(f"""
CREATE TABLE code_features AS
SELECT problem_name, leaderboard_id, gpu, rank, score, plausible, user_name, submission_id, code_id,
       code_bytes,
       length(code) - length(replace(code, chr(10), '')) + 1 AS loc,
       {flags}
FROM best_kernels
""")

    n_sub = con.sql("SELECT count(*) FROM submissions").fetchone()[0]
    n_ker = con.sql("SELECT count(*) FROM kernels").fetchone()[0]
    n_best = con.sql("SELECT count(*) FROM best_kernels").fetchone()[0]
    print(f"\n{db}")
    print(f"  submissions   {n_sub:>9,} rows (view over parquet)")
    print(f"  kernels       {n_ker:>9,} passing leaderboard runs")
    print(f"  best_kernels  {n_best:>9,} per-user bests, with code")
    con.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=pathlib.Path,
                   default=pathlib.Path(__file__).resolve().parent.parent / "reference" / "kernelbot-data")
    p.add_argument("--rebuild", action="store_true", help="delete the .duckdb file first")
    a = p.parse_args()
    build(a.data.resolve(), a.rebuild)
