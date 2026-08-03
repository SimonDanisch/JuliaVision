#!/usr/bin/env python
"""Query the GPUMODE kernelbot submission corpus.

    tools/kernelbot.py problems                       # what's in there
    tools/kernelbot.py top matmul_v2 --gpu A100       # the leaderboard
    tools/kernelbot.py show 12345                     # print one submission's code
    tools/kernelbot.py dump matmul_v2 --gpu A100 -n 5 # write top-N to files
    tools/kernelbot.py search 'cp\\.async\\.bulk'      # regex over winning kernels
    tools/kernelbot.py techniques --problem trimul    # which tricks the fast ones use
    tools/kernelbot.py progression trimul --user X    # one author's attempts + deltas
    tools/kernelbot.py sql "select ..."               # anything else

Scores are wall-clock seconds; lower is better.
"""

import argparse
import pathlib
import re
import sys

import duckdb

DB = pathlib.Path(__file__).resolve().parent.parent / "reference" / "kernelbot-data" / "kernelbot.duckdb"
SCORE_FLOOR = 1e-7  # keep in sync with kernelbot_build.py


def connect(readonly=True):
    if not DB.exists():
        sys.exit(f"{DB} missing — run tools/kernelbot_build.py first")
    return duckdb.connect(str(DB), read_only=readonly)


def fmt_score(s):
    if s is None:
        return "-"
    for unit, scale in (("s", 1), ("ms", 1e-3), ("us", 1e-6), ("ns", 1e-9)):
        if s >= scale:
            return f"{s / scale:8.3f} {unit}"
    return f"{s:.3e} s"


def table(rows, headers):
    if not rows:
        print("(no rows)")
        return
    cells = [[("" if c is None else str(c)) for c in r] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in cells)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w[i] for i in range(len(headers))))
    for r in cells:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))


def cmd_problems(a, con):
    # Driven by the data, not by leaderboards.parquet: that file is missing 776
    # (cholesky) and lists the pmpp v1 boards whose submissions were never exported.
    # No join to best_kernels here: a submission scored on several GPUs has one
    # row per GPU, so submission_id is not unique and a join fans the count out.
    rows = con.sql(f"""
SELECT leaderboard_id, any_value(problem_name), string_agg(DISTINCT gpu, ',' ORDER BY gpu),
       count(*), count(DISTINCT user_id), min(score) FILTER (WHERE score >= {SCORE_FLOOR})
FROM kernels GROUP BY 1 ORDER BY 1
""").fetchall()
    table([(r[0], r[1], r[2], f"{r[3]:,}", r[4], fmt_score(r[5])) for r in rows],
          ["id", "problem", "gpu", "passing", "users", "best"])
    empty = con.sql("""SELECT string_agg(name, ', ' ORDER BY id) FROM leaderboards
                       WHERE id NOT IN (SELECT DISTINCT leaderboard_id FROM kernels)""").fetchone()[0]
    if empty:
        print(f"\nlisted but not exported (no submissions in the dump): {empty}")


def cmd_top(a, con):
    where, params = ["(problem_name = ? OR CAST(leaderboard_id AS VARCHAR) = ?)"], [a.problem, a.problem]
    if a.gpu:
        where.append("gpu = ?")
        params.append(a.gpu)
    rows = con.execute(f"""
SELECT rank, score, plausible, gpu, user_name, submission_id, code_bytes, file_name
FROM best_kernels WHERE {' AND '.join(where)}
ORDER BY NOT plausible, score LIMIT {a.n}
""", params).fetchall()
    table([(r[0], fmt_score(r[1]) + ("" if r[2] else " (!)"), r[3], r[4], r[5], f"{r[6]:,}", r[7]) for r in rows],
          ["#", "score", "gpu", "user", "submission_id", "bytes", "file"])
    if any(not r[2] for r in rows):
        print("\n(!) score below 100 ns — broken measurement, not a fast kernel")


def cmd_show(a, con):
    r = con.execute("SELECT code, problem_name, gpu, score, user_name, file_name FROM best_kernels WHERE submission_id = ?",
                    [a.submission_id]).fetchone()
    if r is None:
        r = con.execute("SELECT code, problem_name, gpu, score, user_name, file_name FROM kernels WHERE submission_id = ?",
                        [a.submission_id]).fetchone()
    if r is None:
        sys.exit(f"submission {a.submission_id} not found")
    print(f"# {r[1]} on {r[2]} — {fmt_score(r[3])} — {r[4]} — {r[5]}\n", file=sys.stderr)
    print(r[0])


def cmd_dump(a, con):
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    where, params = ["(problem_name = ? OR CAST(leaderboard_id AS VARCHAR) = ?)"], [a.problem, a.problem]
    if a.gpu:
        where.append("gpu = ?")
        params.append(a.gpu)
    rows = con.execute(f"""
SELECT rank, score, gpu, user_name, submission_id, file_name, code
FROM best_kernels WHERE {' AND '.join(where)} ORDER BY score LIMIT {a.n}
""", params).fetchall()
    for rank, score, gpu, user, sid, fname, code in rows:
        ext = pathlib.Path(fname or "kernel.py").suffix or ".py"
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(user or sid))
        p = out / f"{rank:02d}_{gpu}_{safe}_{sid}{ext}"
        p.write_text(f"# kernelbot {a.problem} / {gpu} rank {rank} — {fmt_score(score)} — {user} — submission {sid}\n{code}")
        print(f"{p}  {fmt_score(score)}")


def cmd_search(a, con):
    where, params = ["regexp_matches(code, ?)"], [a.pattern]
    if a.problem:
        where.append("problem_name = ?")
        params.append(a.problem)
    if a.gpu:
        where.append("gpu = ?")
        params.append(a.gpu)
    src = "best_kernels" if not a.all else "kernels"
    order = "score" if not a.all else "score"
    rows = con.execute(f"""
SELECT problem_name, gpu, score, user_name, submission_id
FROM {src} WHERE {' AND '.join(where)} ORDER BY {order} LIMIT {a.n}
""", params).fetchall()
    table([(r[0], r[1], fmt_score(r[2]), r[3], r[4]) for r in rows],
          ["problem", "gpu", "score", "user", "submission_id"])
    print(f"\n({len(rows)} shown; `show <submission_id>` to read one)")


def cmd_techniques(a, con):
    flags = [r[0] for r in con.sql("DESCRIBE code_features").fetchall()
             if r[0].startswith(("is_", "uses_"))]
    where, params = [], []
    if a.problem:
        where.append("problem_name = ?")
        params.append(a.problem)
    if a.gpu:
        where.append("gpu = ?")
        params.append(a.gpu)
    where.append(f"rank <= {a.top}")
    w = "WHERE " + " AND ".join(where)
    total = con.execute(f"SELECT count(*) FROM code_features {w}", params).fetchone()[0]
    if not total:
        sys.exit("no kernels match")
    sel = ", ".join(f"sum({f}::INT)" for f in flags)
    counts = con.execute(f"SELECT {sel} FROM code_features {w}", params).fetchone()
    rows = sorted(zip(flags, counts), key=lambda t: -t[1])
    scope = f" of {a.problem}" if a.problem else ""
    print(f"rank <= {a.top} per (problem, gpu){scope} — {total} kernels; keyword heuristics, not semantics\n")
    table([(f, c, f"{100 * c / total:5.1f}%", "#" * round(30 * c / total)) for f, c in rows if c],
          ["technique", "n", "share", ""])


def cmd_progression(a, con):
    """One author's successive attempts on a problem.  The score delta between
    consecutive submissions is the closest thing here to a labelled optimisation."""
    rows = con.execute("""
SELECT submission_time, score, gpu, submission_id, file_name, length(code)
FROM kernels
WHERE (problem_name = ? OR CAST(leaderboard_id AS VARCHAR) = ?) AND user_name = ?
ORDER BY submission_time
""", [a.problem, a.problem, a.user]).fetchall()
    if not rows:
        sys.exit(f"no passing submissions by {a.user!r} on {a.problem}")
    best, out = None, []
    for t, score, gpu, sid, fname, n in rows:
        delta = "" if best is None else f"{100 * (score - best) / best:+6.1f}%"
        best = min(best, score) if best is not None else score
        out.append((str(t)[:16], fmt_score(score), delta, gpu, sid, f"{n:,}", fname))
    table(out, ["time", "score", "vs best", "gpu", "submission_id", "bytes", "file"])
    print(f"\n{len(rows)} attempts, {fmt_score(min(r[1] for r in rows))} best "
          f"— `show <id>` two of them and diff to see what the gain was")


def cmd_sql(a, con):
    res = con.sql(a.query)
    table(res.fetchall(), [d for d in res.columns])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("problems", help="list leaderboards with counts").set_defaults(fn=cmd_problems)

    t = sub.add_parser("top", help="leaderboard for one problem")
    t.add_argument("problem"); t.add_argument("--gpu"); t.add_argument("-n", type=int, default=20)
    t.set_defaults(fn=cmd_top)

    s = sub.add_parser("show", help="print one submission's code to stdout")
    s.add_argument("submission_id", type=int); s.set_defaults(fn=cmd_show)

    d = sub.add_parser("dump", help="write the top-N kernels to files")
    d.add_argument("problem"); d.add_argument("--gpu"); d.add_argument("-n", type=int, default=5)
    d.add_argument("--out", default="reference/kernelbot-dumps"); d.set_defaults(fn=cmd_dump)

    g = sub.add_parser("search", help="regex over kernel source")
    g.add_argument("pattern"); g.add_argument("--problem"); g.add_argument("--gpu")
    g.add_argument("-n", type=int, default=30)
    g.add_argument("--all", action="store_true", help="search every passing run, not just per-user bests (slow)")
    g.set_defaults(fn=cmd_search)

    k = sub.add_parser("techniques", help="technique-flag histogram over the fastest kernels")
    k.add_argument("--problem"); k.add_argument("--gpu"); k.add_argument("--top", type=int, default=10)
    k.set_defaults(fn=cmd_techniques)

    r = sub.add_parser("progression", help="one author's attempts over time, with score deltas")
    r.add_argument("problem"); r.add_argument("--user", required=True); r.set_defaults(fn=cmd_progression)

    q = sub.add_parser("sql", help="run raw SQL")
    q.add_argument("query"); q.set_defaults(fn=cmd_sql)

    a = p.parse_args()
    with connect() as con:
        a.fn(a, con)


if __name__ == "__main__":
    main()
