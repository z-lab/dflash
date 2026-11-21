"""Small, deterministic workloads for comparing separately configured inference servers."""
import hashlib
import json
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from .benchmark import send_openai


def fenced(text, language):
    return f"```{language}\n{text}```"


def unwrap(text):
    if not text.startswith("```"):
        return text
    match = re.fullmatch(r"```[^\n]*\n(.*)^\x60{3}[ \t]*\n?", text, flags=re.S | re.M)
    return match.group(1) if match else text


# Generate the repetitive source, but never include a reference answer in the request.
SOURCE = "".join(f"feature_{i:03} = {i}\n" for i in range(96))
MODULES = {
    "policy.py": "RETRY_LIMIT = 2\nBACKOFF = 1\n",
    "defaults.ini": "[network]\nretry_limit = 2\nbackoff = 1\n",
}
CASES = {
    "single-file": "Change only feature_043 from 43 to 143. Return the complete file in a code fence, "
                   "preserving its final newline.\n" + fenced(SOURCE, "python"),
    "multi-file": "Change the retry limit from 2 to 4 in both files. Return one or more blocks as "
                  "filename, <<<<<<< SEARCH, original lines, =======, replacement lines, >>>>>>> REPLACE, "
                  "each marker on its own line.\n" + "\n".join(
                      f"{name}\n{fenced(body, '')}" for name, body in MODULES.items()
                  ),
    "new-code": "Write a single SQLite SELECT query for table readings(sensor TEXT, value INTEGER). "
                "Return sensor and sum(value) for each sensor with a positive total, ordered by total "
                "descending then sensor ascending. Return only the SQL query.",
}


def patch_matches(text):
    files = MODULES.copy()
    block = re.compile(r"([^\n]+)\n<<<<<<< SEARCH\n(.*?)^=======\n(.*?)^>>>>>>> REPLACE[ \t]*(?:\n|$)", re.S | re.M)
    rest = unwrap(text).strip("\n") + "\n"
    while rest:
        match = block.match(rest)
        if not match:
            return False
        name, before, after = match.groups()
        if name not in files or not before or files[name].count(before) != 1:
            return False
        files[name] = files[name].replace(before, after, 1)
        rest = rest[match.end():].lstrip("\n")
    return files == {"policy.py": MODULES["policy.py"].replace("LIMIT = 2", "LIMIT = 4"),
                     "defaults.ini": MODULES["defaults.ini"].replace("limit = 2", "limit = 4")}


def sql_matches(text):
    sql = unwrap(text).strip()
    def authorize(op, table, field, database, trigger):
        if (op == sqlite3.SQLITE_SELECT or
            (op == sqlite3.SQLITE_READ and table == "readings") or
            (op == sqlite3.SQLITE_FUNCTION and field in {"sum", "total", "coalesce", "ifnull"})):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    fixtures = [
        ([("b", 4), ("a", 5), ("a", -1), ("z", -1), ("x", 0), ("c", 9)],
         [("c", 9), ("a", 4), ("b", 4)]),
        ([("p", -2), ("q", 9), ("p", 12), ("q", -9)], [("p", 10)]),
        ([], []),
    ]
    for entries, expected in fixtures:
        with closing(sqlite3.connect(":memory:")) as db:
            db.execute("CREATE TABLE readings(sensor TEXT, value INTEGER)")
            db.executemany("INSERT INTO readings VALUES (?, ?)", entries)
            db.set_authorizer(authorize)
            db.set_progress_handler(lambda: 1, 50000)
            try:
                if db.execute(sql).fetchmany(len(expected) + 1) != expected:
                    return False
            except sqlite3.Error:
                return False
    return True


def grade(case, text):
    if case == "single-file":
        return unwrap(text) == SOURCE.replace("feature_043 = 43\n", "feature_043 = 143\n")
    if case == "multi-file":
        return patch_matches(text)
    if case == "new-code":
        return sql_matches(text)
    raise ValueError(f"Unknown edit-benchmark case {case!r}")


def run(args):
    def request(prompt, limit):
        return send_openai(args.base_url, [{"role": "user", "content": prompt}], model=args.model,
                           max_new_tokens=limit, temperature=0.0, top_p=1.0, top_k=0,
                           timeout_s=args.timeout_s, reasoning=args.reasoning)

    request("Reply with OK.", min(8, args.max_new_tokens))
    rows = []
    for rep in range(args.repeats):
        for case, prompt in CASES.items():
            start = time.perf_counter()
            response = request(prompt, args.max_new_tokens)
            wall_s = time.perf_counter() - start
            choice = response["choices"][0]
            text = choice["message"].get("content") or ""
            timing = response.get("timings") or {}
            meta = response.get("meta_info") or {}
            output_tokens = (response.get("usage") or {}).get("completion_tokens")
            rows.append(dict(case=case, repeat=rep + 1,
                             passed=choice["finish_reason"] == "stop" and grade(case, text),
                             finish_reason=choice["finish_reason"],
                             text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                             output_tokens=output_tokens, wall_s=wall_s,
                             request_tps=output_tokens / wall_s if output_tokens is not None and wall_s > 0 else None,
                             server_decode_ms=timing.get("predicted_ms"),
                             server_decode_tps=timing.get("predicted_per_second"),
                             proposed=timing.get("draft_n"), accepted=timing.get("draft_n_accepted"),
                             mean_accepted_length=meta.get("spec_accept_length")))
    report = {"label": args.label, "model": args.model, "repeats": args.repeats,
              "max_new_tokens": args.max_new_tokens, "reasoning": args.reasoning, "requests": rows}
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
