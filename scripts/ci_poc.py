#!/usr/bin/env python3
"""Poll a GitHub Actions run; on failure, hand its log + artifacts to Claude
for a read-only diagnosis.

Usage:
    python scripts/ci_poc.py <run_id> [--workflow-repo owner/repo] \\
        [--gcp-project PROJECT_ID]

Auth: no login/mount needed in this environment. Set two env vars, sourced
from your own already-authenticated machine, before running:
    export GH_TOKEN="$(gh auth token)"                     # on your machine
    export CLOUDSDK_AUTH_ACCESS_TOKEN="$(gcloud auth print-access-token)"

`gh` reads GH_TOKEN natively; `gcloud` and ci_poc_firestore_get.py both read
CLOUDSDK_AUTH_ACCESS_TOKEN. The access token expires in about an hour --
re-export if a run takes longer than that. Both env vars are inherited by the
`claude -p` subprocess this script spawns, same as any other subprocess.
"""

# cspell:ignore CLOUDSDK firestore

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

POLL_INTERVAL_S = 15
MAX_WAIT_S = 30 * 60
LOG_CHAR_LIMIT = 20_000

REPO_ROOT = Path(__file__).resolve().parent.parent
FIRESTORE_HELPER = REPO_ROOT / "scripts" / "ci_poc_firestore_get.py"

# Tool/command surface Claude is allowed to use for the diagnosis call.
# Read-only by construction: no Write/Edit, no gcloud/gh mutating verbs, no
# git. `gcloud logging read` is a genuinely read-only subcommand (distinct
# from `gcloud logging write`), so it's safe to allow broadly. Firestore has
# no equivalent generic read-only CLI subcommand, so instead of opening up
# arbitrary Python execution, the only Firestore access allowed is the one
# purpose-built, read-only helper script below.
# Verified against an installed `claude` binary (2.1.247): --allowedTools and
# --permission-mode dontAsk (used in ask_claude below) are both real flags;
# anything not matching an allow pattern is denied automatically rather than
# prompting, so this can't hang waiting for input.
# Deliberately NOT included: `gh api` -- unlike `gh run view` (no write mode
# exists), `gh api` is a generic passthrough that accepts --method POST/PUT/
# DELETE. A live test confirmed `Bash(gh api:*)` lets Claude write (it
# created a real PR comment), so it's excluded even though nothing here
# currently asks Claude to use it. Add a purpose-built read-only wrapper
# (like ci_poc_firestore_get.py) if `gh api` reads are ever actually needed --
# never re-add the bare pattern.
READ_ONLY_ALLOWED_TOOLS = [
    "Read",
    "Grep",
    "Glob",
    "Bash(find:*)",
    "Bash(cat:*)",
    "Bash(gh run view:*)",
    "Bash(gcloud logging read:*)",
    f"Bash(uv run {FIRESTORE_HELPER}:*)",
]


def run_gh(*args: str) -> str:
    result = subprocess.run(
        ["gh", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout


def poll_run(run_id: str, repo: str | None) -> dict:
    repo_args = ["--repo", repo] if repo else []
    fields = "status,conclusion,name,headBranch,headSha,url"
    waited = 0
    while True:
        try:
            out = run_gh("run", "view", run_id, *repo_args, "--json", fields)
        except subprocess.CalledProcessError as e:
            sys.exit(
                f"gh failed (exit {e.returncode}): {e.stderr.strip()}\n"
                f"Check `gh auth status` / that GH_TOKEN is exported in this shell."
            )
        run = json.loads(out)
        if run["status"] == "completed":
            return run
        if waited >= MAX_WAIT_S:
            raise TimeoutError(f"Run {run_id} did not complete within {MAX_WAIT_S}s")
        print(
            f"  ...status={run['status']}, waiting {POLL_INTERVAL_S}s", file=sys.stderr
        )
        time.sleep(POLL_INTERVAL_S)
        waited += POLL_INTERVAL_S


def fetch_failure_log(run_id: str, repo: str | None) -> str:
    repo_args = ["--repo", repo] if repo else []
    try:
        return run_gh("run", "view", run_id, *repo_args, "--log-failed")
    except subprocess.CalledProcessError as e:
        return f"(failed to fetch failed-step log: {e.stderr})"


def download_artifacts(run_id: str, repo: str | None) -> Path:
    out_dir = REPO_ROOT / "ci_poc_runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_args = ["--repo", repo] if repo else []
    result = subprocess.run(
        ["gh", "run", "download", run_id, *repo_args, "-D", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        downloaded_anything = any(out_dir.rglob("*"))
        if downloaded_anything:
            print(
                f"  (gh run download reported an error, but some artifacts did "
                f"land -- likely one bad artifact among several, not a total "
                f"failure: {result.stderr.strip()})",
                file=sys.stderr,
            )
        else:
            print(
                f"  (no artifacts downloaded: {result.stderr.strip()})", file=sys.stderr
            )
    return out_dir


def listing(directory: Path) -> str:
    find = subprocess.run(
        ["find", str(directory), "-type", "f"], capture_output=True, text=True
    )
    return find.stdout.strip() or "(no files)"


def build_prompt(run: dict, log: str, artifact_dir: Path, gcp_project: str) -> str:
    log_excerpt = log
    if len(log) > LOG_CHAR_LIMIT:
        log_excerpt = (
            log[-LOG_CHAR_LIMIT:] + f"\n...(truncated to last {LOG_CHAR_LIMIT} chars)"
        )
    logging_command = (
        '`gcloud logging read \'resource.type="cloud_run_revision" AND '
        'resource.labels.service_name=~"^staging-neuracore-backend" AND '
        f'textPayload:"<RECORDING_ID>"\' --project={gcp_project} '
        "--freshness=1h --limit=200 --format=json`"
    )

    return f"""You are investigating a failed GitHub Actions CI run. This is a
read-only investigation: you must not create, update, delete, or modify anything
anywhere -- no file writes, no git commits/pushes, no `gh`/`gcloud` write or
mutating commands.

You have three sources of evidence, in addition to the repo itself (you're
running from inside its checkout, so its test/source code is readable too):

1. The failed step log and downloaded artifacts below (local files -- read them
   directly). The downloaded daemon's local SQLite DB (`state.db` in the
   artifacts, if present) has a `recordings` table whose `recording_id` column
   is the SAME id as the Firestore doc id below -- that's the bridge from
   "local artifact" to "cloud lookup".

2. Cloud Logging (project `{gcp_project}`, region us-central1): three Cloud Run
   services share one image with different `NC_SERVICE` roles --
   `staging-neuracore-backend-us-central1` (api: start/stop/register-traces),
   `...-stream-...` (SSE), `...-worker-...` (Cloud Tasks: save, sync,
   transcode -- a recording that stopped uploading but never got saved usually
   shows up HERE, not in the api logs). Logging is plain-text stdout, not
   structured -- there's no `jsonPayload`, only `textPayload`; filter on
   substrings of it, not on structured fields.
   Example:
   {logging_command}

3. Firestore (project `{gcp_project}`, database
   `staging-firestore-us-central1` -- NOT `(default)`, the helper below
   defaults to the right one already):
   `uv run {FIRESTORE_HELPER} <collection> <doc_id>` for a single document, or
   `uv run {FIRESTORE_HELPER} <collection> --where FIELD OP VALUE` to query.
   This is the only Firestore access available (no generic gcloud command
   exists for document reads), and it can only read, never write. Collections
   that matter here, all keyed by `recording_id` (UUID4) unless noted:
   - `pending_recordings/{{recording_id}}` -- exists from start until promoted,
     deleted on save. Fields: `end_time` (null until stopped),
     `expected_trace_count` (<=0 means traces were never registered), `status`
     (STARTED/UPLOADING/UPLOADED -- a display field, not the completion oracle).
   - `pending_recordings/{{recording_id}}/data_traces` -- per-trace `status`
     (QUEUED/UPLOAD_STARTED/UPLOAD_COMPLETE). This is the actual completion
     signal.
   - `recordings/{{recording_id}}` -- only exists after promotion; check
     `deleted` is false.
   - The real "is it done?" rule
     (`RecordingManager.is_recording_save_ready`): (a) `pending.end_time is not
     None`, (b) `pending.expected_trace_count > 0`, (c) `count(data_traces) ==
     expected_trace_count` AND `count(data_traces where status ==
     UPLOAD_COMPLETE) == expected_trace_count`. A recording can legitimately
     be mid-upload (no Firestore doc looks "wrong" yet) even while a local
     wait-helper has already timed out -- check trace-level progress before
     concluding backend failure.
   - A recording with no `pending_recordings` doc means either not-started or
     already-promoted -- check `recordings/{{recording_id}}` too before
     concluding anything from an absence.

Run: {run['name']} on branch {run['headBranch']} @ {run['headSha']}
Conclusion: {run['conclusion']}
URL: {run['url']}

Failed step log (may be truncated to the last {LOG_CHAR_LIMIT} characters):
---
{log_excerpt}
---

Downloaded artifacts are at: {artifact_dir}
Files present:
{listing(artifact_dir)}

Diagnose the root cause of this failure -- don't stop at what the traceback
suggests on its face; use the logs/Firestore evidence above to confirm or rule
that out. State your conclusion plainly, cite the specific evidence (log lines,
file paths, or query results) that supports it, propose a concrete fix, and flag
explicitly if you are not confident.
"""


def _describe_tool_use(name: str, tool_input: dict) -> str:
    if name == "Bash":
        description = tool_input.get("description", "")
        command = tool_input.get("command", "")
        return f"{description}\n       $ {command}"
    if name in ("Read", "Glob"):
        return str(tool_input.get("file_path") or tool_input.get("pattern", ""))
    if name == "Grep":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", ".")
        return f'"{pattern}" in {path}'
    return ", ".join(f"{k}={v!r}" for k, v in tool_input.items())


def _print_event_live(event: dict) -> None:
    """Human-readable rendering of one stream-json event to stderr, live."""
    if event.get("type") != "assistant":
        return
    for block in event.get("message", {}).get("content", []):
        block_type = block.get("type")
        if block_type == "thinking":
            text = (block.get("thinking") or "").strip()
            if text:
                print(f"\n[thinking] {text}", file=sys.stderr)
        elif block_type == "tool_use":
            name = block.get("name", "?")
            detail = _describe_tool_use(name, block.get("input") or {})
            print(f"\n[tool] {name}: {detail}", file=sys.stderr)
        elif block_type == "text":
            text = (block.get("text") or "").strip()
            if text:
                print(f"\n[claude] {text}", file=sys.stderr)


def ask_claude(
    prompt: str, timeout_s: float, max_budget_usd: float, log_path: Path
) -> tuple[str, str | None]:
    """Runs claude -p in streaming mode, writing every raw event to log_path as
    it arrives (tail -f it to watch live). Returns (result_text, session_id) --
    session_id lets you `claude --resume <id>` into the exact same session
    afterwards to ask it follow-up questions about what it investigated.
    """
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--allowedTools",
        *READ_ONLY_ALLOWED_TOOLS,
        "--max-budget-usd",
        str(max_budget_usd),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    timed_out = threading.Event()

    def _kill_on_timeout() -> None:
        timed_out.set()
        proc.kill()

    timer = threading.Timer(timeout_s, _kill_on_timeout)
    timer.start()

    result_text = None
    session_id = None
    try:
        with open(log_path, "w") as log_file:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                session_id = event.get("session_id", session_id)
                _print_event_live(event)
                if event.get("type") == "result":
                    result_text = event.get("result")
        proc.wait()
    finally:
        timer.cancel()

    if timed_out.is_set():
        return f"(claude did not finish within {timeout_s:.0f}s -- killed)", session_id
    if result_text is not None:
        return result_text, session_id
    stderr = proc.stderr.read() if proc.stderr else ""
    return f"(claude invocation failed, exit={proc.returncode}: {stderr})", session_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="GitHub Actions run ID to investigate")
    parser.add_argument(
        "--workflow-repo",
        dest="repo",
        default=None,
        help="owner/repo, if not run from inside the repo",
    )
    parser.add_argument(
        "--gcp-project",
        dest="gcp_project",
        default="neuracore-staging",
        help="GCP project for Cloud Logging/Firestore queries (default: staging)",
    )
    parser.add_argument(
        "--claude-timeout-s",
        dest="claude_timeout_s",
        type=float,
        default=600.0,
        help="Kill the diagnosis session if it runs longer than this (default: 300s)",
    )
    parser.add_argument(
        "--max-budget-usd",
        dest="max_budget_usd",
        type=float,
        default=2.0,
        help="Cap on API spend for the diagnosis session (default: $2)",
    )
    args = parser.parse_args()

    print(f"Polling run {args.run_id}...", file=sys.stderr)
    run = poll_run(args.run_id, args.repo)
    print(f"Run completed: conclusion={run['conclusion']}", file=sys.stderr)

    if run["conclusion"] != "failure":
        print(
            f"Run did not fail (conclusion={run['conclusion']}) - nothing to diagnose."
        )
        return

    print("Fetching failed-step log...", file=sys.stderr)
    log = fetch_failure_log(args.run_id, args.repo)

    print("Downloading artifacts...", file=sys.stderr)
    artifact_dir = download_artifacts(args.run_id, args.repo)

    prompt = build_prompt(run, log, artifact_dir, args.gcp_project)

    log_path = artifact_dir / "claude_stream.jsonl"
    print("Asking Claude to diagnose (read-only)...", file=sys.stderr)
    print(f"Live events streaming to: {log_path}", file=sys.stderr)
    print(f"  Watch it in another terminal: tail -f {log_path} | \\", file=sys.stderr)
    print(
        '    jq -c \'select(.type=="assistant") | .message.content[] '
        "| {type, text, tool: .name, input}'",
        file=sys.stderr,
    )
    diagnosis, session_id = ask_claude(
        prompt, args.claude_timeout_s, args.max_budget_usd, log_path
    )

    print("\n=== Diagnosis ===\n")
    print(diagnosis)

    if session_id:
        print("\n=== Continue this investigation ===\n")
        print(f"claude --resume {session_id}")
        print(
            "(opens the same session interactively, full history intact -- ask "
            'it follow-up questions like "what did the Firestore doc actually '
            'say?" or "walk me through the timeline". This is a normal '
            "interactive session, not the locked-down read-only mode the "
            "investigation ran under -- it'll prompt you before doing anything "
            "new, same as any Claude Code session.)"
        )


if __name__ == "__main__":
    main()
