# Brynse Operations Runbook

This document defines the canonical operational workflow used to validate Brynse against real radio streams and preserved sessions.

Its purpose is to make live validation, replay, regression analysis, and release gates reproducible without relying on shell history or conversation context.

The commands in this document are intentionally split into two categories:

- **CANONICAL**: workflow and commands that should remain stable.
- **TUNE ME**: values that change between runs, such as session names, stream URLs, baselines, candidates, or target tracks.

Brynse development should remain evidence-first:

> Observer observes. Router understands. Specialist cuts.

The operational counterpart is:

> Capture first. Compare second. Listen before concluding.

---

## 1. Live radio run

A live validation run should preserve enough information to reproduce the session later.

### TUNE ME

```bash
run="live-example-$(date +%Y%m%d-%H%M%S)"
stream_url="https://example.invalid/stream"
```

Use a descriptive run name that reflects the feature, hypothesis, or checkpoint under validation.

### CANONICAL

```bash
mkdir -p "$run"

printf '%s\n' "$run" > current_start

.venv/bin/brynse-radio \
    "$stream_url" \
    --output "$run/tracks" \
    --codec mp3 \
    --routing-trace "$run/routing.jsonl" \
    --session-capture "$run/session" \
    2>&1 | tee "$run/live.log"

live_rc=${PIPESTATUS[0]}

echo "live_rc=$live_rc"
echo "run=$run"
```

A replayable live run should contain:

```text
<run>/
├── live.log
├── routing.jsonl
├── session/
└── tracks/
```

### What each artifact is for

- `live.log`: runtime evidence, metadata timing, failures, warnings, and operational trace.
- `routing.jsonl`: structured transition-routing evidence and final temporal decisions.
- `session/`: preserved source session used for deterministic replay.
- `tracks/`: generated product output used for manual listening and binary comparison.

`current_start` is a convenience pointer to the active run. It is operational state and should not be staged accidentally.

---

## 2. Stop and preserve

Stopping a live run must preserve the evidence gathered so far.

Do not remove or overwrite the run directory while investigating it.

The run should remain replayable after stopping.

### CANONICAL principles

1. Stop the live process cleanly.
2. Preserve `session/`, `routing.jsonl`, `live.log`, and `tracks/`.
3. Do not reuse the same run directory for a different implementation state.
4. Do not stage live-capture directories.
5. Do not delete `current_start` as part of unrelated source changes.
6. Record the Git commit used for the run when practical.

When a live run is part of an important validation checkpoint, record at minimum:

```text
branch
commit
Python version
ffmpeg version
stream URL
codec
run directory
```

---

## 3. Replay captured session

Replay is the canonical way to test a code change against previously captured real-world input.

A replay should write to a new candidate directory. Never overwrite the accepted baseline.

### TUNE ME

```bash
session="live-example/session"
candidate="live-example-replay-candidate"
```

### CANONICAL

```bash
.venv/bin/python -m brynse.integrations.radio.replay \
    "$session" \
    --output "$candidate/tracks" \
    --routing-trace "$candidate/routing.jsonl"

replay_rc=$?

echo "replay_rc=$replay_rc"
```

A successful replay should finish with metadata validation passing.

The preserved session is the input. The current code is the implementation under test.

---

## 4. Compare routing

Routing comparison is used to isolate semantic and temporal changes before listening to audio.

Compare an accepted baseline against the new candidate.

### TUNE ME

```bash
baseline="live-example-replay-baseline"
candidate="live-example-replay-candidate"
```

### CANONICAL

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

baseline = Path("BASELINE/routing.jsonl")
candidate = Path("CANDIDATE/routing.jsonl")

old = [
    json.loads(line)
    for line in baseline.read_text().splitlines()
    if line.strip()
]

new = [
    json.loads(line)
    for line in candidate.read_text().splitlines()
    if line.strip()
]

interesting = (
    "reference_time_seconds",
    "router_kind",
    "temporal_kind",
    "outgoing_end_seconds",
    "incoming_start_seconds",
    "semantic_outgoing_handoff_seconds",
    "semantic_incoming_handoff_seconds",
    "semantic_inserted_region_kind",
    "bounded_insert_ownership",
    "acoustic_island_count",
)

print(f"baseline_records={len(old)}")
print(f"candidate_records={len(new)}")

changed = 0

for index, (before, after) in enumerate(
    zip(old, new, strict=False),
    start=1,
):
    if before == after:
        print(f"RECORD {index}: SAME")
        continue

    changed += 1
    print(f"RECORD {index}: CHANGED")

    for key in interesting:
        if before.get(key) != after.get(key):
            print(f"  {key}:")
            print(f"    baseline={before.get(key)!r}")
            print(f"    candidate={after.get(key)!r}")

print(f"changed_records={changed}")
PY
```

Replace `BASELINE` and `CANDIDATE` with the intended directories before running the comparison.

### Interpretation

A changed routing record is not automatically a regression.

Classify the change first:

- ownership changed;
- semantic handoff changed;
- temporal edge changed;
- final PCM decision changed;
- inserted-region geometry changed;
- only descriptive evidence changed.

Then listen to the affected product output.

---

## 5. Compare generated tracks

Routing is engineering evidence. The generated audio is the product.

Binary comparison identifies exactly which tracks need manual listening.

### CANONICAL

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib

baseline = Path("BASELINE/tracks")
candidate = Path("CANDIDATE/tracks")

def digest(path):
    h = hashlib.sha256()

    with path.open("rb") as fh:
        for chunk in iter(
            lambda: fh.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()

old = {
    p.name: p
    for p in baseline.iterdir()
    if p.is_file()
}

new = {
    p.name: p
    for p in candidate.iterdir()
    if p.is_file()
}

changed = 0

for name in sorted(set(old) | set(new)):
    before = old.get(name)
    after = new.get(name)

    if before is None or after is None:
        changed += 1
        print(f"CHANGED {name}: missing on one side")
        continue

    before_size = before.stat().st_size
    after_size = after.stat().st_size

    same = (
        before_size == after_size
        and digest(before) == digest(after)
    )

    if same:
        print(f"SAME {name} {before_size}")
    else:
        changed += 1
        print(
            f"CHANGED {name}: "
            f"{before_size} -> {after_size} "
            f"delta={after_size - before_size:+d}"
        )

print(f"changed_tracks={changed}")
PY
```

### Listening rule

Listen to every changed track that may have been affected by the patch.

For transition work, record the relevant edge explicitly:

```text
Track name
start: ...
end: ...
```

Manual listening is first-class product evidence.

Tests protect invariants. Listening validates the actual product.

---

## 6. Run Phase gate

Before staging a source change, run the project gate appropriate to the phase.

At minimum:

```bash
git diff --check
```

Run focused tests for the modified subsystem first.

When the implementation has been validated with real audio, run the full test suite:

```bash
.venv/bin/pytest -q
```

Run Ruff over the changed files:

```bash
.venv/bin/ruff check \
    <changed source files> \
    <changed test files>
```

A full-project Ruff run may still be useful to detect new global issues:

```bash
.venv/bin/ruff check .
```

Pre-existing unrelated lint debt should not be cleaned incidentally as part of a focused transition fix.

Record all return codes.

For long gates, redirect output to `/tmp/*.txt` and inspect the resulting file.

---

## 7. Historical regression replay

A change that fixes the current live session should also be exercised against accepted historical captures.

This is a canonical regression gate for transition work.

### Workflow

```text
current implementation change
    ->
replay the live session that motivated the change
    ->
compare routing
    ->
compare generated tracks
    ->
listen to changed tracks
    ->
replay one or more accepted historical sessions
    ->
compare routing
    ->
compare generated tracks
    ->
listen only to historical collateral changes
    ->
full automated gate
```

### Purpose

Historical replay can expose combinations that focused unit tests do not cover.

A historical session is especially useful when it contains transitions previously accepted by manual listening.

Good regression sessions should therefore be treated as reusable product fixtures even if they are not part of the unit-test tree.

### Baseline selection

Prefer a baseline produced by the last accepted implementation state for that same captured session.

This isolates the effect of the current patch from older unrelated changes.

Do not compare only against an ancient original output if a newer accepted replay exists.

### Acceptance

A historical change is acceptable only when its effect is understood.

For changed tracks:

1. identify the routing/temporal cause;
2. listen to the affected edge;
3. classify it as better, equivalent, or regression;
4. do not hide unexplained collateral changes inside aggregate test success.

---

## 8. Pre-merge inspection

Before merging a development branch, establish a recoverable checkpoint.

### CANONICAL checks

```bash
git branch --show-current
git log -1 --oneline

git diff --name-only
git diff --cached --name-only

git status --short
```

Verify:

- the intended branch is checked out;
- the intended commits exist;
- tracked worktree is clean;
- staging is empty unless intentionally reviewing a staged commit;
- live captures remain untracked;
- `current_start` remains untracked;
- no unrelated files were included.

Use explicit staging:

```bash
git add \
    path/to/file1 \
    path/to/file2
```

Never use:

```bash
git add .
```

Review before committing:

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached
```

Patch and commit are separate steps.

---

## 9. Post-merge gate

After integration into the destination branch, verify the resulting tree rather than assuming the merge preserved behavior.

Recommended checks:

```bash
git branch --show-current
git log --oneline --decorate -n 10
git status --short
```

Then run:

```bash
.venv/bin/pytest -q
```

and focused lint for the files or subsystem affected by the merge.

For transition-engine changes, perform at least one replay of a known captured session after merge.

For large or architectural changes, prefer both:

- one known historical replay;
- one fresh live session.

---

## 10. Operational working rules

These rules describe the workflow used during Brynse transition development.

### Evidence first

Do not patch from intuition when a captured session can answer the question.

Preferred order:

```text
audit
-> evidence
-> hypothesis
-> focused patch
-> focused tests
-> replay
-> routing comparison
-> track comparison
-> listening
-> full gate
-> explicit staging
-> commit
```

### One bounded action at a time

Avoid combining unrelated operations into one large shell command.

In particular:

- do not patch and commit in the same step;
- do not start a new architectural front while validating the current one;
- do not clean unrelated lint debt during a focused fix.

### Preserve the terminal session

Operational shell blocks should not terminate the interactive shell.

Avoid adding `exit` to diagnostic or gate blocks.

Capture return codes instead.

### Preserve evidence

Do not delete live captures, routing traces, logs, or replay outputs during active investigation unless they are explicitly known to be redundant.

### Long output

Redirect substantial output to a numbered or descriptive file under `/tmp/`.

Example:

```bash
out=/tmp/brynse-example-gate.txt
: > "$out"

{
    # commands
} > "$out" 2>&1

run_rc=$?

echo "run_rc=$run_rc"
wc -l "$out"
cat "$out"
```

This keeps the terminal usable and makes review reproducible.

---

## 11. Transition architecture invariants

Operational debugging should respect Brynse's architectural ownership model.

### Core responsibility split

```text
Observer
    observes neutral descriptive evidence

Router
    identifies the transition phenomenon
    assigns specialist ownership
    provides semantic roles

Specialist
    resolves final temporal timestamps

Pipeline
    materializes the TemporalSplitDecision
```

Short form:

> Observer observes. Router understands. Specialist cuts.

### PCM authority

Geometry proposes candidate structure.

PCM verifies final temporal edges inside the domain owned by the specialist.

Short form:

> Geometries propose. PCM disposes.

Important refinement:

> PCM only disposes over an edge the specialist actually owns.

When a semantic boundary explicitly defines the edge of an inserted phenomenon, PCM may refine within the owned domain but must not cross that semantic boundary into the inserted material.

### Debugging classification

Use this distinction when investigating a bad track:

```text
wrong boundary evidence
    -> Observer / evidence problem

wrong phenomenon or ownership
    -> Router problem

correct ownership, wrong final timestamp
    -> Specialist / PCM problem

correct TemporalSplitDecision, wrong generated file
    -> Pipeline / materialization problem
```

Metadata is a temporal reference, not ground truth.

When metadata and acoustic evidence disagree, the acoustic signal determines the actual boundary.

---

## 12. Regression-session catalogue

Maintain a small curated catalogue of preserved sessions that exercise known transition topologies.

For each important session, record:

```text
session directory
accepted commit
important transitions
manual-listening notes
known topology
accepted replay directory
```

Example categories:

```text
direct fade-to-floor
outgoing-only fade-to-floor
region-backed acoustic island
reset-pair acoustic island
symmetric bounded insert
asymmetric bounded insert
long inserted region
hard cut
crossfade
metadata shift
```

The goal is not to preserve every live forever as a formal regression fixture.

The goal is to maintain a small, high-value corpus whose expected behavior has been verified by ear.

---

## 13. Documentation maintenance

Update this runbook when the operational workflow changes.

A command should become canonical only after it has been used successfully in real development.

When adding a new workflow:

1. validate it on a real session;
2. identify which parts are stable;
3. move stable mechanics into **CANONICAL**;
4. leave run-specific values under **TUNE ME**;
5. remove obsolete variants instead of accumulating multiple contradictory recipes.

The runbook should describe the current supported workflow, not the full history of how that workflow evolved.
