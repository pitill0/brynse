# Brynse

Brynse is a boundary-driven continuous-stream segmentation engine.

It incrementally consumes encoded streams, retains only the working data required
for segmentation, resolves pluggable boundary signals, and materializes bounded
segments without requiring source-specific semantics.

Audio is its current media domain. Radio ripping is provided as a first-party
integration built on top of the generic engine.

The engine originated inside FluxTuner's radio ripping workflow, but has since
evolved into an independent boundary-driven streaming segmentation core.
FluxTuner remains a first-party integration and reference consumer.

## Current scope

### Generic segmentation engine

- MP3 and AAC/ADTS frame timelines
- Bounded in-memory analysis ring plus reclaimable disk spool retention
- Bounded acoustic-window extraction
- FFmpeg decode for analysis
- RMS acoustic profiling
- Pluggable fixed, manual, and external boundary providers
- Boundary candidate matching and split policy
- Crossfade-aware byte-range planning
- Streaming MP3 finalization
- Streaming AAC/ADTS to M4A finalization
- Incremental source-agnostic CLI segmentation

### First-party radio integration

- Incremental ICY metadata parsing
- Metadata semantic tracking
- Radio-specific track state and transition handling
- Multi-signal acoustic candidate selection
- Neutral transition evidence and routing
- Dedicated hard-cut, fade, interstitial, and bounded-insert specialists
- Conservative ambiguous-transition fallback
- Atomic track persistence

## Installation and build

Brynse requires Python 3.11 or newer.

The project uses `setuptools` and can be installed directly from a source
checkout.

A basic local installation is:

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -U pip
    python -m pip install .

This installs the Brynse package and exposes both command-line entry points:

    brynse
    brynse-radio

You can verify the installation with:

    brynse --help
    brynse-radio --help

FFmpeg is required for workflows that perform acoustic decoding or media
finalization, including the first-party radio ripping path.

For development, install the optional development dependencies with:

    python -m pip install -e '.[dev]'

The repository also provides:

    make test

for the test suite, and:

    make gate

for the complete development validation gate.

To build distributable package artifacts from the checkout, use the standard
Python build frontend after installing it:

    python -m pip install build
    python -m build

The resulting source and wheel artifacts are written under:

    dist/

## Architecture

Brynse is organized around a source-agnostic segmentation core.

    SOURCE
      ↓
    INGESTION
      ↓
    RETENTION
      ↓
    BOUNDARY PROVIDERS
      ↓
    BOUNDARY CANDIDATES
      ↓
    BOUNDARY EVIDENCE / RESOLUTION
      ↓
    SEGMENT PLAN
      ↓
    MATERIALIZATION
      ↓
    SEGMENT
      ↓
    SINK

Domain-specific integrations provide semantics and boundary signals without
changing the core segmentation model.

The first-party radio integration adds a routed transition-resolution layer:

    semantic radio candidate
            ↓
    multi-signal acoustic selection
            ↓
    neutral transition observations
            ↓
    Transition Router
            ↓
    transition specialist
            ↓
    TemporalSplitDecision
            ↓
    frame alignment / materialization

The responsibilities are intentionally separated:

- observers describe available evidence
- the Router decides which transition phenomenon owns the observation
- the selected specialist owns temporal geometry
- downstream code applies and aligns the resulting temporal split

The current routed transition families are:

- `HARD_CUT`
- `FADE`
- `INTERSTITIAL`
- `BOUNDED_INSERT`
- `AMBIGUOUS`

`AMBIGUOUS` is a conservative compatibility fallback rather than a general
acoustic specialist.

The first-party radio integration depends on the generic engine:

    integrations.radio
          ↓
         core

The core never depends on radio concepts.

For a detailed description of the architecture, data flow, transition routing,
specialists, retention model, boundary resolution, and radio integration, see
[`docs/architecture.md`](docs/architecture.md).

## Radio ripping example

Brynse includes a first-party radio integration for splitting a continuous ICY
radio stream into finalized tracks.

A basic MP3 ripping session is:

    brynse-radio \
      "https://example.com/radio-stream" \
      --output ./tracks \
      --codec mp3

During the session Brynse consumes the stream incrementally, follows ICY metadata,
settles transitions using bounded acoustic evidence, routes each transition to the
appropriate specialist, and writes completed tracks as they become available.

Typical output includes the detected metadata and finalized files:

    codec: mp3
    output: ./tracks
    ripping stream; press Ctrl+C to stop
    [   267.598s] Artist - Track
    written: tracks/Artist - Previous Track.mp3

### Recording filenames and collision safety

Brynse preserves readable Unicode metadata in recording filenames, while
replacing unsafe path characters and control characters. Long titles are
limited by their UTF-8 byte length, accounting for the file extension and
any collision suffix.

Completed recordings are published without overwriting existing files.
Repeated titles receive unique names such as `Artist - Track.mp3`,
`Artist - Track-2.mp3`, and `Artist - Track-3.mp3`. This also applies to
AAC recordings finalized as `.m4a` and to generic segment output.

Publication occurs only after successful finalization. Existing recordings
are never overwritten. If publication fails, completed audio may remain in
private recovery/staging storage; do not delete these artifacts blindly. An
uncertain directory `fsync` can mean that a completed track is already visible
while its crash durability is not confirmed. See the recovery guide below.

Codec detection can normally be left automatic by omitting `--codec`.

See [`docs/examples/live-radio-track-ripping.md`](docs/examples/live-radio-track-ripping.md)
for the complete workflow, behavior, and validation notes.

## Recording storage requirements and recovery

**Platform:** recording publication currently requires **Linux** with a mounted,
accessible `/proc/self/fd`. This requirement applies to both `brynse-radio`
and `brynse` when writing files. Linux kernels older than 6.10 are supported
without elevated privileges: Brynse links its already-open file descriptor via
`/proc/self/fd`, not privileged `AT_EMPTY_PATH`. Environments that restrict
hardlinks, descriptor links, or directory writes are rejected during preflight.
Do not assume recording publication works on other operating systems.

**Filesystem:** the destination must support hardlinks, `fsync` on files and
directories, and owner-only `.brynse-recovery/` (mode `0700`, owned by the
running user, on the same filesystem as output). The output directory itself
may be a symlink; the session checks the resolved target identity before
reading and when publishing. A destination change or disappearing mount stops the session
rather than silently redirecting recording output. Permissions and free space
must remain valid for the duration of the session; preflight cannot guarantee
that a NAS, mount, or ACL will not change later.

**Completed file permissions:** on Linux, Brynse derives the conventional
`0666 & ~umask` permission bits from the `Umask:` entry in `/proc/self/status`.
If the umask cannot be read, it fails closed to mode `0600`. Private recovery
storage remains mode `0700` regardless of the umask.

**Recovery procedure:** when recording or final publication fails, first read
Brynse's error and warning messages for the exact output, private recovery, or
retained spool paths. Stop any writers using the same destination. A completed
track may already exist even after a durability-related error; **never blindly
retry or overwrite it**. Inspect `.brynse-recovery/` and any reported staging
files before deciding whether to restore or remove anything. Brynse checks
known published hardlinks during startup reconciliation and only cleans a
recovery link after verifying the matching published inode. Unknown entries,
quarantined files, incomplete spool chunks, and ambiguous links are retained
for manual assessment: copy to a safe location before experimenting. Startup
reconciliation is not a general-purpose audio repair facility. Never delete
`.brynse-recovery/` wholesale to reclaim space.

**Operational notes:** Avoid switching between root and non-root recording
sessions for a shared output: the private recovery directory must be owned
by the current user (or startup fails safely). Network-mounted directories
may block during identity checks if the NAS becomes unresponsive; no timeout
is guaranteed. Retained `.brynse-spool-*.chunk` artifacts are inventoried on
startup and **never deleted automatically**. Preserve the error-log ordering
of chunks for concatenation. New spool files have a session identifier and a
hexadecimal starting-byte offset in their names; sort by offset **within the
same session**, never across different sessions. Legacy random-named chunks
must use the original error log for reliable ordering. Review retained data before manual removal.

**CLI behavior:** `brynse-radio` and the `brynse` generic streaming CLI
both perform output preflight and destination identity checks and retain
unfinished spool chunks when a stream or storage error occurs. The generic
CLI may return a CLI error after consuming stdin; it cannot replay consumed
stdin, so retained chunks and the reported recovery paths are essential.
The in-memory JSON-only mode (without an output directory) has no durable spool.

**Durability limits:** a retained spool is synchronized before reported
recovery where possible, but abrupt power loss, filesystem failure, or sudden
storage removal can still lose the segment being recorded or leave durability
uncertain. `fsync` confirmation is not a guarantee against physical media
failure. A recording remaining readable from a second process does not by
itself prove survival across a power cut. Preserve logs and recovery files for
forensic inspection. Recovery artifacts can consume additional disk space after
an interrupted cleanup, even when a published track exists.

## Generic streaming CLI

`brynse` reads MP3 or AAC input incrementally from a file or
stdin. Without `--output-dir` it reports resolved boundaries as JSON. With
`--output-dir`, encoded input is retained in a reclaimable disk spool while a
bounded ring remains available for boundary analysis; completed segments are
materialized incrementally from the retained source.

Fixed-interval segmentation:

    brynse input.mp3 \
      --codec mp3 \
      --provider fixed \
      --interval 180 \
      --output-dir segments

Manual boundaries can be supplied by repeating `--boundary`:

    brynse input.aac \
      --codec aac \
      --provider manual \
      --boundary 120 \
      --boundary 245.5 \
      --output-dir segments

External boundaries are loaded from JSON or JSONL:

    brynse - \
      --codec mp3 \
      --provider external \
      --boundaries-file boundaries.jsonl \
      --output-dir segments

Use `--min-tail SECONDS` to merge a final tail shorter than the configured
threshold into the preceding segment. The default is 1 second.

## Machine-oriented API

Brynse also exposes a Python API for embedding the segmentation engine
in applications, automation, or agent-oriented workflows. This is a library API;
it is not an additional installed CLI.

The structured request contract is `SegmentRequest`:

```python
from brynse.machine import SegmentRequest

request = SegmentRequest(
    codec="mp3",
    provider="fixed",
    interval_seconds=10.0,
)
```

The same request can cross a JSON boundary using the equivalent object:

```json
{
  "codec": "mp3",
  "provider": "fixed",
  "interval_seconds": 10.0
}
```

The current machine request contract exposes `codec`, `provider`, and
`interval_seconds`. The generic CLI has a wider surface, including manual and
external boundary inputs and output materialization options.

Three entry points provide progressively safer boundaries:

- `segment_source(source=..., request=...)` is the strict Python API. It consumes
  a `StreamSource`, returns a `SegmentResult`, and may propagate exceptions.
- `run_segment_source(source=..., request=...)` is the safe Python API. It returns
  either `SegmentResult` or `MachineError`.
- `run_segment_json(source=..., request_json=...)` is the machine-oriented JSON
  boundary. It parses the request, executes through the safe API, and returns a
  JSON string containing either the result or a structured error.

A successful result contains the normalized request information together with
ingestion and segmentation data:

```json
{
  "boundaries": [],
  "bytes_ingested": 12,
  "codec": "mp3",
  "interval_seconds": 10.0,
  "provider": "fixed_interval",
  "segments": []
}
```

`MachineError` uses three fields: `code`, `message`, and `kind`. The currently
defined error codes are:

- `invalid_request` for validation errors
- `io_error` for input/output failures
- `finalize_error` for segment finalization failures
- `write_error` for persistence failures
- `internal_error` for otherwise unclassified internal failures

The error kinds are `validation`, `operational`, and `internal`.

`StreamSource` is the source abstraction used by the machine API. A source
provides transport metadata, incremental `read(max_bytes)` access, and `close()`.
During segmentation, encoded input is read incrementally and the source is closed
when consumption finishes or aborts.

Example:

```python
from brynse.machine import run_segment_json
from brynse.source import BinaryIOStreamSource

with open("input.mp3", "rb") as stream:
    source = BinaryIOStreamSource(stream)

    response = run_segment_json(
        source=source,
        request_json=('{"codec":"mp3","provider":"fixed","interval_seconds":10.0}'),
    )

print(response)
```

## Development

For an editable development checkout:

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -U pip
    python -m pip install -e '.[dev]'

Run the normal test suite with:

    make test

Run the complete repository gate with:

    make gate

FFmpeg is required for integration tests that exercise decoding and
finalization.

## License

Mozilla Public License 2.0. See `LICENSE`.
