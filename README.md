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
- Atomic track persistence

## Architecture

Brynse is organized around a source-agnostic segmentation core.

```text
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
BOUNDARY RESOLUTION
  ↓
SEGMENT PLAN
  ↓
MATERIALIZATION
  ↓
SEGMENT
  ↓
SINK
```

Domain-specific integrations provide semantics and boundary signals without
changing the core segmentation model.

The first-party radio integration depends on the generic engine:

```text
integrations.radio
      ↓
     core
```

The core never depends on radio concepts.

For a detailed description of the architecture, data flow, extension points,
retention model, boundary resolution, and radio integration, see
[`docs/architecture.md`](docs/architecture.md).

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

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
make gate
```

FFmpeg is required for integration tests that exercise decoding and finalization.

## License

Mozilla Public License 2.0. See `LICENSE`.
