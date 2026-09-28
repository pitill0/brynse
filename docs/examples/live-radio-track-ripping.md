# Radio stream to finalized tracks

## Status

| Evidence | Status |
| --- | --- |
| Implemented | Yes |
| Automated | Yes |
| Live boundary-analysis validation | Yes |
| Full live end-to-end track materialization | Not yet recorded |

This is the first complete domain-specific execution path built on top of the
generic segmentation engine.

The implementation consumes continuous ICY radio streams and is designed to
produce finalized track files while keeping radio semantics outside the generic
core.

The complete path is covered by deterministic repository tests at its
architectural boundaries.

Real radio streams have also been used during development to validate metadata,
timeline, boundary-analysis, and re-entry behavior. The full CLI-to-finalized-
track workflow has not yet been recorded as a separate live end-to-end
validation and is therefore not claimed as such here.

## Command

The first-party radio CLI is:

```text
brynse-radio
```

Basic usage:

```bash
brynse-radio \
  "https://example.com/radio-stream" \
  --output ./tracks
```

The URL must reference an HTTP or HTTPS stream that exposes ICY metadata through
`icy-metaint`.

The output directory is created when required.

## Codec detection

By default, codec selection is automatic:

```text
--codec auto
```

The current radio integration recognizes:

- `audio/mpeg` and `audio/mp3` as MP3
- `audio/aac` and `audio/aacp` as AAC

A codec can also be forced explicitly:

```bash
brynse-radio \
  "https://example.com/radio-stream" \
  --output ./tracks \
  --codec mp3
```

or:

```bash
brynse-radio \
  "https://example.com/radio-stream" \
  --output ./tracks \
  --codec aac
```

## ICY metadata requirement

The current radio pipeline requires the stream to provide `icy-metaint`.

The HTTP request explicitly asks for ICY metadata.

Metadata blocks are removed from the encoded audio stream and `StreamTitle`
events are mapped onto the encoded media timeline.

Those radio-specific events remain inside the radio integration.

They are eventually adapted into generic boundary candidates for the
segmentation core.

## Runtime pipeline

The complete runtime path is:

```text
HTTP/HTTPS radio stream
        ↓
StreamSource
        ↓
ICY parser
        ├──────────────→ timed metadata events
        ↓
clean encoded audio
        ↓
frame timeline
        ↓
bounded analysis ring
        +
reclaimable encoded spool
        ↓
radio semantic tracking
        ↓
TrackCandidate
        ↓
generic BoundaryCandidate adapter
        ↓
wait for bounded acoustic post-context
        ↓
MULTISIGNAL-RESOLVER-v2-MINIMAX
        ↓
Resolved Boundary
        ↓
segment planning
        ↓
materialization / finalization
        ↓
finalized track file
```

The radio integration owns concepts such as ICY metadata and tracks.

The generic engine owns boundaries, retention, segment planning, and
materialization.

## Metadata durability

Metadata changes are not immediately treated as durable track transitions.

The default durability threshold is:

```text
8 seconds
```

It can be changed with:

```text
--metadata-threshold SECONDS
```

For example:

```bash
brynse-radio \
  "https://example.com/radio-stream" \
  --output ./tracks \
  --metadata-threshold 10
```

This helps prevent short-lived metadata changes from being promoted directly
into durable track transitions.

## Acoustic boundary resolution

The radio integration resolves durable semantic metadata transitions against
nearby acoustic and timeline evidence.

The production radio path uses the frozen
`MULTISIGNAL-RESOLVER-v2-MINIMAX` resolver. The acoustic radius is fixed at:

```text
24 seconds
```

This radius is not currently exposed as a CLI option. The runner deliberately
defers resolution until the required post-transition acoustic context is available.
Metadata durability remains a separate setting controlled by
`--metadata-threshold`; acoustic settlement is not derived from that threshold.

Within the acoustic window, the production resolver:

- builds the candidate set from BASIN candidates plus direct-D2 local maxima;
- ranks candidate evidence using family RRF and forward novelty;
- selects hard-cut boundaries with the frozen MINIMAX ordering;
- uses MINIMAX for the incoming side of a crossfade and the latest qualifying rise
  for its outgoing side; and
- abstains when the resulting ordering is invalid.

MP3 and AAC use the same multisignal resolver. There is no MP3-only post-refinement
stage in the frozen production path.

This is an important architectural property: the radio integration supplies radio
semantics, while the resolver operates on generic candidate, acoustic, and timeline
evidence.

## Transient exclusion

Short metadata intervals explicitly identified as transient non-track content,
such as advertisements or jingles, can be excluded from normal track
transitions.

Transient exclusion is enabled by default.

It can be disabled with:

```text
--no-transient-exclusion
```

Example:

```bash
brynse-radio \
  "https://example.com/radio-stream" \
  --output ./tracks \
  --no-transient-exclusion
```

## Bounded retention in the real workflow

The radio runner uses two retention mechanisms:

```text
bounded in-memory ring
    → analysis and timeline-sensitive work

reclaimable encoded spool
    → source bytes still required for future track materialization
```

After a resolved transition has been written, the output writer exposes the
earliest encoded offset still needed by the next track.

The runner then discards spool data before that retained offset.

This means total source duration does not require total source data to remain
available indefinitely.

The radio use case therefore exercises the same bounded-retention architecture
documented for the generic engine.

## Output

When a track transition becomes resolved, the outgoing track is materialized and
written to the configured output directory.

The CLI reports completed files as they are written:

```text
written: /path/to/output/Artist - Track.ext
```

Track-oriented naming belongs to the radio integration.

The underlying materialization machinery operates on generic segment ranges.

## Stopping a live stream

A live stream can be stopped with `Ctrl+C`.

The runner requests a stop and closes the active source so a blocking stream
read can terminate.

If a current track is still incomplete when the run stops, the CLI reports it
explicitly:

```text
incomplete track not finalized: Artist - Track
```

The incomplete track is not presented as successfully finalized output.

## FFmpeg

FFmpeg is used by acoustic analysis and media finalization stages.

The default executable is:

```text
ffmpeg
```

A different binary can be supplied with:

```text
--ffmpeg /path/to/ffmpeg
```

## Automated evidence

The repository test suite covers the radio workflow at multiple boundaries.

The automated tests include:

- codec detection
- `icy-metaint` validation
- CLI argument handling
- mocked stream ingestion
- clean keyboard interruption
- broken output pipe handling
- stream-opening failures
- `StreamSource` wrapping
- direct runner consumption of `StreamSource`
- metadata semantic tracking
- transition creation
- multisignal acoustic candidate generation and ranking
- deferred resolution until required post-context is available
- shared MP3/AAC multisignal resolution behavior
- unresolved-boundary behavior
- transient exclusion
- track-range planning
- track finalization
- bounded spool retention
- spool reclamation after completed transitions
- failure behavior during append, write, discard, and construction
- safe resource cleanup

These tests use deterministic local or synthetic data rather than depending on
a public radio service.

## Live validation

Real radio streams have been used during development and boundary-validation
work.

The final frozen production resolver was also evaluated prospectively against
untouched Corpus E. Excluding three pre-declared session-level exclusions, the
resolver scored 15 transitions with MAE 0.562 s, median absolute error 0.200 s,
12/15 within 1.00 s, and maximum error 2.375 s. No tuning was performed from the
Corpus E results.

See [`../ripper-research-closeout-2026-09-28.md`](../ripper-research-closeout-2026-09-28.md)
for the frozen evaluation record and artifact hashes.

That work validated real-stream behavior around:

- continuous radio capture
- ICY-derived metadata and timeline observations
- boundary analysis
- identity return / re-entry analysis
- acoustic and temporal evidence around candidate transitions

It did not constitute a recorded validation of the complete CLI workflow through
finalized track files.

For that reason, this use case is currently classified as partially live
validated rather than fully live validated.

Public radio streams are external dependencies and may change or disappear, so
they should not become requirements of the normal repository gate.

## What this use case demonstrates

This workflow demonstrates more than radio ripping.

It proves that the generic engine can support a domain integration where:

1. a continuous external source is consumed incrementally;
2. domain-specific signals arrive alongside the media stream;
3. those signals are translated into generic boundary evidence;
4. generic resolution machinery determines usable segmentation boundaries;
5. only the required working source window is retained;
6. completed bounded segments are materialized before the source ends;
7. domain-specific output is produced without moving domain semantics into the
   core.

The architecture is therefore exercised by a real domain workflow rather than
existing only as a generic abstraction.
