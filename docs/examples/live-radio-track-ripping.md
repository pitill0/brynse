# Radio stream to finalized tracks

## Status

| Evidence | Status |
| --- | --- |
| Implemented | Yes |
| Automated | Yes |
| Live boundary-analysis validation | Yes |
| Full live end-to-end track materialization | Yes |

This is the first complete domain-specific execution path built on top of the
generic segmentation engine.

The implementation consumes continuous ICY radio streams and produces finalized
track files while keeping radio semantics outside the generic core.

The complete path is covered by deterministic repository tests at its
architectural boundaries.

Real external radio streams have also exercised the documented `brynse-radio`
workflow end to end, including finalized track materialization and manual
verification of generated output.

Live validation confirms the operational workflow. It does not imply perfect
perceptual boundary placement for every possible transition.

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
    multi-signal acoustic candidate selection
            ↓
    neutral transition evidence
            ↓
    Transition Router
            ↓
    transition specialist
            ↓
    TemporalSplitDecision
            ↓
    encoded-frame alignment
            ↓
    segment planning
            ↓
    materialization / finalization
            ↓
    finalized track file

The radio integration owns concepts such as ICY metadata and tracks.

The generic engine owns boundaries, acoustic and temporal evidence, transition
routing contracts, temporal split decisions, retention, segment planning, and
materialization.

The routed transition layer is source-neutral. It reasons from acoustic and
temporal evidence rather than radio-specific semantic labels.

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

## Acoustic evidence and routed transition resolution

The radio integration resolves durable semantic metadata transitions against
nearby acoustic and timeline evidence.

The production path retains the frozen multi-signal candidate-selection work
originating from `MULTISIGNAL-RESOLVER-v2-MINIMAX`.

The acoustic radius remains fixed at:

    24 seconds

This radius is not currently exposed as a CLI option. The runner deliberately
defers resolution until the required post-transition acoustic context is
available.

Metadata durability remains a separate setting controlled by
`--metadata-threshold`; acoustic settlement is not derived from that threshold.

Within the acoustic window, the multi-signal layer:

- builds candidate evidence from basin and structural observations;
- ranks acoustic candidates using the established multi-signal selection logic;
- selects primary acoustic evidence around the semantic transition; and
- provides evidence for subsequent transition analysis.

That selected acoustic candidate is not universally the final cut.

The production path then builds neutral transition observations from available
acoustic and temporal evidence, including:

- local discontinuity
- fade evidence
- basin geometry
- structural proposals
- incoming-boundary hypotheses
- symmetric and asymmetric bounded-insert regions

`DefaultTransitionRouter` assigns the observation to one of the current
transition families:

- `HARD_CUT`
- `FADE`
- `INTERSTITIAL`
- `BOUNDED_INSERT`
- `AMBIGUOUS`

The selected transition specialist then owns final temporal reasoning for that
phenomenon.

Current dedicated specialists include:

- `HardCutSpecialist`
- `FadeSpecialist`
- `InterstitialSpecialist`
- `BoundedInsertSpecialist`

`LegacyAmbiguousSpecialist` remains a conservative compatibility fallback when
the evidence does not justify stronger transition ownership.

The specialist returns a `TemporalSplitDecision`, whose outgoing and incoming
positions may be equal or different depending on the transition geometry.

The resulting temporal decision is then aligned to the encoded frame timeline
before segment planning and materialization.

MP3 and AAC use the same routed production architecture.

This separation is intentional:

    acoustic selection
        → describes useful candidate evidence

    Transition Router
        → decides which phenomenon owns the transition

    transition specialist
        → decides where that phenomenon should be cut

The radio integration supplies radio semantics, while transition routing and
specialist temporal reasoning operate on generic acoustic and temporal evidence.

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

The repository test suite covers the radio workflow at multiple architectural
boundaries.

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
- multi-signal acoustic candidate generation and ranking
- deferred resolution until required post-context is available
- shared MP3/AAC acoustic selection behavior
- local discontinuity evidence
- fade observation
- incoming-boundary discovery and refinement
- symmetric and asymmetric bounded-insert discovery
- transition routing
- hard-cut specialist behavior
- fade specialist behavior
- interstitial specialist behavior
- bounded-insert specialist behavior
- conservative ambiguous-transition fallback
- temporal split handling
- stale-transition rejection
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

Real external radio streams have been used throughout development to exercise
the production workflow.

Earlier in the research cycle, the frozen multi-signal acoustic selector was
also evaluated prospectively against untouched Corpus E.

Excluding three pre-declared session-level exclusions, that evaluation scored
15 transitions with:

- MAE: 0.562 s
- median absolute error: 0.200 s
- 12/15 transitions within 1.00 s
- maximum error: 2.375 s

No tuning was performed from the Corpus E results.

See
[`../ripper-research-closeout-2026-09-28.md`](../ripper-research-closeout-2026-09-28.md)
for that frozen evaluation record and artifact hashes.

Subsequent live development exercised the complete routed production
architecture against real ICY streams.

That validation covered:

- continuous external radio ingestion
- ICY-derived metadata and timeline observations
- deferred acoustic analysis
- multi-signal acoustic candidate selection
- local discontinuity and fade evidence
- incoming-boundary analysis
- bounded-insert discovery
- transition routing
- dedicated transition specialists
- temporal split decisions
- encoded-frame alignment
- finalized track materialization
- stale-transition protection
- repeated track transitions during continuous live sessions

The documented `brynse-radio` workflow has been executed end to end against real
radio sources.

Generated track files were manually listened to and verified during the live
validation runs.

This establishes full live validation of the operational path from stream input
through finalized track output.

It does not claim that every possible transition is perceptually perfect.

Some difficult transitions may still produce slightly early or late boundaries,
or retain residual inserted audio. Those cases are future specialist-refinement
work rather than evidence that the end-to-end execution path is incomplete.

Public radio streams are external dependencies and may change or disappear, so
they are not requirements of the normal repository gate.

## What this use case demonstrates

This workflow demonstrates more than radio ripping.

It proves that the generic engine can support a domain integration where:

1. a continuous external source is consumed incrementally;
2. domain-specific signals arrive alongside the media stream;
3. those signals are translated into generic boundary candidates and evidence;
4. different transition phenomena can be identified without moving domain
   semantics into the core;
5. transition ownership and temporal reasoning can remain separate
   responsibilities;
6. phenomenon-specific specialists can produce different temporal geometries
   through a common `TemporalSplitDecision` contract;
7. outgoing and incoming boundaries can represent different temporal meanings;
8. only the required working source window is retained;
9. completed bounded segments are materialized before the source ends; and
10. domain-specific output is produced without moving domain semantics into the
    core.

The live radio integration therefore exercises the complete architectural chain:

    domain semantics
        ↓
    generic boundary candidates
        ↓
    acoustic and temporal evidence
        ↓
    transition observation
        ↓
    Transition Router
        ↓
    transition specialist
        ↓
    TemporalSplitDecision
        ↓
    aligned segment boundaries
        ↓
    bounded materialization
        ↓
    domain-specific output

The architecture is exercised by a real, automated, and live-validated domain
workflow rather than existing only as a generic abstraction.
