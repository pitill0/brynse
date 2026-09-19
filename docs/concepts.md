# Core concepts

FluxTuner Ripper is a boundary-driven continuous-stream segmentation engine.

This document defines the core vocabulary used by the engine independently of
any source-specific integration.

The goal is to keep the domain model stable even when new sources, media types,
boundary providers, or integrations are added.

For the behavioral guarantees associated with these concepts, see
[`guarantees.md`](guarantees.md).

## Source

A **Source** is the origin of the continuous byte stream consumed by the engine.

A source is responsible for providing incremental access to encoded data and
transport-level metadata where relevant.

Examples include:

- a local file
- stdin
- an HTTP stream
- an application-provided byte stream
- a future non-radio continuous source

The source does not define segmentation semantics.

In the current implementation, `StreamSource` is the generic source abstraction.

## Ingestion

**Ingestion** is the process of incrementally consuming data from a source and
establishing the low-level timeline required by the segmentation engine.

Depending on the current media domain, ingestion may include:

- incremental reads
- encoded-frame parsing
- byte-offset tracking
- timestamp tracking
- mapping encoded offsets to media time

Ingestion answers questions such as:

> What data has arrived, where is it in the encoded stream, and when does it
> occur on the media timeline?

It does not answer:

> What does this part of the stream mean?

and it does not decide where domain-specific segments should begin or end.

## Retention

**Retention** is the bounded preservation of source data that may still be
required for analysis or future segment materialization.

Continuous streams cannot be assumed to fit in memory and may have no natural
end.

The engine therefore retains only the working data required by the active
segmentation horizon.

Retention may involve different storage tiers.

The current audio implementation uses:

- a bounded in-memory ring for analysis-sensitive data
- a reclaimable disk spool for encoded data that may still be needed for
  materialization

Retention is a core architectural concern, not an incidental implementation
detail.

## Boundary signal

A **Boundary Signal** is any observation that suggests a meaningful transition
may exist at or near a particular point in the stream.

A signal may come from:

- elapsed time
- a human-provided timestamp
- an external application
- acoustic analysis
- metadata
- a domain-specific integration
- an AI or perception system
- a physical or software sensor

Boundary signals are observations, not final segmentation decisions.

They may be exact, approximate, noisy, incomplete, or contradictory.

## Boundary provider

A **Boundary Provider** is a component that produces boundary candidates from one
or more boundary signals.

A provider answers:

> Where might a segmentation boundary exist?

Current generic examples include:

- fixed-interval providers
- manual boundary providers
- external JSON/JSONL boundary providers

A provider may be deterministic, algorithmic, externally driven, or backed by
another system.

A provider should not need to control ingestion, retention, materialization, or
output persistence.

## Boundary candidate

A **Boundary Candidate** is a proposed segmentation point.

It represents evidence that a boundary may exist, but it is not necessarily the
final boundary used for segmentation.

A candidate may contain information such as:

- an approximate timestamp
- a source or provider identity
- an encoded reference offset
- associated confidence or evidence

A candidate can originate from a generic provider or from a domain integration
that adapts its own semantics into the generic boundary model.

The distinction between candidate and resolved boundary is fundamental:

> Candidate means proposed. Resolved means accepted for segmentation.

## Boundary resolver

A **Boundary Resolver** converts one or more boundary candidates into usable
segmentation decisions.

Resolution may involve:

- matching a semantic or approximate candidate to nearby timeline evidence
- aligning with encoded-frame boundaries
- consulting acoustic evidence
- applying temporal policies
- rejecting unsuitable candidates
- selecting between competing alternatives

The resolver answers:

> Given the available evidence, where should the segmentation boundary actually
> be placed?

Resolution is separate from candidate generation.

## Resolved boundary

A **Resolved Boundary** is a segmentation point accepted by the engine after
resolution.

A resolved boundary is suitable for downstream segment planning.

It represents an operational decision rather than merely an observation.

The engine may retain additional evidence about how the decision was reached,
but downstream segmentation should not need to understand the domain-specific
meaning of the original signal.

## Segment plan

A **Segment Plan** describes which retained source range belongs to a segment.

It is a planning object rather than the final persisted output.

A plan may encode information such as:

- start and end offsets
- start and end timestamps
- codec-aligned cut positions
- overlap or crossfade treatment
- final-tail policy
- output-relevant metadata

Separating planning from materialization allows the engine to decide what should
be emitted before deciding how or where it should be persisted.

## Segment

A **Segment** is the core representation of one bounded portion of the continuous
source.

`Segment` is deliberately domain-neutral.

A segment is not inherently:

- a song
- a track
- a programme
- a speaker turn
- an incident
- a scene
- an alarm
- an object detection

Those meanings belong to integrations or consuming applications.

The engine's responsibility is to preserve the bounded portion of the source
identified by resolved boundaries.

## Materialization

**Materialization** is the process of turning a segment plan and retained source
data into concrete bounded output.

For the current audio domain this may include:

- copying encoded ranges
- finalizing MP3 output
- converting AAC/ADTS into M4A
- writing temporary output safely
- atomically publishing completed files

Materialization is downstream from segmentation decisions.

It should not decide the semantic meaning of a segment.

## Materialized segment

A **Materialized Segment** is the concrete output produced from a segment plan.

Examples may include:

- a media file
- a byte sequence
- an application-owned object
- a future non-file output representation

This concept is distinct from the abstract `Segment`.

The abstract segment describes the bounded source portion.

The materialized segment is one concrete representation of that portion.

## Sink

A **Sink** receives completed materialized segments.

A sink may:

- write files
- publish objects
- enqueue work
- forward data to another application
- store results in an external system

A sink is downstream of segmentation and should not control the meaning of
boundaries.

The engine should be able to evolve sink behavior independently from boundary
generation and resolution.

## Policy

A **Policy** is a deterministic rule that influences how the engine converts
evidence into segmentation behavior.

Examples include:

- temporal split policy
- final-tail handling
- crossfade-aware planning
- transient exclusion
- candidate acceptance thresholds

A policy is different from a provider.

A provider proposes possible boundaries.

A policy constrains or selects how those proposals are interpreted and applied.

## Domain integration

A **Domain Integration** adapts source-specific semantics into the generic engine
contracts.

An integration may understand concepts that the core deliberately does not.

For example, the first-party radio integration understands:

- ICY metadata
- StreamTitle
- radio track state
- transient non-track intervals
- track-oriented output naming

It adapts those concepts into generic boundary and segment contracts.

The dependency direction must remain:

```text
domain integration
       ↓
      core
```

The core must not import or depend on domain-specific concepts.

## Core vocabulary relationships

The main conceptual flow is:

```text
SOURCE
  ↓
INGESTION
  ↓
RETENTION
  ↓
BOUNDARY SIGNALS
  ↓
BOUNDARY PROVIDERS
  ↓
BOUNDARY CANDIDATES
  ↓
BOUNDARY RESOLVER
  ↓
RESOLVED BOUNDARIES
  ↓
SEGMENT PLANS
  ↓
MATERIALIZATION
  ↓
MATERIALIZED SEGMENTS
  ↓
SINK
```

A domain integration may feed boundary candidates into this flow, but it does
not replace the generic engine model.

## Important distinctions

### Boundary signal vs boundary candidate

A boundary signal is raw evidence.

A boundary candidate is a normalized proposal expressed through the engine's
boundary contract.

### Boundary candidate vs resolved boundary

A boundary candidate is provisional.

A resolved boundary is accepted for segmentation.

### Segment plan vs segment

A segment plan describes how a bounded range should be produced.

A segment is the engine-level representation of that bounded portion of the
source.

### Segment vs materialized segment

A segment is abstract and domain-neutral.

A materialized segment is a concrete representation produced from retained
source data.

### Provider vs resolver

A provider proposes where a boundary may exist.

A resolver decides where the segmentation boundary should actually be placed.

### Core semantics vs domain semantics

The core understands boundaries, ranges, retention, planning, materialization,
and segments.

Domain integrations understand meanings such as tracks, programmes, speakers,
incidents, scenes, or application-specific events.

This separation is intentional and should remain stable as the project evolves.
