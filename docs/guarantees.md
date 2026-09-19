# Engine guarantees

This document describes the behavioral guarantees and explicit non-guarantees of
the generic segmentation engine.

These guarantees define what consumers, integrations, and future extension
points may rely on independently of radio-specific behavior.

For the contracts that extension implementations should follow, see
[`extensions.md`](extensions.md).

## Incremental processing

The engine processes continuous input incrementally.

Consumers should not need to provide the complete source before segmentation can
begin.

The engine is designed around repeated bounded reads from a `StreamSource`
rather than whole-input loading.

This means that:

- ingestion can begin before the source has finished
- boundary processing can occur while additional data continues to arrive
- completed segments can be materialized before the source ends
- live or effectively unbounded sources are supported by the architecture

Incremental processing does not imply that every boundary can be finalized
immediately.

A boundary may require additional future evidence before it can be resolved or
safely materialized.

## Bounded in-memory retention

The engine does not require the complete encoded stream to remain in memory.

Memory used for active stream analysis is intentionally bounded.

For the current audio implementation, the in-memory ring contains only the
working encoded window required for analysis and timeline operations.

Consumers may rely on the architectural invariant that continuous operation must
not require memory growth proportional to total stream duration.

This guarantee concerns the engine's retained working data.

It does not imply that every provider, integration, external service, or sink
used alongside the engine is itself memory-bounded.

## Reclaimable encoded retention

Encoded source data that may still be required for future segment
materialization can be retained outside the bounded analysis ring.

The current implementation uses a reclaimable disk spool.

Retained encoded data remains available only while it may still contribute to a
future materialized segment.

Once the engine determines that an encoded range is no longer required by any
active or future segment plan, that range may be reclaimed.

Consumers must therefore not treat retention storage as permanent archival
storage.

The retention layer exists to support correct bounded segmentation, not long-term
preservation.

## Retention lifetime

The lifetime of retained data is driven by segmentation requirements rather than
source lifetime.

Conceptually:

```text
source bytes arrive
      ↓
bytes become retained working data
      ↓
boundary decisions advance
      ↓
segments become materializable
      ↓
older retained bytes become unnecessary
      ↓
those bytes may be reclaimed
```

A consumer that requires permanent access to completed source ranges must
persist materialized segments through an appropriate sink.

## Materialization preconditions

A segment can only be materialized while all source data required by its segment
plan remains available.

The engine therefore coordinates retention and segment planning so that required
encoded ranges are not reclaimed prematurely.

Materialization requires:

- a valid segment plan
- resolved boundaries sufficient to define the required range
- retained source data covering that range
- a suitable materializer or finalizer for the current media representation

If any of these conditions cannot be satisfied, successful materialization is
not guaranteed.

## Planned vs materialized segments

A planned segment and a materialized segment are different states.

A segment plan describes what bounded source range should be emitted.

A materialized segment is the concrete output produced from that plan.

A valid plan does not by itself guarantee successful persistence.

Materialization may still fail because of:

- I/O errors
- finalization errors
- unsupported media conditions
- sink failures
- unavailable retained data
- operational interruption

Consumers should therefore treat planning and persistence as separate stages.

## Segment completeness

A completed segment is materialized only when the engine has enough information
to determine its required bounded range.

For continuous sources, the current final segment may remain incomplete while
the stream continues.

The engine must not invent an artificial end boundary solely to force
materialization of an otherwise incomplete segment.

A consuming interface may expose explicit end-of-stream or final-tail behavior,
but that behavior is a policy decision rather than a universal semantic
guarantee.

## Boundary resolution

Boundary candidates are provisional.

The existence of a candidate does not guarantee that the candidate will become a
resolved boundary.

A resolver may:

- accept a candidate
- align it to nearby evidence
- replace its effective position
- reject it
- defer resolution until sufficient evidence exists

Consumers should rely on resolved boundaries for segmentation decisions rather
than assuming that candidate timestamps are final.

## Provider-dependent behavior

The engine guarantees the boundary-processing contract, not the quality of every
boundary provider.

Different providers may have different characteristics.

For example:

- a fixed-interval provider may be deterministic
- a manual provider may reflect exact user input
- an acoustic provider may depend on signal characteristics
- an external AI provider may be probabilistic or approximate

The engine does not convert uncertain provider semantics into certainty.

It normalizes and resolves candidate evidence according to the configured
resolver and policies.

## Determinism

Given identical:

- source bytes
- source timeline
- provider outputs
- resolver configuration
- policies
- materialization configuration

the generic deterministic parts of the engine should produce equivalent
segmentation decisions.

This does not imply global reproducibility when one or more external components
are nondeterministic.

Examples include:

- remote AI services
- stochastic models
- changing external metadata
- network-dependent event sources

Determinism therefore applies only to the inputs and components that are
themselves deterministic.

## Source semantics

The core does not guarantee semantic interpretation of source content.

It does not inherently know whether a segment represents:

- a song
- a programme
- a speaker
- an incident
- a scene
- an alarm
- an object
- any other domain concept

Such interpretation belongs to domain integrations or external intelligence.

The core guarantees generic segmentation behavior around boundaries and retained
source ranges.

## Media preservation

The engine aims to preserve source media without unnecessary transformation.

Depending on codec and output requirements, materialization may involve:

- copying encoded ranges
- frame-aligned extraction
- container finalization
- minimally necessary transformation

The engine does not promise bit-for-bit identity for every output format.

Containerization, framing, metadata handling, or finalization may legitimately
change the byte representation while preserving the intended media content.

## Failure semantics

Operational failures are not silently converted into successful segmentation.

Interfaces should expose failures through explicit exceptions, result types, or
structured machine errors.

Current machine-oriented error categories distinguish between:

- validation failures
- operational failures
- internal failures

Examples of operational failures include:

- source I/O errors
- output I/O errors
- finalization failures
- persistence failures

A failure to materialize a segment must not be reported as successful output.

## Atomic persistence

Where atomic file publication is supported by the current output path, incomplete
temporary output should not be exposed as a successfully finalized segment.

Atomic persistence protects consumers from observing a partially written final
file as if it were complete.

This guarantee applies to output paths that explicitly use the engine's atomic
persistence mechanisms.

It does not automatically apply to arbitrary external sinks implemented by
consumers.

## Sink responsibility

Once a materialized segment is handed to a sink, long-term storage guarantees
belong to that sink.

The segmentation engine does not guarantee:

- permanent storage
- replication
- indexing
- backup
- remote durability
- searchability
- lifecycle management

Those responsibilities belong to consuming systems.

## Integration isolation

Domain integrations may add semantics and source-specific behavior, but they must
not weaken the generic core boundary.

The intended dependency direction is:

```text
domain integration
       ↓
      core
```

The core must remain usable without importing or understanding domain-specific
concepts.

This is both an architectural rule and a behavioral guarantee for generic
consumers.

## External intelligence

The engine may consume boundary signals from external intelligence such as AI,
ASR, diarization, scene detection, or application-specific event systems.

The engine does not guarantee the correctness of those external interpretations.

Its responsibility begins when observations are adapted into boundary
candidates.

This separation allows:

- intelligence systems to interpret the stream
- the segmentation engine to preserve and materialize bounded source evidence

without requiring the engine itself to become an inference platform.

## Explicit non-guarantees

The engine does not guarantee:

- semantic correctness of domain interpretations
- perfect boundary detection
- permanent retention of source data
- permanent storage of materialized output
- whole-stream in-memory availability
- bit-for-bit output identity across all codecs and containers
- deterministic behavior from nondeterministic external providers
- successful materialization after required retained data has been reclaimed
- interpretation of tracks, speakers, incidents, scenes, or other domain
  concepts by the core
- compatibility with arbitrary future providers without adapting them to the
  engine's boundary contracts

These non-guarantees are intentional.

They keep the engine focused on bounded, incremental, boundary-driven
segmentation rather than turning it into a general-purpose inference, archival,
or workflow platform.

## Summary

The core guarantees are:

1. Continuous sources are processed incrementally.
2. In-memory working retention remains bounded.
3. Encoded retention exists only as long as required for active segmentation.
4. Required retained data is preserved until dependent materialization is no
   longer needed.
5. Boundary candidates remain provisional until resolved.
6. Segment planning and materialization are distinct stages.
7. Materialization failures are surfaced rather than treated as success.
8. Domain semantics remain outside the generic core.
9. Integrations depend on the core; the core does not depend on integrations.
10. Permanent storage and interpretation belong to consumers and external
    systems.
