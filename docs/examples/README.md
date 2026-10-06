# Examples and validated use cases

This directory contains reproducible examples of Brynse in use.

Examples are classified by evidence level.

## Evidence levels

### Implemented

The execution path exists in the current codebase.

### Automated

The behavior is covered by automated repository tests.

### Live validated

The complete workflow has also been exercised against a real external source or
environment.

Live validation is complementary to automated testing. It is not a substitute
for deterministic repository tests.

## Validated use cases

| Use case | Implemented | Automated | Live validation |
| --- | --- | --- | --- |
| [Radio stream to tracks](live-radio-track-ripping.md) | Yes | Yes | Yes |

Additional examples should only be added to this table at the evidence level
actually demonstrated by the project.

Conceptual extension ideas belong in the architecture or extension
documentation until an executable and reproducible path exists.

For the radio use case, the documented `brynse-radio` workflow has been exercised
end to end against real external ICY streams.

Live validation has covered:

- real stream ingestion
- ICY metadata tracking
- multi-signal acoustic evidence generation
- transition routing
- dedicated transition specialists
- temporal split application
- finalized track materialization
- manual listening and verification of generated tracks

This validates the complete operational path from live stream input to finalized
track output.

Live validation does not imply perfect perceptual boundary placement for every
possible transition.

Residual difficult cases may still exist and remain candidates for future
specialist refinement without invalidating the demonstrated end-to-end workflow.

A use case should only be promoted to full live validation after the exact
documented workflow has been executed against a real source and its resulting
output has been verified.
